# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from uis1 import core_uis1
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.checkpoint_manager import (BaseCheckpointManager,
                                                      find_latest_ckpt_path)
from verl.utils.dataset.universal_multiround import (MultiRoundGenerator,
                                                     QwenMessages2Inputs)
from verl.utils.debug.performance import _timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import masked_mean, torch_to_numpy
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          _timer, apply_kl_penalty,
                                          compute_advantage,
                                          compute_response_mask)


class RayTrajDAPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msg_man = QwenMessages2Inputs(
            hf_tokenizer(self.config.actor_rollout_ref.model.path),
            {},
            hf_processor(self.config.actor_rollout_ref.model.path)
        )

        print("self.config.ppo_mini_batch_size:", self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        total_num = 0
        failed_num = 0
        for batch_dict in self.val_dataloader:
            # print(batch_dict)
            # print(len(batch_dict['line']))
            total_num += len(batch_dict['line'])
            traj_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # print("val_rollout_n",self.config.actor_rollout_ref.rollout.val_kwargs.n)
            mr_gen = MultiRoundGenerator(traj_batch, rollout_n=self.config.actor_rollout_ref.rollout.val_kwargs.n, msg_man=self.msg_man,actions_only=self.config.algorithm.actions_only)
            for _step,test_batch in enumerate(tqdm(mr_gen.fetch_batch(), desc="Rollout validation batched steps")):
                test_batch = DataProto.from_single_dict(test_batch)
                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}
                
                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # pad to be divisible by dp_size
                print("self.actor_rollout_wg.world_size",self.actor_rollout_wg.world_size)
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
        
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    self.async_rollout_manager.wake_up()
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    self.async_rollout_manager.sleep()

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print("validation generation end")

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                reward_extra_infos_dict["reward"].extend(scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                    test_batch.non_tensor_batch.update({k: np.array(v) for k, v in result["reward_extra_info"].items()})
                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
                failed_num += mr_gen.apply_response(test_batch)
        # print("data_source_lst",data_source_lst)
        # print("failed_num",failed_num)
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        # dump generations
        # print("sample_inputs",sample_inputs)
        # print("sample_outputs",sample_outputs)
        # print("sample_scores",sample_scores)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
        # print("reward_extra_infos_dict", reward_extra_infos_dict)
        # print("sample_scores",len(sample_scores),sample_scores)
        # print("total_num",total_num)
        
        data_sources = np.concatenate(data_source_lst, axis=0)
        # metric_dict
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        metric_dict["val-core/gui_traj_action_match/task_acc"] = 1 - failed_num / total_num
        metric_dict["val-aux/gui_traj_action_match/reward/mean"] = np.mean(sample_scores)
        metric_dict["val-aux/gui_traj_action_match/reward/std"] = np.std(sample_scores)
        metric_dict["val-aux/gui_traj_action_match/extract_match/mean"] = sum(reward_extra_infos_dict['extract_match']) / len(reward_extra_infos_dict['extract_match'])
        metric_dict["val-aux/gui_traj_action_match/type_match/mean"] = sum(reward_extra_infos_dict['type_match']) / len(reward_extra_infos_dict['type_match'])
        metric_dict["val-aux/gui_traj_action_match/format_score/mean"] = np.mean(reward_extra_infos_dict['format_score'])
        # for data_source, var2metric2val in data_src2var2metric2val.items():
        #     core_var = "acc" if "acc" in var2metric2val else "reward"
        #     for var_name, metric2val in var2metric2val.items():
        #         n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
        #         for metric_name, metric_val in metric2val.items():
        #             if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
        #                 metric_sec = "val-core"
        #             else:
        #                 metric_sec = "val-aux"
        #             pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
        #             metric_dict[pfx] = metric_val

        return metric_dict      
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        print("self.actor_rollout_wg.world_size",self.actor_rollout_wg.world_size)
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_step_num = 0
        error_step_num = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                step_batch_list = []
                num_gen_batches += 1
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        traj_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        mr_gen = MultiRoundGenerator(traj_batch, rollout_n=self.config.actor_rollout_ref.rollout.n, msg_man=self.msg_man, patch_threshold=self.config.algorithm.patch_threshold,actions_only=self.config.algorithm.actions_only,hint=self.config.algorithm.hint)
                        for _step, step_batch in enumerate(tqdm(mr_gen.fetch_batch(), desc="Rollout batched steps")):
                        # pop those keys for generation
                            total_step_num += len(step_batch["input_ids"])
                            step_batch = DataProto.from_single_dict(step_batch)
                            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                            if "multi_modal_data" in step_batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("multi_modal_data")
                            if "raw_prompt" in step_batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("raw_prompt")
                            if "tools_kwargs" in step_batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("tools_kwargs")
                            gen_batch = step_batch.pop(
                                batch_keys=batch_keys_to_pop,
                                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                            )
                            is_last_step = self.global_steps >= self.total_training_steps

                       
                            gen_batch.meta_info['n'] = 1
                            # padding
                            gen_size = len(gen_batch)
                            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_padded) # TODO do not need pad
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_padded)
                                self.async_rollout_manager.sleep()
                            gen_batch_output = gen_batch_output.slice(0, gen_size)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                           
                            step_batch = step_batch.union(gen_batch_output)
                            reward_tensor, reward_extra_infos_dict = compute_reward(step_batch, self.reward_fn) # batch_size, 512
                            step_batch.batch["token_level_scores"] = reward_tensor
                            
                            if reward_extra_infos_dict:
                                step_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                step_batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                step_batch.batch["token_level_rewards"] = step_batch.batch["token_level_scores"]

                            print(reward_extra_infos_dict['score'])
                            step_batch.non_tensor_batch['rewards'] = torch_to_numpy(torch.tensor(reward_extra_infos_dict['score']), is_object=True)
                            # # step-level rule-based reward
                            # if len(rewards.shape) == 2:
                            #     rewards = rewards.squeeze(1)

                            step_batch_list.append(step_batch) # if RAM OOM, offload it to disk! and reload
                            # set response
                            error_step_num += mr_gen.apply_response(step_batch)
                    new_batch = DataProto.concat(step_batch_list)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.UIS1:
                        step_rewards_tensor = core_uis1.compute_step_discounted_returns(
                            batch=new_batch,
                            gamma=self.config.algorithm.gamma
                        )
                        new_batch.batch['step_rewards'] = step_rewards_tensor
                    
                    # ======= DAPO =========
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        else:
                            new_batch.non_tensor_batch["seq_future_reward"] = new_batch.batch['step_rewards'].numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        
                        if metric_name == "seq_reward" or metric_name == "seq_final_reward":
                            prompt_uid2max_step_id = defaultdict(int)

                            for uid, step_id in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch["step_id"]):
                                prompt_uid2max_step_id[uid] = max(prompt_uid2max_step_id[uid], step_id)
                            for uid, metric_val, step_id in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], new_batch.non_tensor_batch["step_id"]):

                                if step_id == prompt_uid2max_step_id[uid]:
                                    prompt_uid2metric_vals[uid].append(metric_val)
                        else:
                            for uid, metric_val, step_id in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], new_batch.non_tensor_batch["step_id"]):
                                if step_id == 0:
                                    prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                        print(f"prompt_uid2metric_std: {prompt_uid2metric_std}")
                        
                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > self.config.algorithm.filter_groups.std_threshold or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            kept_traj_idxs = []
                            seen_uid_set = set()
                            for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]): #TODO：选std最高的那几个
                                if traj_from_prompt_uid in seen_uid_set:
                                    kept_traj_idxs.append(idx)
                                elif len(seen_uid_set) < prompt_bsz:
                                    seen_uid_set.add(traj_from_prompt_uid)
                                    kept_traj_idxs.append(idx)
                            # traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[kept_traj_idxs]

                    # ======= update batch =========
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        
                        raw_batch_size = len(batch)
                        pad_batch, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(pad_batch)
                        old_log_prob = old_log_prob.slice(0, raw_batch_size)

                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            raw_batch_size = len(batch)
                            
                            if not self.ref_in_actor:
                                pad_batch, _ = pad_dataproto_to_divisor(batch, self.ref_policy_wg.world_size)
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(pad_batch)
                            else:
                                pad_batch, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(pad_batch)
                            ref_log_prob = ref_log_prob.slice(0, raw_batch_size)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            step_advantage_w=self.config.algorithm.uis1.step_advantage_w,
                            episode_advantage_w=self.config.algorithm.uis1.episode_advantage_w,
                            uis1_mode=self.config.algorithm.uis1.mode,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            raw_batch_size = len(batch)
                            if self.config.actor_rollout_ref.actor.use_fixed_num_mini_batches:
                                pad_batch, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size*self.config.actor_rollout_ref.actor.fixed_num_mini_batches*self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
                            else:
                                pad_batch, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size*self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
                            print('pad_batch size:', len(pad_batch))
                            actor_output = self.actor_rollout_wg.update_actor(pad_batch)
                            actor_output = actor_output.slice(0, raw_batch_size)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    print("inputs",inputs,"outputs", outputs,"scores", scores)
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "critic/step_success_rate": 1-error_step_num/total_step_num,
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return