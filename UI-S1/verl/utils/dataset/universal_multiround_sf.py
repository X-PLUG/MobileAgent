

import copy
import logging
import os
import random
import re
import time
import traceback
import uuid
from collections import defaultdict
from typing import Any, List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from x.data.agent.json import JsonFormat
from x.data.text import parse_tags
from x.io import JsonWrap
from x.parallel.parallel_task import ParallelTask
from x.qwen.data_format import slim_messages

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask


class QwenMessages2Inputs():
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig, processor: Any | None = None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
      
        self.max_pixels = 12800*28*28
        self.min_pixels = 4*28*28
        self.num_image_limit = config.get("num_image_limit", 2)

        self.max_prompt_length = config.get("max_prompt_length", 32768)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)

    def __call__(self, state):
        messages = state['messages']
        check_options = state['check_options']
        row_dict = {}
        messages = slim_messages(messages, num_image_limit=self.num_image_limit)
        last_image_ele = None
        for msg in messages:
            for content in msg['content']:
                # Very Important
                if 'image' in content:
                    if 'min_pixels' not in content: # TODO fix bug, respect to the resized height
                        content['min_pixels'] = self.min_pixels
                    if 'max_pixels' not in content:
                        content['max_pixels'] = self.max_pixels
                    last_image_ele = content
        assert messages[-1]['role'] == 'user'

        assert self.processor is not None
        from verl.utils.dataset.vision_utils import (process_image,
                                                     process_video)

        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}
        image_inputs, video_inputs = process_vision_info(messages)
        assert 0 < len(image_inputs)<=self.num_image_limit
        
        width, height = last_image_ele['width'], last_image_ele['height']
        resized_width, resized_height = image_inputs[-1].size
        
        model_inputs = self.processor(text=[raw_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        if image_inputs is not None:
            assert sum(round(_.size[0]*_.size[1]/(28*28)) for _ in image_inputs) == (model_inputs['input_ids'] == 151655).sum()

        multi_modal_data = {
            'image': image_inputs
        }
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)


        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        assert self.processor.image_processor.__class__.__name__ != "Qwen2_5VLImageProcessor"
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        row_dict['reward_model'] = {
            "style": "rule",
            "ground_truth": check_options
        }

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = message_translate(messages, to_format="openai")

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
      
        row_dict["index"] = index
        if 'extra_info' not in row_dict:
            row_dict['extra_info'] = {}
        row_dict['extra_info']['resized_width'] = resized_width
        row_dict['extra_info']['resized_height'] = resized_height
        row_dict['extra_info']['width'] = width
        row_dict['extra_info']['height'] = height

        return row_dict



class StdTrajectory():
    def __init__(self, line,actions_only) -> None:
        self.line = line[()]
        self.num_steps = len(self.line['steps'])
        from x.data.agent.space.std_space import RAW_SPACE
        self.fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True,actions_only=actions_only)
        self.state = None

    def get_next(self, model_response):
        state = self.fm.gen_next_round(self.line, self.state, previous_model_response=model_response)
        if state is None:
            return "Finished"
        return state
class StdTrajectorySF():
    def __init__(self, line,actions_only) -> None:
        self.line = line[()]
        self.num_steps = len(self.line['steps'])
        from x.data.agent.json_self_fix import JsonFormatSF
        from x.data.agent.space.std_space import RAW_SPACE
        self.fm = JsonFormatSF(RAW_SPACE, add_thought=True, force_add_thought=True,actions_only=actions_only)
        self.state = None

    def get_next(self, model_response):
        state = self.fm.gen_next_round(self.line, self.state, previous_model_response=model_response)
        if state is None:
            return "Finished"
        return state
    
class MultiRoundGeneratorSF():
    def __init__(self, batch: DataProto, rollout_n, msg_man, patch_threshold=0,actions_only=None) -> None:
        self.rollout_n = rollout_n
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.non_tensor_batch["line"]))], dtype=object)
        
        repeat_batch = batch.repeat(repeat_times=self.rollout_n, interleave=True) # need set rollout kwargs to 1
        self.batch = repeat_batch
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(len(self.batch))], dtype=object)
        self.batch.non_tensor_batch["traj_uid"] = traj_uid
        # TODO replay buffer可以保存一个前缀 然后在这里恢复
        self.task_queue = [StdTrajectory(line,actions_only) for line in self.batch.non_tensor_batch["line"]]
        self.task_queue_sf = [StdTrajectorySF(line,actions_only) for line in self.batch.non_tensor_batch["line"]]
        self.finished = [False for i in range(len(self.task_queue))]
        self.current_response = [None for i in range(len(self.task_queue))]
        # self.current_response_sf = [None for i in range(len(self.task_queue))]
        self.error_num = [0 for i in range(len(self.task_queue))]
        self.msg_man = msg_man
        from x.data.agent.space.std_space import RAW_SPACE
        self.fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
        self.patch_threshold = patch_threshold
        print('Finish generator init')


    def _fetch_next(self, ptr):
        if self.finished[ptr]:
            return True, (None, None)
        current_gen = self.task_queue[ptr]
        current_response = self.current_response[ptr]
        state = current_gen.get_next(current_response)
        if state == "Finished":
            return True, ("Finished", state)
        row_dict = self.msg_man(state)
        row_dict['ptr'] = ptr
        return True, (row_dict, state)
        
    def _fetch_next_sf(self, ptr):
        if self.finished[ptr]:
            return True, (None, None)
        current_gen = self.task_queue_sf[ptr]
        current_response = self.current_response[ptr]
        state = current_gen.get_next(current_response)
        if state == "Finished":
            return True, ("Finished", state)
        row_dict = self.msg_man(state)
        row_dict['ptr'] = ptr
        return True, (row_dict, state)
    def fetch_batch(self):
        while True:
            batch = []
            batch_sf = []
            tasks = list(range(len(self.task_queue)))
            tasks_sf = list(range(len(self.task_queue_sf)))
            # input()
            mid_result = ParallelTask((tasks,), self._fetch_next, total=len(tasks), num_process=len(tasks), passing_indices=False, return_list=True).run_and_collect(tqdm_args={"disable": False})
            mid_result_sf = ParallelTask((tasks_sf,), self._fetch_next_sf, total=len(tasks_sf), num_process=len(tasks_sf), passing_indices=False, return_list=True).run_and_collect(tqdm_args={"disable": False})
            assert len(mid_result) == len(self.task_queue) and len(mid_result_sf) == len(self.task_queue_sf)
            for ptr, res in enumerate(mid_result):
                row_dict, state = res
                row_dict_sf, state_sf = mid_result_sf[ptr]
                if row_dict == None:
                    continue
                self.current_response[ptr]= None
                if row_dict == "Finished":
                    self.finished[ptr] = True
                else:
                    self.task_queue[ptr].state = state
                    self.task_queue_sf[ptr].state = state_sf
                    row_dict_sf['uid'] = row_dict['uid'] = self.batch.non_tensor_batch['uid'][ptr]
                    row_dict_sf['traj_uid'] = row_dict['traj_uid'] = self.batch.non_tensor_batch['traj_uid'][ptr]
                    row_dict_sf['step_id'] = row_dict['step_id'] = state['step_id']
                    row_dict['data_source'] = self.batch.non_tensor_batch['data_source'][ptr] if 'data_source' in self.batch.non_tensor_batch else "gui_traj_action_match"
                    row_dict['reward_model'] = {
                        "style": "rule",
                        "ground_truth": {
                            "check_options": state['check_options'],
                            'num_steps': self.task_queue[ptr].num_steps,
                            'thought': state['thought'],
                            }
                    }
                    batch.append(row_dict)
                    batch_sf.append(row_dict_sf)
            if len(batch) == 0:
                break
            yield collate_fn(batch), collate_fn(batch_sf)

        # batch = []
        # for item in self._fetch_next():
        #     batch.append(item)
        #     if len(batch) == self.loader_size:
        #         yield collate_fn(batch)
        #         batch = []
        # if len(batch):
        #     yield collate_fn(batch)
    def apply_response(self, batch , batch_sf):
        failed_num = 0
        for ptr, response, extract_match, reward_model,extra_info,response_sf  in zip(batch.non_tensor_batch['ptr'], batch.batch['responses'], batch.non_tensor_batch['extract_match'], batch.non_tensor_batch['reward_model'], batch.non_tensor_batch['extra_info'],batch_sf.batch['responses']):
            response_text = self.msg_man.tokenizer.decode(response)
            response_text_sf = self.msg_man.tokenizer.decode(response_sf)            
            self.current_response[ptr] = response_text
            if not extract_match:
                failed_num += 1
                if self.patch_threshold > self.error_num[ptr] or self.patch_threshold == -1:
                    step = {}
                    step['action_content'] = reward_model['ground_truth']['check_options']
                    keys_to_remove = ['bbox', 'candidate_bbox','annotation','thought']
                    for key in keys_to_remove:
                        step['action_content'].pop(key, None)
                    print("reward_model['ground_truth']",reward_model['ground_truth'])
                    
                    step['thought'] = parse_tags(response_text_sf,['think'])['think'] if parse_tags(response_text_sf,['think'])['think']!=None else ""
                    ground_truth_response = self.fm.format_response(step,extra_info) # resize coordinate
                    print("patch_threshold===========:",ground_truth_response)
                    # ground_truth = reward_model['ground_truth']['check_options']
                    self.current_response[ptr] = ground_truth_response
                    self.error_num[ptr] += 1
                else:
                    self.finished[ptr] = True
                    
        return failed_num
            
            
                


def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line

if __name__ == "__main__":
    from x.io import read_json
    lines = read_json("/datasets/GUI_Data/std/std_lines/androidcontrol_sft_fc_open.std.omniparser.jsonl")
    batch_lines = lines[:16]

    msg_man = QwenMessages2Inputs(
        hf_tokenizer("checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"),
        {},
        hf_processor("checkpoints/Qwen/Qwen2.5-VL-7B-Instruct")
    )

    batch_dict = collate_fn([
        {'line': np.array(fix_line(line), dtype=object)}
        for line in batch_lines])
    batch = DataProto.from_single_dict(batch_dict)
    mr_gen = MultiRoundGenerator(batch, rollout_n=5, msg_man=msg_man)
    for sub_batch in mr_gen.fetch_batch():
        print(sub_batch)
        sub_batch = DataProto.from_single_dict(sub_batch)
        for ptr in sub_batch.non_tensor_batch['ptr']:
            mr_gen.current_response[ptr] = '<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [670, 2060]}}\n</tool_call>'
            ## calculate reward
