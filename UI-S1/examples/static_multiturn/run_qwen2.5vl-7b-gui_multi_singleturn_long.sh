# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
PROJECT_DIR="$(pwd)"

PROJECT_NAME="online_gui"
CONFIG_PATH="$PROJECT_DIR/examples/static_multiturn/config"

## Exp Setting
TRACE_SAMPLE_K=16
BATCH_SIZE=16
N_EPOCH=256
EXPERIMENT_NAME='v11.1.3_grpo_trspk_'${TRACE_SAMPLE_K}_bs_${BATCH_SIZE}_ep_${N_EPOCH}
##
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='multi_singleturn' \
    data.train_batch_size=$BATCH_SIZE \
    data.trace_sample_k=$TRACE_SAMPLE_K \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/nas-wulanchabu/jiabo.ye/checkpoints/2_5_gui/to_hf/nlp_v11.1.3_mix_single_new_fix_ckpt-Qwen2-5vl-7b-mp4-pp-lr-1-lr-3.0e-6-minlr--iters-1.0e-7-iters-1148-warmup-12-bs-256-gpus-2-seqlen-32768-iter_0001148-hf \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=fake_multiturn \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    data.train_files=/datasets/GUI_Data/agent_rl/android_world/grpo/grpo_multi_single_turn.0511.jsonl \
    data.val_files=/datasets/GUI_Data/agent_rl/android_world/grpo/grpo_multi_single_turn.0511.jsonl \
    trainer.total_epochs=$N_EPOCH $@
