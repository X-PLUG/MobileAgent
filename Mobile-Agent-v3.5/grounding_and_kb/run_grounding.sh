export MODEL_PATH="your model path"
export DS_PATH="your benchmark dir which should contain images/ and, depending on the dataset, either annotations/ or annotation.json, see benchmark class __init__"
export SAVE_PATH="save path for grounding evaluation results"
export EVAL_TYPE="grounding benchmark type, i.e., ssp, spv2, osg, mmbench_l2"
torchrun --nproc_per_node=8 --nnodes=1 eval_grounding_benchmarks.py \
    --ds_path $DS_PATH \
    --save_path $SAVE_PATH \
    --eval_benchmark_type $EVAL_TYPE \
    --model_path $MODEL_PATH