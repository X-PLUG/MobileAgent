export MODEL_PATH="your model path"
export DS_PATH="your benchmark dir which should contain Knowledge_Bench/ ,Image/ and AnnotateImage/, following official benchmark."
export SAVE_PATH="save path for evaluation results"
export EVAL_TYPE="knowledge benchmark type, i.e., kb and kb-thinking"
torchrun --nproc_per_node=8 --nnodes=1 eval_gui_knowledge_benchmark.py \
    --ds_path $DS_PATH \
    --save_path $SAVE_PATH \
    --eval_benchmark_type $EVAL_TYPE \
    --model_path $MODEL_PATH