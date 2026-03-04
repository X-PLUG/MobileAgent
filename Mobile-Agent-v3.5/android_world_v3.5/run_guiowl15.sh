current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG="log_ma3_"$current_time".log"

MODEL_NAME="gui_owl"
MODEL="your model name"
API_KEY="your api key"
BASE_URL="your base url"
TRAJ_OUTPUT_PATH="traj_"$current_time

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=8554 \
  --console_port=5554 2>&1 | tee "$LOG"