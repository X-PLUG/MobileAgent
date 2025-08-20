
TASK_JSON_PATH="data/custom_tasks_example.json"
SETTING="evolution"
python run.py \
    --run_name "example-$SETTING" \
    --setting $SETTING \
    --tasks_json "$TASK_JSON_PATH" 