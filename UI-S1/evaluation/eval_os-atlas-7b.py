






import argparse
import copy
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from agentcpm_utils import map_action_space2qwenvl
from os_atlas_utils import (build_history_actions_str, os_atlas_2minicpm,
                            predict)
from PIL import Image
from qwenvl_utils import evaluate_android_control_action
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages

# 全局变量（只读），由主线程初始化

result_lock = Lock()  # 用于安全写文件

def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' in step:
            continue
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line

def process_line(line, args):

    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_id = 0
    task_success = False
    fixed_line = fix_line(line)
    history_list = []
    try:
        while step_id < num_steps:
            step = fixed_line['steps'][step_id]
            image_path = step['screenshot']
            low_instruction = step['step_instruction'] if 'step_instruction' in step else ''
            current_check_pam = step['check_options']
            # image = __resize__(Image.open(image_path))
            history = build_history_actions_str(history_list)
            image = Image.open(image_path)
            width, height = image.size
            # 调用模型预测
            model_response = predict(
                model_name=args.model_name,
                instruction=line['goal'],
                low_instruction='',
                history=history,
                image=image
            )
            
            # print("Model Response:", model_response)
            action_minicpm,action_type = os_atlas_2minicpm(model_response)
            # print("Action Minicpm:", action_minicpm)
            # print("Action Type:", action_type)
            history_list.append(low_instruction)
            pred_action = map_action_space2qwenvl(action_minicpm,[width, height])

            type_match, extract_match = evaluate_android_control_action(
                pred_action,
                current_check_pam,
                width, height,
                width, height,
                ignore_actions = []
            )
            print("Type Match:", type_match)
            print("Extract Match:", extract_match)
            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}': {e}")

        traceback.print_exc()
        task_success = False
        step_id = 0

    # 构造结果
    result = {
        "goal": line['goal'],
        "num_steps": num_steps,
        "task_success": task_success,
        "final_step_id": step_id,
    }

    # 线程安全写入
    with result_lock:
        result_path = os.path.join(args.output_dir, f"OS_Atlas_7b.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def main(args):

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 读取数据
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"Loaded {len(std_data)} tasks. Starting parallel evaluation...")

    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_line = {executor.submit(process_line, line, args): line for line in std_data}
        for future in as_completed(future_to_line):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    # 统计最终结果
    success_count = sum(1 for r in results if r["task_success"])
    success_rate = success_count / len(results) * 100 if results else 0
    avg_progress = sum(r["final_step_id"] / r['num_steps'] for r in results) / len(results) if results else 0.0

    
    print(f"\nEvaluation completed.")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{len(results)})")
    print(f"Average Progress: {avg_progress:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate mobile agent on Android control tasks (parallel).")

    parser.add_argument(
        "--jsonl_file",
        type=str,
        default="/evaluation/dataset/android_control_evaluation_std.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/evaluation/result_ac_mp",
        help="Directory to save evaluation results."
    )


    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use in call_mobile_agent_vllm."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel threads (API calls). Default: 4"
    )

    args = parser.parse_args()
    main(args)
