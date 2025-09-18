import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image
from qwenvl_utils import (call_mobile_agent_vllm,
                          evaluate_android_control_action, find_last_image_ele)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages

# 全局变量（只读），由主线程初始化
RAW_SPACE = None
fm = None
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
    global fm

    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_id = 0
    task_success = False
    fixed_line = fix_line(line)
    try:
        while step_id < num_steps:
            
            current_check_pam = fixed_line['steps'][step_id]['check_options']
            state = fm.gen_next_round(fixed_line, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']
            messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)

            current_image_ele, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(

                messages=messages,
                model_name=args.model_name
            )

            pred_action = fm.parse_response(model_response)
            type_match, extract_match = evaluate_android_control_action(
                pred_action['action_content'],
                current_check_pam,
                width, height,
                resized_width, resized_height
            )

            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}': {e}")
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
        result_path = os.path.join(args.output_dir, f"{args.model_name}.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def main(args):
    global RAW_SPACE, fm

    # 初始化全局只读组件（在主线程）
    from x.data.agent.space.std_space import RAW_SPACE as _RAW_SPACE
    RAW_SPACE = _RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

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
        "--n_history_image_limit",
        type=int,
        default=2,
        help="Maximum number of historical images to keep. Default: 2"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel threads (API calls). Default: 4"
    )

    args = parser.parse_args()
    main(args)
