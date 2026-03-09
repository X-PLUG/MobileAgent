import argparse
import os
from .methods.agenttrek_eval import *
from .methods.automomous_eval import *
from .methods.webjudge_general_eval import *
from .methods.webjudge_online_mind2web import *
from .methods.webvoyager_eval import *
from .utils import OpenaiEngine, extract_predication
import json
import copy
import asyncio
import multiprocessing

async def auto_eval(args, task_subset, final_predicted_labels, model):

    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json")
    already_ids = []
    # if os.path.exists(output_json_path):
    #     with open(output_json_path,"r") as f:
    #         already_data = f.read()
    #     already_tasks = already_data.splitlines()
    #     for item in already_tasks:
    #         item = json.loads(item)
    #         already_ids.append(item["task_id"])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:
        #Skip already done task
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = None
        task_description = None
        # Load results
        with open(os.path.join(args.trajectories_dir, task_id, "result.json")) as f:
            result = json.load(f)
            output_results = copy.deepcopy(result)
            task_description = result["task"]
            if "action_history" in result:
                action_history = result["action_history"]
            if "thoughts" in result:
                thoughts = result["thoughts"]
            if "final_result_response" in result:
                final_result_response = result["final_result_response"]
            if "input_image_paths" in result:
                input_image_paths = result["input_image_paths"]

        print(f"Start evaluation for {task_description}")
        # Do the auto-eval
        if args.mode == "Autonomous_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])
        
        elif args.mode == "AgentTrek_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                    screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])
        
        elif args.mode == "WebVoyager_eval":
            for id, image in enumerate(sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0]))):
                if id in args.ignore_step:
                    continue
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg = await WebVoyager_eval(task_description, screenshot_paths, final_result_response)
        
        elif args.mode == "WebJudge_Online_Mind2Web_eval":
            for id, image in enumerate(sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0]))):
                if id in args.ignore_step:
                    continue
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = await WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model, args.score_threshold)
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        elif args.mode == "WebJudge_general_eval":
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        response = await model.generate(messages)
        print(response)
        response = response[0]
        predicted_label = extract_predication(response, args.mode)
        
        #Store evaluation details
        evaluation_results = {"response": response, "predicted_label": predicted_label}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label

        final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("="*20)
        os.makedirs(args.output_path, exist_ok=True)

        with open(os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json"), "w") as f_out:
            f_out.write(json.dumps(output_results) + "\n")


async def process_subset(task_subset, args, final_predicted_labels, model):

    await auto_eval(args, task_subset, final_predicted_labels, model)

async def parallel_eval(args, num_workers=60):

    #Evaluate in parallel based on num of works
    task_dirs = [
        ""
    ]
    print(f"Evaluating {len(task_dirs)} tasks in total.")
    chunk_size = len(task_dirs) // num_workers
    task_subsets = [task_dirs[i:i + chunk_size] for i in range(0, len(task_dirs), chunk_size)]

    #Load model
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key
    )

    final_predicted_labels = []
    for subset in task_subsets:
        await process_subset(subset, args, final_predicted_labels, model) 


    # lock = multiprocessing.Lock()
    # with multiprocessing.Manager() as manager:
    #     final_predicted_labels = manager.list()
    #     processes = []
    #     for subset in task_subsets:
    #         p = multiprocessing.Process(target=process_subset, args=(subset, args, final_predicted_labels, lock, model))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     success_num = sum(final_predicted_labels) 

    # print("Evaluation complete.")
    # print(f"The success rate is {(success_num / len(task_dirs)) * 100}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument('--mode', type=str, default='Online_Mind2Web_eval', help='the mode of evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--trajectories_dir", type=str, required=True, help="Path to trajectories directory")
    parser.add_argument("--api_key", type=str, required=True, help="The api key")
    parser.add_argument("--output_path", type=str, required=True, help="The output path")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=60)
    args = parser.parse_args()

    parallel_eval(args, args.num_worker)

