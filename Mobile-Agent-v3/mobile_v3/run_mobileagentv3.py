import os
import uuid
import json
import time
import argparse
from PIL import Image
from datetime import datetime

from utils.mobile_agent_e import (
    InfoPool, 
    Manager, 
    Executor, 
    Notetaker, 
    ActionReflector,
    INPUT_KNOW
)
import utils.controller as controller
from utils.call_mobile_agent_e import GUIOwlWrapper

def run_instruction(adb_path, api_key, base_url, model, instruction, add_info, coor_type, if_notetaker, max_step=25, log_path="./logs"):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_path}/{time_str}_{instruction[:10]}"
    os.mkdir(save_path)
    image_save_path = os.path.join(save_path, "images")
    os.mkdir(image_save_path)

    info_pool = InfoPool(
        additional_knowledge_manager=add_info,
        additional_knowledge_executor=INPUT_KNOW,
        err_to_manager_thresh=2
    )
    
    vllm = GUIOwlWrapper(api_key, base_url, model)
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    message_manager, message_operator, message_reflector, message_notekeeper = None, None, None, None
    info_pool.instruction = instruction

    for step in range(max_step):
        if step == max_step:
            task_result_path = os.path.join(save_path, "task_result.json")
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 1.0}
            with open(task_result_path, 'w', encoding='utf-8') as json_file:
                json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
            break
        
        if step == 0:
            current_time = datetime.now()
            formatted_time = current_time.strftime(f'%Y-%m-%d-{current_time.hour * 3600 + current_time.minute * 60 + current_time.second}-{str(uuid.uuid4().hex[:8])}')
            local_image_dir = os.path.join(image_save_path, f"screenshot_{formatted_time}.png")
        else:
            local_image_dir = local_image_dir2
        
        # get the screenshot
        for _ in range(5):
            if not controller.get_screenshot(adb_path, local_image_dir):
                print("Get screenshot failed, retry.")
                time.sleep(5)
            else:
                break
        
        width, height = Image.open(local_image_dir).size
        
        info_pool.error_flag_plan = False
        err_to_manager_thresh = info_pool.err_to_manager_thresh
        if len(info_pool.action_outcomes) >= err_to_manager_thresh:
            # check if the last err_to_manager_thresh actions are all errors
            latest_outcomes = info_pool.action_outcomes[-err_to_manager_thresh:]
            count = 0
            for outcome in latest_outcomes:
                if outcome in ["B", "C"]:
                    count += 1
            if count == err_to_manager_thresh:
                info_pool.error_flag_plan = True

        skip_manager = False
        ## if previous action is invalid, skip the manager and try again first ##
        if not info_pool.error_flag_plan and len(info_pool.action_history) > 0:
            if info_pool.action_history[-1]['action'] == 'invalid':
                skip_manager = True
        
        if not skip_manager:
            print("\n### Manager ... ###\n")
            prompt_planning = manager.get_prompt(info_pool)
            output_planning, message_manager, raw_response = vllm.predict_mm(
                prompt_planning,
                [local_image_dir]
            )
        
        message_save_path = os.path.join(save_path, f"step_{step+1}")
        os.mkdir(message_save_path)
        message_file = os.path.join(message_save_path, "manager.json")
        message_data = {"name": "manager", "messages": message_manager, "response": output_planning, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)

        parsed_result_planning = manager.parse_response(output_planning)
        info_pool.completed_plan = parsed_result_planning['completed_subgoal']
        info_pool.plan = parsed_result_planning['plan']
        if not raw_response:
            raise RuntimeError('Error calling vLLM in planning phase.')
        
        print('Completed subgoal: ' + info_pool.completed_plan)
        print('Planning thought: ' + parsed_result_planning['thought'])
        print('Plan: ' + info_pool.plan, "\n")
        
        if "Finished" in info_pool.plan.strip() and len(info_pool.plan.strip()) < 15:
            print("Instruction finished, stop the process.")
            task_result_path = os.path.join(save_path, "task_result.json")
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 0.0}
            with open(task_result_path, 'w', encoding='utf-8') as json_file:
                json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
            break
        else:
            print("\n### Operator ... ###\n")

            prompt_action = executor.get_prompt(info_pool)
            output_action, message_operator, raw_response = vllm.predict_mm(
                prompt_action,
                [local_image_dir],
            )
            
            if not raw_response:
                raise RuntimeError('Error calling LLM in operator phase.')
            parsed_result_action = executor.parse_response(output_action)
            action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action['action'], parsed_result_action['description']
            
            info_pool.last_action_thought = action_thought
            info_pool.last_summary = action_description
            
            if (not action_thought) or (not action_object_str):
                print('Action prompt output is not in the correct format.')
                info_pool.last_action = {"action": "invalid"}
                info_pool.action_history.append({"action": "invalid"})
                info_pool.summary_history.append(action_description)
                info_pool.action_outcomes.append("C")
                info_pool.error_descriptions.append("invalid action format, do nothing.")
                continue
        
        action_object_str = action_object_str.replace("```", "").replace("json", "").strip()
        print('Thought: ' + action_thought)
        print('Action: ' + action_object_str)
        print('Action description: ' + action_description)

        try:
            action_object = json.loads(action_object_str)
            operator_response = f'''### Thought ###
{action_thought}

### Action ###
{action_object}

### Description ###
{action_description}'''
            
            if action_object['action'] == "answer":
                message_file = os.path.join(message_save_path, "operator.json")
                message_data = {"name": "operator", "messages": message_operator, "response": operator_response, "step_id": step+1}
                with open(message_file, 'w', encoding='utf-8') as json_file:
                    json.dump(message_data, json_file, ensure_ascii=False, indent=4)

                answer_content = action_object['text']
                print(f"Instruction finished, answer: {answer_content}, stop the process.")
                task_result_path = os.path.join(save_path, "task_result.json")
                current_time = datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                task_result_data = {"goal": instruction, "finish_dtime": formatted_time, "hit_step_limit": 0.0}
                with open(task_result_path, 'w', encoding='utf-8') as json_file:
                    json.dump(task_result_data, json_file, ensure_ascii=False, indent=4)
                break
            
            if coor_type != "abs":
                if "coordinate" in action_object:
                    action_object['coordinate'] = [int(action_object['coordinate'][0] / 1000 * width), int(action_object['coordinate'][1] / 1000 * height)]
                if "coordinate2" in action_object:
                    action_object['coordinate2'] = [int(action_object['coordinate2'][0] / 1000 * width), int(action_object['coordinate2'][1] / 1000 * height)]
            
            if action_object['action'] == "click":
                controller.tap(adb_path, action_object['coordinate'][0], action_object['coordinate'][1])
            elif action_object['action'] == "swipe":
                controller.slide(adb_path, action_object['coordinate'][0], action_object['coordinate'][1], action_object['coordinate2'][0], action_object['coordinate2'][1])
            elif action_object['action'] == "type":
                controller.type(adb_path, action_object['text'])
            elif action_object['action'] == "system_button":
                if action_object['button'] == "Back":
                    controller.back(adb_path)
                elif action_object['button'] == "Home":
                    controller.home(adb_path)
            
        except:
            info_pool.last_action = {"action": "invalid"}
            info_pool.action_history.append({"action": "invalid"})
            info_pool.summary_history.append(action_description)
            info_pool.action_outcomes.append("C")
            info_pool.error_descriptions.append("invalid action format, do nothing.")
            local_image_dir2 = local_image_dir
            continue
        
        message_file = os.path.join(message_save_path, "operator.json")
        message_data = {"name": "operator", "messages": message_operator, "response": operator_response, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)

        info_pool.last_action = json.loads(action_object_str)
        
        if step == 0:
            time.sleep(8) # maybe a pop-up when first open an app
        time.sleep(2)
        
        current_time = datetime.now()
        formatted_time = current_time.strftime(f'%Y-%m-%d-{current_time.hour * 3600 + current_time.minute * 60 + current_time.second}-{str(uuid.uuid4().hex[:8])}')
        local_image_dir2 = os.path.join(image_save_path, f"screenshot_{formatted_time}.png")
        
        # get the screenshot
        for _ in range(5):
            if not controller.get_screenshot(adb_path, local_image_dir2):
                print("Get screenshot failed, retry.")
                time.sleep(5)
            else:
                break
        
        print("\n### Action Reflector ... ###\n")
        prompt_action_reflect = action_reflector.get_prompt(info_pool)
        output_action_reflect, message_reflector, raw_response = vllm.predict_mm(
            prompt_action_reflect,
            [
                local_image_dir,
                local_image_dir2,
            ],
        )
        
        message_file = os.path.join(message_save_path, "reflector.json")
        message_data = {"name": "reflector", "messages": message_reflector, "response": output_action_reflect, "step_id": step+1}
        with open(message_file, 'w', encoding='utf-8') as json_file:
            json.dump(message_data, json_file, ensure_ascii=False, indent=4)
        
        parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
        outcome, error_description = (
            parsed_result_action_reflect['outcome'], 
            parsed_result_action_reflect['error_description']
        )
        progress_status = info_pool.completed_plan
        
        if "A" in outcome: # Successful. The result of the last action meets the expectation.
          action_outcome = "A"
        elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
            action_outcome = "B"
        elif "C" in outcome: # Failed. The last action produces no changes.
            action_outcome = "C"
        else:
            raise ValueError("Invalid outcome:", outcome)
        
        print('Action reflection outcome: ' + action_outcome)
        print('Action reflection error description: ' + error_description)
        print('Action reflection progress status: ' + progress_status, "\n")
        
        info_pool.action_history.append(json.loads(action_object_str))
        info_pool.summary_history.append(action_description)
        info_pool.action_outcomes.append(action_outcome)
        info_pool.error_descriptions.append(error_description)
        info_pool.progress_status = progress_status
        
        if action_outcome == "A" and if_notetaker:
            print("\n### NoteKeeper ... ###\n")
            prompt_note = notetaker.get_prompt(info_pool)
            output_note, message_notekeeper, raw_response = vllm.predict_mm(
                prompt_note,
                [local_image_dir2],
            )
            
            message_file = os.path.join(message_save_path, "notekeeper.json")
            message_data = {"name": "notekeeper", "messages": message_notekeeper, "response": output_note, "step_id": step+1}
            with open(message_file, 'w', encoding='utf-8') as json_file:
                json.dump(message_data, json_file, ensure_ascii=False, indent=4)
            
            parsed_result_note = notetaker.parse_response(output_note)
            important_notes = parsed_result_note['important_notes']
            info_pool.important_notes = important_notes

            print('Important notes: ' + important_notes, "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Mobile-Agent-v3 with a given model and instruction"
    )
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--add_info", type=str, default="")
    parser.add_argument("--coor_type", type=str, default="abs")
    parser.add_argument("--notetaker", type=bool, default=False)
    args = parser.parse_args()
    
    run_instruction(args.adb_path, args.api_key, args.base_url, args.model, args.instruction, args.add_info, args.coor_type, args.notetaker)
    