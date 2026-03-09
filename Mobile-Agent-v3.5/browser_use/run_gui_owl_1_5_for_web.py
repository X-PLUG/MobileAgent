import argparse
import os
from pathlib import Path
import json
import asyncio
from tqdm import tqdm
import requests

from browser.playwright.browser_playwright import PlaywrightComputer
from agent import Agent



def save_args_to_json(args, json_path):
    args_dict = vars(args)
    json_path = Path(json_path) / "meta_info.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)


def download_with_progress(url: str, file_path: str):
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with tqdm(
            total=total_size,
            unit='B',         
            unit_scale=True,  
            desc=os.path.basename(file_path)
        ) as progress_bar:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error: Download failed - {e}")
        return False

async def main(args):
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    if args.task_id == "":
        args.task_id = "WebAgent_task_0"
    task_dir = os.path.join(result_dir, f"{args.task_id}/rollout_{args.rollout_id}")
    os.makedirs(task_dir, exist_ok=True)
    args.task_dir = task_dir

    save_args_to_json(args, task_dir)


    if args.init_image_path != "":
        download_with_progress(args.init_image_path, os.path.join(task_dir, "screenshot_task.png"))
        args.init_image_path = os.path.join(task_dir, "screenshot_task.png")
        

    if args.eval_only:
        web = None
        web_agent = Agent(web, args)
        await web_agent.eval_only()

    else:
        env = PlaywrightComputer(args, highlight_mouse=args.highlight_mouse)

        async with env as web:
            web_agent = Agent(web, args)
            await web_agent._agent_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for task
    parser.add_argument('--task_id', type=str, default="")
    parser.add_argument('--task', type=str, default="")
    parser.add_argument('--web', type=str, default="")
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--rollout_id', type=str, default="0")
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument("--current_time", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='results_log')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_attached_imgs", type=int, default=2)
    parser.add_argument("--text_only", action='store_true')
    parser.add_argument("--image_type", type=str, default="base64") # file
    parser.add_argument("--init_image_path", type=str, default="")
    parser.add_argument("--download_dir", type=str, default="downloads")

    # for agent
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repetition_penalty", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=20)

    # for judge
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='', help='the mode of evaluation')
    parser.add_argument('--eval_model', type=str, default='')
    parser.add_argument('--eval_score_threshold', type=int, default=3)
    parser.add_argument("--eval_only", action='store_true')

    # for web browser
    parser.add_argument("--use_css_som", action='store_true')
    parser.add_argument("--use_omni_som", action='store_true')
    parser.add_argument("--omni_url", type=str, default='')
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1080)
    parser.add_argument("--window_height", type=int, default=1440)
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--keep_user_info", action='store_true')
    parser.add_argument("--highlight_mouse", action='store_true')

    args = parser.parse_args()
    asyncio.run(main(args))
