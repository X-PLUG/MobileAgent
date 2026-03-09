import argparse
import os
from pathlib import Path
import json
import asyncio
from tqdm import tqdm
import requests

from browser.playwright.browser_playwright import PlaywrightComputer
from agent_v3 import Agent

BASE_URL = 'http://127.0.0.1'
visualwebarena_url_map = {
    "__CLASSIFIEDS__": f"{BASE_URL}:9093",
    "__SHOPPING__": f"{BASE_URL}:9094",
    "__REDDIT__": f"{BASE_URL}:9096",
    "__WIKIPEDIA__": f"{BASE_URL}:9097",
    "__HOMEPAGE__": f"{BASE_URL}:900"
}
webarena_url_map = {
    "__SHOPPING__": f"{BASE_URL}:8094",
    "__SHOPPING_ADMIN__": f"{BASE_URL}:8095/admin",
    "__MAP__": f"{BASE_URL}:443",
    "__WIKIPEDIA__": f"{BASE_URL}:8097",
    "__GITLAB__": f"{BASE_URL}:9001",
    "__REDDIT__": f"{BASE_URL}:8096"
}

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
    

def get_real_env_name_from_task_id(task_id):
    if "training_" in task_id:
        return task_id.split("training_")[-1].split("__")[0]
    elif "validation_" in task_id:
        return task_id.split("validation_")[-1].split("__")[0]
    else:
        raise ValueError("task_id must start with training_ or validation_.")
    
def replace_url(url, task_id, env_name=None):
    if "VisualWebArena" == get_real_env_name_from_task_id(task_id):
    # if "VisualWebArena" == env_name:
        url_map = visualwebarena_url_map
    elif "WebArena" == get_real_env_name_from_task_id(task_id):
    # elif "WebArena" == env_name:
        url_map = webarena_url_map
    else:
        raise NotImplementedError
    for placeholder in url_map.keys():
        if placeholder in url:
            url = url.replace(placeholder, url_map[placeholder])
    return url


SPLIT_LABEL = "__"

def get_real_env_name_from_task_id(task_id: str) -> str:
    if "training_" in task_id:
        return task_id.split("training_")[-1].split(SPLIT_LABEL)[0]
    elif "validation_" in task_id:
        return task_id.split("validation_")[-1].split(SPLIT_LABEL)[0]
    else:
        raise ValueError("task_id must start with training_ or validation_.")

async def safe_goto(page, url: str, retry: int = 5):
    for i in range(retry):
        try:
            await page.goto(url, wait_until="load")
            return
        except Exception:
            await asyncio.sleep(2)

    await page.goto(url, wait_until="commit")

async def login_webarena(page, args, task_dir: str = None):
    web_url = args.web

    img_path_login_1 = os.path.join(
        args.task_dir, f"screenshot_login_1.png"
    )

    img_path_login_2 = os.path.join(
        args.task_dir, f"screenshot_login_2.png"
    )

    # SHOPPING
    if webarena_url_map['__SHOPPING__'] in web_url:
        login_url = f"{webarena_url_map['__SHOPPING__']}/customer/account/login/"
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input[type="email"], input#email, input[name*="email"]', "emma.lopez@gmail.com")
        await page.fill('input[type="password"], input#pass, input[name*="pass"]', "Password.123")
        await page.click('button:has-text("Sign In"), input[type="submit"]')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        return

    # SHOPPING_ADMIN
    if webarena_url_map['__SHOPPING_ADMIN__'] in web_url:
        login_url = webarena_url_map['__SHOPPING_ADMIN__']
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input[type="text"], input#username, input[name*="user"]', "admin")
        await page.fill('input[type="password"], input#login, input[name*="pass"]', "admin1234")
        await page.click('button:has-text("Sign in"), input[type="submit"]')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        return

    # REDDIT
    if webarena_url_map['__REDDIT__'] in web_url:
        login_url = f"{webarena_url_map['__REDDIT__']}/login"
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input[name*="username"], input[type="text"]', "MarvelsGrantMan136")
        await page.fill('input[type="password"], input[name*="password"]', "test1234")
        await page.click('button:has-text("Log in"), input[type="submit"]')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        return

    # GITLAB
    if webarena_url_map['__GITLAB__'] in web_url:
        login_url = webarena_url_map['__GITLAB__']
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input#user_login, input[name="user[login]"], input[type="text"]', "byteblaze")
        await page.fill('input#user_password, input[name="user[password]"], input[type="password"]', "hello1234")
        await page.click('input[name="commit"], button:has-text("Sign in")')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        return

    # 其他情况
    await safe_goto(page, web_url)

async def login_visualwebarena(page, args, task_dir: str = None, do_login_flag: dict = None):
    if do_login_flag is None:
        do_login_flag = {"done": False}
    if do_login_flag["done"]:
        return

    web_url = args.web
    img_path_login_1 = os.path.join(
        args.task_dir, f"screenshot_login_1.png"
    )

    img_path_login_2 = os.path.join(
        args.task_dir, f"screenshot_login_2.png"
    )

    # SHOPPING
    if visualwebarena_url_map["__SHOPPING__"] in web_url:
        login_url = f"{visualwebarena_url_map['__SHOPPING__']}/customer/account/login/"
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input[type="email"], input#email, input[name*="email"]', "emma.lopez@gmail.com")
        await page.fill('input[type="password"], input#pass, input[name*="pass"]', "Password.123")
        await page.click('button:has-text("Sign In"), input[type="submit"]')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        do_login_flag["done"] = True
        return

    # REDDIT
    if visualwebarena_url_map["__REDDIT__"] in web_url:
        login_url = f"{visualwebarena_url_map['__REDDIT__']}/login"
        await safe_goto(page, login_url)

        shot = await page.screenshot()
        with open(img_path_login_1, "wb") as f:
            f.write(shot)

        await page.fill('input[name*="username"], input[type="text"]', "MarvelsGrantMan136")
        await page.fill('input[type="password"], input[name*="password"]', "test1234")
        await page.click('button:has-text("Log in"), input[type="submit"]')
        await page.wait_for_timeout(2000)

        shot = await page.screenshot()
        with open(img_path_login_2, "wb") as f:
            f.write(shot)

        await safe_goto(page, web_url)
        do_login_flag["done"] = True
        return
    


    await safe_goto(page, web_url)

async def login_if_required(page, args):
    task_id = args.task_id
    env_name = get_real_env_name_from_task_id(task_id)

    if env_name == "WebArena":
        await login_webarena(page, args)
    elif env_name == "VisualWebArena":
        do_login_flag = {"done": False}
        await login_visualwebarena(page, args, do_login_flag=do_login_flag)
    else:
        await safe_goto(page, args.web)

async def main(args):
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    if args.task_id == "":
        args.task_id = "WebAgent_task_0"
    task_dir = os.path.join(result_dir, f"{args.task_id}/rollout_{args.rollout_id}")
    os.makedirs(task_dir, exist_ok=True)
    args.task_dir = task_dir

    save_args_to_json(args, task_dir)


    raw_file = json.load(open("data/merged_test_raw.json", "r", encoding='utf-8'))
    raw_info = None
    for e in raw_file:
        if ("validation_"+e["task_id"]) == args.task_id:
            raw_info = e
            break

    
    require_login = False
    if raw_info is not None:
        require_login = raw_info.get("require_login", False)
        if raw_info.get("image", None) is not None:
            if isinstance(raw_info["image"], str):
                args.init_image_path = replace_url(raw_info["image"], args.task_id)
                download_with_progress(args.init_image_path, os.path.join(task_dir, "screenshot_task.png"))
                args.init_image_path = [os.path.join(task_dir, "screenshot_task.png")]
            else:
                args.init_image_path = []
                for id, image_path in enumerate(raw_info["image"]):
                    print(replace_url(image_path, args.task_id))
                    download_with_progress(replace_url(image_path, args.task_id), os.path.join(task_dir, f"screenshot_task{id}.png"))
                    args.init_image_path.append(os.path.join(task_dir, f"screenshot_task{id}.png"))


    if args.eval_only:
        web = None
        web_agent = Agent(web, args)
        await web_agent.eval_only()

    else:
        env = PlaywrightComputer(args, highlight_mouse=args.highlight_mouse)

        async with env as web:
            if require_login:
                await login_if_required(web._page, args)
            web_agent = Agent(web, args)
            await web_agent._agent_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for task
    parser.add_argument('--task_id', type=str, default="")
    parser.add_argument('--task', type=str, default="点击地图")
    parser.add_argument('--web', type=str, default="")
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--rollout_id', type=str, default="0")
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument("--current_time", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='results_log')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_attached_imgs", type=int, default=2)
    parser.add_argument("--text_only", action='store_true')
    parser.add_argument("--image_type", type=str, default="oss") # base64
    parser.add_argument("--init_image_path", type=str, default="")
    parser.add_argument("--download_dir", type=str, default="downloads")

    # for agent
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repetition_penalty", type=int, default=1.05)
    parser.add_argument("--top_k", type=int, default=50)

    # for judge
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='WebJudge_Online_Mind2Web_eval', help='the mode of evaluation')
    parser.add_argument('--eval_model', type=str, default='o4-mini-2025-04-16')
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
