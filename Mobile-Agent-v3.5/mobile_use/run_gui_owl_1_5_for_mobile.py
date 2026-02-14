"""
Usage:
    cd Mobile-Agent-v3.5/mobile_use
    python run_gui_owl_1_5_for_mobile.py \
        --adb_path "Your ADB path" \
        --api_key "Your api key of vllm service" \
        --base_url "Your base url of vllm service" \
        --model "Your model name of vllm service" \
        --instruction "The instruction you want Mobile-Agent-v3.5 to complete" \
        --add_info "Some supplementary knowledge, can also be empty"
"""

import argparse
import json
import os
import shutil
import time

from PIL import Image

from packages import PACKAGES_NAME_DICT, NAME_PACKAGE_DICT
from utils import (
    AdbTools,
    annotate_screenshot,
    build_messages,
    resolve_app_name_via_llm,
    smart_resize,
    GUIOwlWrapper
)



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Mobile-Agent-v3.5")
    parser.add_argument("--adb_path", type=str, required=True,
                        help="Path to the ADB binary.")
    parser.add_argument("--device", type=str, default=None,
                        help="ADB device serial (optional, for multi-device).")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for the VLM service.")
    parser.add_argument("--base_url", type=str, required=True,
                        help="Base URL for the VLM service.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name for the VLM service.")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Task instruction for the agent.")
    parser.add_argument("--add_info", type=str, default="",
                        help="Supplementary knowledge (can be empty).")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum number of interaction steps.")
    parser.add_argument("--app_resolver_api_key", type=str, default=None,
                        help="API key for the app-resolver LLM (defaults to --api_key).")
    parser.add_argument("--app_resolver_base_url", type=str, default=None,
                        help="Base URL for the app-resolver LLM (defaults to --base_url).")
    parser.add_argument("--app_resolver_model", type=str, default="qwen-plus",
                        help="Model name for the app-resolver LLM.")
    return parser.parse_args()


def parse_action(output_text):
    """
    Extract the action dict from the model's output text.
    Expects a <tool_call> block containing JSON with nested 'arguments'.
    """
    try:
        tool_call_block = output_text.split("<tool_call>\n")[1]
        json_str = tool_call_block.split("}}\n")[0] + "}}"
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse action from model output: {e}")


def rescale_coordinates(action_parameter, resized_width, resized_height):
    """
    Convert normalized (0-1000) coordinates to actual pixel coordinates
    based on the resized image dimensions.
    """
    for key in ("coordinate", "coordinate1", "coordinate2"):
        if key in action_parameter:
            action_parameter[key][0] = int(
                action_parameter[key][0] / 1000 * resized_width
            )
            action_parameter[key][1] = int(
                action_parameter[key][1] / 1000 * resized_height
            )
    return action_parameter


def handle_open_action(
    action_parameter,
    instruction,
    adb_tools,
    resolver_api_key,
    resolver_base_url,
    resolver_model,
):
    """
    Handle the 'open' action: resolve app name to package and launch it.

    Returns:
        True if the app was successfully opened (or user was prompted),
        False if iteration should continue (e.g., app not found).
    """
    app_name = action_parameter.get("text", "")
    package_candidates = NAME_PACKAGE_DICT.get(app_name, [])
    installed_packages = adb_tools.get_package_name()
    display_name = app_name

    # First attempt: direct lookup
    for pkg in package_candidates:
        if pkg in installed_packages:
            adb_tools.open_app(pkg)
            return True

    # Second attempt: resolve via LLM
    installed_app_names = []
    for pkg in installed_packages:
        if pkg in PACKAGES_NAME_DICT:
            installed_app_names.append(PACKAGES_NAME_DICT[pkg][0])

    resolved_name = resolve_app_name_via_llm(
        instruction,
        ", ".join(installed_app_names),
        api_key=resolver_api_key,
        base_url=resolver_base_url,
        model=resolver_model,
    )

    if resolved_name:
        display_name = resolved_name

    resolved_packages = NAME_PACKAGE_DICT.get(resolved_name, [])
    for pkg in resolved_packages:
        if pkg in installed_packages:
            adb_tools.open_app(pkg)
            return True

    # App not found — ask user to install
    input(f"[ACTION REQUIRED] Please install the app: {display_name}")
    return False


def main():
    args = parse_args()

    # Initialize ADB
    adb_tools = AdbTools(adb_path=args.adb_path, device=args.device)

    # Prepare output directories
    instruction = args.instruction
    if args.add_info:
        instruction = f"{instruction} ({args.add_info})"

    task_dir = instruction.replace(" ", "_")[:80]
    anno_dir = task_dir + "_anno"

    for d in (task_dir, anno_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # App-resolver LLM config (falls back to main config)
    resolver_api_key = args.app_resolver_api_key or args.api_key
    resolver_base_url = args.app_resolver_base_url or args.base_url
    resolver_model = args.app_resolver_model

    history = []

    for step_id in range(args.max_steps):
        print(f"\n{'='*50}")
        print(f"STEP {step_id}")
        print(f"{'='*50}")

        # 1. Capture screenshot
        screenshot_path = os.path.join(task_dir, f"screenshot_{step_id}.png")
        if not adb_tools.get_screenshot(screenshot_path):
            print("[ERROR] Failed to capture screenshot. Retrying...")
            time.sleep(1)
            continue

        # 2. Build messages and call the VLM
        messages = build_messages(
            screenshot_path, instruction, history, args.model
        )

        vllm = GUIOwlWrapper(args.api_key, args.base_url, args.model)
        output_text = vllm.predict_mm(messages)

        print(f"[MODEL OUTPUT]\n{output_text}")

        # 3. Parse the action
        action = parse_action(output_text)
        action_parameter = action["arguments"]

        # 4. Rescale coordinates from 1000x1000 to actual resolution
        img = Image.open(screenshot_path)
        resized_h, resized_w = smart_resize(
            img.height, img.width,
            factor=16,
            min_pixels=3136,
            max_pixels=1003520 * 200,
        )
        action_parameter = rescale_coordinates(action_parameter, resized_w, resized_h)

        # 5. Execute the action
        action_type = action_parameter["action"]

        if action_type == "click":
            adb_tools.click(
                action_parameter["coordinate"][0],
                action_parameter["coordinate"][1],
            )

        elif action_type == "long_press":
            adb_tools.long_press(
                action_parameter["coordinate"][0],
                action_parameter["coordinate"][1],
            )

        elif action_type == "type":
            adb_tools.type(action_parameter["text"])

        elif action_type in ("scroll", "swipe"):
            adb_tools.slide(
                action_parameter["coordinate"][0],
                action_parameter["coordinate"][1],
                action_parameter["coordinate2"][0],
                action_parameter["coordinate2"][1],
            )

        elif action_type == "system_button":
            button = action_parameter["button"]
            if button == "Back":
                adb_tools.back()
            elif button == "Home":
                adb_tools.home()

        elif action_type == "wait":
            wait_time = action_parameter.get("time", 2)
            time.sleep(wait_time)

        elif action_type == "terminate":
            status = action_parameter.get("status", "unknown")
            print(f"[TERMINATED] Status: {status}")
            break

        elif action_type == "open":
            opened = handle_open_action(
                action_parameter,
                instruction,
                adb_tools,
                resolver_api_key,
                resolver_base_url,
                resolver_model,
            )
            if not opened:
                continue

        elif action_type == "answer":
            conclusion = output_text.split("<tool_call>")[0].strip()
            print(f"[ANSWER] {conclusion}")
            print("[TERMINATED] Task completed.")
            break

        elif action_type in ("call_user", "calluser", "interact"):
            user_prompt = action_parameter.get("text", "the required action")
            input(f"[ACTION REQUIRED] Please complete: {user_prompt}")
            print("[INFO] User action completed. Resuming...")

        else:
            print(f"[WARN] Unsupported action type: {action_type}")

        # 6. Record history and annotate screenshot
        history.append({"output": output_text, "image": screenshot_path})
        annotate_screenshot(
            screenshot_path,
            action_parameter,
            os.path.join(anno_dir, f"screenshot_anno_{step_id}.png"),
        )
        time.sleep(2)

    print("\n[DONE] Agent execution finished.")


if __name__ == "__main__":
    main()
