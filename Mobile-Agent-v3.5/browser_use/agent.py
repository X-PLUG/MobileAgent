import copy
import base64
import os
import oss2
import traceback
import time
import io
import re
import uuid
import urllib3
import requests
from PIL import Image
import json
import argparse
import sys
import asyncio


import dashscope
from openai import OpenAI

from browser.playwright.Online_Mind2Web_judge.src.run import parallel_eval
from browser.playwright.clean_html import process_element_tag, SALIENT_ATTRIBUTES
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_FALLBACK


def get_bucket(bucket_name, use_internal=False) -> oss2.Bucket:
    bucket = {}
    bucket_key = bucket_name
    if use_internal:
        bucket_key = bucket_name + "-internal"

    access_key = os.environ.get("OSS_ACCESS_KEY_ID")
    secret_key = os.environ.get("OSS_ACCESS_KEY_SECRET")
    if not all([access_key, secret_key]):
        raise EnvironmentError("OSS credentials missing in environment variables")
    auth = oss2.Auth(access_key, secret_key)

    endpoint = 'oss-cn-zhangjiakou.aliyuncs.com'
    if use_internal:
        endpoint = endpoint.replace(".aliyuncs.com", "-internal.aliyuncs.com")
    bucket[bucket_key] = oss2.Bucket(auth, bucket_name=bucket_name, endpoint=endpoint)
    return bucket[bucket_key]


def get_public_url(oss_path, seconds=3600):
    assert oss_path.startswith("oss://"), f"{oss_path} is not a valid oss path."
    bucket_name = oss_path[len("oss://") :].split("/", 1)[0]
    bucket = get_bucket(bucket_name)

    key_path = oss_path[len("oss://") + len(bucket_name) + 1 :]
    if not bucket.object_exists(key_path):
        raise ValueError(f"{key_path} not exists")
    url = bucket.sign_url("GET", key_path, seconds, slash_safe=True)
    return url


def upload_object(data, oss_path, use_internal=False):
    assert oss_path.startswith("oss://"), f"{oss_path} is not a valid oss path."
    bucket_name = oss_path[len("oss://") :].split("/", 1)[0]
    bucket = get_bucket(bucket_name, use_internal=use_internal)

    key_path = oss_path[len("oss://") + len(bucket_name) + 1 :]
    while True:
        try:
            bucket.put_object(key=key_path, data=data)
            break
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            time.sleep(1)


def get_image(raw_image, tmp_dir="oss://nlp-mobile-agent/temp_images", retry_num=50):
    image = Image.open(raw_image)
    byte_stream = io.BytesIO()
    image.save(byte_stream, format="PNG")
    byte_stream.seek(0)
    for retry in range(retry_num):
        try:
            oss_path = f"{tmp_dir}/{str(uuid.uuid4())}.png"
            upload_object(byte_stream, oss_path)
            url = get_public_url(oss_path)
            return url
        except (urllib3.exceptions.ProtocolError, oss2.exceptions.RequestError) as e:
            print(e, f"Retrying: {retry} / {retry_num}")
            time.sleep(1)
        except Exception:
            traceback.print_exc()
            time.sleep(1)

    raise ValueError("make_tmp_image_url failed")



def _call_api(inference_bot, messages, args):

    kwargs = {
        "extra_body": {
            "repetition_penalty": args.repetition_penalty,
            "top_k": args.top_k,
            "seed": args.seed,
            "presence_penalty": 0.0
        }
    }
    response = inference_bot.chat.completions.create(model=args.model, messages=messages, **kwargs)


    try:
        if response["choices"][0]["message"].get("reasoning_content", None) is not None:
            reasoning_content = response["choices"][0]["message"]["reasoning_content"]
            response = f"<think>{reasoning_content}</think>" + response["choices"][0]["message"]["content"][0]["text"]
        else:
            response = response["choices"][0]["message"]["content"][0]["text"]
    except:
        if getattr(response.choices[0].message, "reasoning_content", None) is not None:
            reasoning_content = response.choices[0].message.reasoning_content
            response = f"<think>{reasoning_content}</think>" + response.choices[0].message.content
        else:
            response = response.choices[0].message.content

    return response





class Agent:
    def __init__(self, web, args):
        self.web = web
        self.args = args
        self.history_messages = []
        self.history_action_info = []

        self.messages = []
        self.pattern = r"Thought:|Action:|Observation:"

        self._init()

    def _init(self):
        api_key = os.environ.get('API_KEY')
        if api_key is None:
            raise RuntimeError("API_KEY is not set")
        self.inference_bot = OpenAI(
            api_key=api_key, base_url=self.args.base_url
        )
        self._call = _call_api

        self.messages = [
            {
                "role": "system",
                "content": [
                    {"text": SYSTEM_PROMPT},
                ],
            }
        ]


    def _get_format_messages(self):
        if self.args.provider == "poc-dashscope":
            return self.messages

        formatted_messages = []

        for msg in copy.deepcopy(self.messages):
            new_msg = {
                "role": msg["role"],
                "content": [],
            }

            content = msg.get("content", [])

            if isinstance(content, str):
                new_msg["content"].append(
                    {
                        "type": "text",
                        "text": content,
                    }
                )
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "type" in part:
                        new_msg["content"].append(part)
                        continue

                    if isinstance(part, dict) and "text" in part and "image" not in part:
                        new_msg["content"].append(
                            {
                                "type": "text",
                                "text": part["text"],
                            }
                        )
                    elif isinstance(part, dict) and "image" in part:
                        new_msg["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part["image"]},
                            }
                        )
                        pass
                    else:
                        new_msg["content"].append(part)
            else:
                new_msg["content"].append(
                    {
                        "type": "text",
                        "text": str(content),
                    }
                )

            formatted_messages.append(new_msg)

        return formatted_messages

    def _process_image(self, image_path):
        if self.args.image_type == "file":
            return "file://" + image_path
        elif self.args.image_type == "base64":
            with open(image_path, "rb") as image_file:
                b64 = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
        elif self.args.image_type == "oss":
            return get_image(image_path)



    def _update_messages_with_env_state(self, state, obs, it):
        img_path = state["img_path"]
        SoM_format_ele_text = state["SoM"]["format_ele_text"]

        user_messages = {"role": "user", "content": []}

        image = self._process_image(img_path)

        pre_action = ""
        for i, action in enumerate(self.history_action_info[:-1]):
            pre_action += f"\nStep{i+1}: {action['info']['action_text']}"

        user_messages = {
            "role": "user",
            "content": [
                {
                    "text": f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {self.args.task}\n\nPrevious actions:{pre_action}"
                },
            ]
        }
        if self.args.init_image_path != "":
            user_messages["content"].extend([
                {
                    "text": "Query relevant images are as follows:"
                },
            ])
            for image_path in self.args.init_image_path:
                user_messages["content"].extend([
                    {
                        "image": self._process_image(image_path),
                    }
                ])

        if it == 1:
            user_messages["content"].extend([
                {
                    "image": image,
                },
                {
                    "text": f"observation:{SoM_format_ele_text}"
                }
            ])
            self.messages.append(user_messages)
        else:
            user_messages["content"].extend([
                {
                    "text": "Current screenshot:"
                },
                {
                    "image": self._process_image(self.history_action_info[-1]["info"]["img_path"])
                }
            ])
            assistant_messages = self.messages[-1]
            user_messages_new = {
                "role": "user",
                "content": [
                    {
                        "image": image
                    },
                    {
                        "text": f"observation:{SoM_format_ele_text}"
                    }
                ]
            }
            self.messages = [
                self.messages[0],
                user_messages,
                assistant_messages,
                user_messages_new
            ]
            

    def _update_messages_with_response(self, response):
        assistant_messages = {
            "role": "assistant",
            "content": [{"text": response}],
        }
        self.messages.append(copy.deepcopy(assistant_messages))

    def _step(self):
        for retry_time in range(50):
            try:
                response = self._call(
                    self.inference_bot,
                    self._get_format_messages(),
                    self.args,
                )
                if response is None:
                    continue
                return response
            except Exception as e:
                self._log_error(f"Model call failed, attempting retry #{retry_time}", e)
                time.sleep(1)

    def _handle_response(self, llm_output: str):
        llm_output = llm_output.split("</think>")[-1]
        try:
            lines = llm_output.strip().splitlines()
        except:
            lines = ""
        
        if not lines:
            return None, "Empty input."
        
        if not lines[0].startswith("Action: "):
            return None, "First line must start with 'Action: '."
        
        action_text = lines[0][len("Action: "):].strip()
        if not action_text:
            return None, "Action description is empty."

        match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', llm_output, re.DOTALL)
        if not match:
            return {"action_text": action_text}, "No valid JSON block found between<tool_call> tags."

        json_str = match.group(1).strip()
        try:
            tool_call = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"action_text": action_text}, f"JSON parsing failed: {str(e)}"

        if not isinstance(tool_call, dict):
            return {"action_text": action_text}, "Tool call JSON must be an object."

        if "name" not in tool_call:
            return {"action_text": action_text}, "Missing 'name' field in tool call."
        if "arguments" not in tool_call:
            return {"action_text": action_text}, "Missing 'arguments' field in tool call."

        result = {
            "action_text": action_text,
            "tool_call": tool_call
        }
        return result, "SUCCESS"


    async def get_reward(self, ignore_step):
        format_rewards = []
        step_rewards = []

        for action_info in self.history_action_info:
            obs = action_info.get("obs", "no obs")
            if obs == "":
                format_rewards.append(0)
            else:
                format_rewards.append(-0.5)

        eval_path = os.path.join(self.args.task_dir, f"{self.args.eval_mode}_{self.args.eval_model}_score_threshold_{self.args.eval_score_threshold}_auto_eval_results.json")
        with open(eval_path, encoding='utf-8') as f:
            eval_result = json.load(f)

        step_rewards_tmp = []
        for judge_res in eval_result["image_judge_record"]:
            if eval_result["predicted_label"] == 1:
                step_rewards_tmp.append(judge_res["Score"] / 5)
            elif eval_result["predicted_label"] == 0:
                step_rewards_tmp.append(-(1 - judge_res["Score"] / 5))
            else:
                step_rewards_tmp.append(0)

        step_rewards = [(0 if i in ignore_step else -1) for i in range(len(format_rewards))]

        filled_idx = 1
        for i in range(len(format_rewards)):
            if step_rewards[i] == -1:
                step_rewards[i] = step_rewards_tmp[filled_idx]
                filled_idx += 1


        final_reward = 1 if (eval_result["predicted_label"] == 1) else 0
        step_rewards[-1] = final_reward

        reward_info = {
            "format_rewards": format_rewards,
            "step_rewards": step_rewards,
            "final_reward": final_reward
        }

        
        with open(
            os.path.join(self.args.task_dir, "reward.json"),
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                reward_info,
                json_file,
                ensure_ascii=False,
                indent=4,
            )
        
        


    async def judge_result(self):
        if not self.history_action_info or self.history_action_info[-1]["action"] != "answer":
            return False

        action_result_for_online_mind2web = {
            "task_id": "000",
            "task": self.args.task,
            "final_result_response": self.history_action_info[-1]["info"]["tool_call"]["arguments"]["text"],
            "action_history": [],
            "thoughts": [],
        }

        ignore_step = []
        for action_info in self.history_action_info:
            if action_info["action"] == "answer":
                ignore_step.append(int(action_info["it"])-1)
                break

            if (action_info["info"]["tool_call"]["arguments"].get("action_html", None) is None) or (action_info["info"].get("action_text", None) is None):
                ignore_step.append(int(action_info["it"])-1)
            else:
                action_result_for_online_mind2web["action_history"].append(
                    action_info["info"]["tool_call"]["arguments"]["action_html"]
                )
                action_result_for_online_mind2web["thoughts"].append(
                    action_info["info"]["action_text"]
                )


        with open(
            os.path.join(self.args.task_dir, "result.json"),
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                action_result_for_online_mind2web,
                json_file,
                ensure_ascii=False,
                indent=4,
            )

        self.args.eval_api_key = os.environ.get("EVAL_API_KEY")
        if self.args.eval_api_key is None:
            return

        parser = argparse.ArgumentParser()
        eval_args = parser.parse_args([])
        eval_args.mode = self.args.eval_mode
        eval_args.model = self.args.eval_model
        eval_args.trajectories_dir = self.args.task_dir
        eval_args.api_key = self.args.eval_api_key
        eval_args.output_path = self.args.task_dir
        eval_args.score_threshold = self.args.eval_score_threshold
        eval_args.num_worker = 1
        eval_args.ignore_step = ignore_step

        await parallel_eval(eval_args, eval_args.num_worker)
        await self.get_reward(ignore_step)

        

    async def eval_only(self):
        self.history_action_info = []
        with open(os.path.join(self.args.task_dir, "action_info.json"), encoding="utf-8") as f:
            self.history_action_info = json.load(f)
        await self.judge_result()


    async def _agent_loop(self):
        it = 1
        obs = ""

        start_total = time.time()
        while it <= self.args.max_iter:
            try:
                t0 = time.time()
                current_state = await self._get_environment_state(it)
                t1 = time.time()
                print(f"[DEBUG] Step {it} - _get_environment_state: {t1 - t0:.2f}s")

                t2 = time.time()
                self._update_messages_with_env_state(current_state, obs, it)
                t3 = time.time()
                print(f"[DEBUG] Step {it} - context build & clip: {t3 - t2:.2f}s")

                t4 = time.time()
                if it == self.args.max_iter:
                    response, action_key, info, obs = await self._handle_final_step(current_state, it)
                else:
                    response, action_key, info, obs = await self._process_model_step(current_state, it)

                self._update_messages_with_response(response)
                t5 = time.time()
                print(f"[DEBUG] Step {it} - model inference: {t5 - t4:.2f}s")

                t6 = time.time()
                if action_key != "answer":
                    if obs == "":
                        obs = await self._execute_action_safely(action_key, info, it)
                t7 = time.time()
                exec_time = t7 - t6
                print(f"[DEBUG] Step {it} - action execution: {exec_time:.2f}s")


                t8 = time.time()
                self._record_step(action_key, info, obs, it, current_state)
                t9 = time.time()
                print(f"[DEBUG] Step {it} - record step: {t9 - t8:.2f}s")

                if action_key == "answer":
                    break

                it += 1

            except Exception as e:
                self._log_error(f"Step {it} failed", e)
                break

        if self.args.eval:
            await self.judge_result()


    async def _get_environment_state(self, step: int) -> dict:
        try:
            return await self.web.current_state(step)
        except Exception as e:
            self._log_error(f"Failed to get state at step {step}", e)
            raise

    async def _process_model_step(self, state: dict, step: int):
        response = self._step()
        action_info, parse_obs = self._handle_response(response)

        if action_info is None:
            action_info = state

        if parse_obs != "SUCCESS":
            action_info.update(state)
            return response, None, action_info, parse_obs

        action_key = action_info["tool_call"]["arguments"]["action"]

        action_info.update(state)
        return response, action_key, action_info, ""


    async def _handle_final_step(self, state: dict, step: int):
        self.messages[0] = {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT_FALLBACK}]
        }

        for _ in range(5):
            response = self._step()
            action_info, parse_obs = self._handle_response(response)
            action_key = action_info["tool_call"]["arguments"]["action"]
            if action_key == "answer":
                break

        if action_key == "answer":
            action_info.update(state)
            return response, action_key, action_info, ""
        else:
            think_content = ""
            if action_info is not None:
                print(action_info)
                think_content = action_info["action_text"]
            response = f"Action: {think_content}"+" However, MAX steps reached, I need to answer.\n<tool_call>\n{\"name\": \"web\", \"arguments\": {\"action\": \"answer\", \"text\": \"Terminated due to step limit. Insufficient information to answer.\", \"conclusion\": \"{think_content} However, MAX steps reached, I need to answer. Terminated due to step limit. Insufficient information to answer.\"}}\n</tool_call>"
            action_info, parse_obs = self._handle_response(response)
            action_key = action_info["tool_call"]["arguments"]["action"]
            action_info.update(state)
            return response, action_key, action_info, ""


    async def exec_action(self, action_key, info, it):
        try:
            label = info["tool_call"]["arguments"].get("label", -1)
            text = info["tool_call"]["arguments"].get("text", "")

            if info["tool_call"]["arguments"].get("label", None) is not None:
                if info["tool_call"]["arguments"]["label"] == "WINDOW":
                    x = self.args.window_width // 2
                    y = self.args.window_height // 2
                else:
                    ele = info["SoM"]["SoM_list"][int(info["tool_call"]["arguments"]["label"])]
                    box = ele["bbox"]
                    x, y, w, h = (
                        box.get("x"),
                        box.get("y"),
                        box.get("width"),
                        box.get("height"),
                    )
                    x = x + w / 2
                    y = y + h / 2

                info["tool_call"]["arguments"]["x"] = x
                info["tool_call"]["arguments"]["y"] = y

            action_element_html = ""
            if info["tool_call"]["arguments"].get("x", None) is not None:
                x = info["tool_call"]["arguments"]["x"]
                y = info["tool_call"]["arguments"]["y"]
                action_element_html = await self.web._page.evaluate(
                    """([x, y]) => {
                        const el = document.elementFromPoint(x, y);
                        return el ? el.outerHTML : null;
                    }""",
                    [x, y],
                )
                if action_element_html is None:
                    action_element_html = ""
                action_element_html = process_element_tag(
                    action_element_html, SALIENT_ATTRIBUTES
                )
                action_element_html += f" -> {action_key}"
                if action_key == "type":
                    action_element_html += f" {text}"
            info["tool_call"]["arguments"]["action_html"] = action_element_html

            if action_key == "wait":
                await asyncio.sleep(5)
            elif action_key == "scroll":
                await self.web.scroll_at(x, y, info["tool_call"]["arguments"]["direction"])
            elif action_key == "select":
                await self.web._select(x, y, info)
            elif action_key == "goback":
                await self.web.go_back()
            elif action_key == "goto":
                await self.web.navigate(text)
            elif action_key == "call":
                await self.web._call(info)
            elif action_key == "click":
                await self.web.click_at(x, y)
            elif action_key == "type":
                await self.web.type_text_at(x, y, text)
            elif action_key == "wikipedia":
                await self.web.navigate("https://www.wikipedia.org/")

        except Exception as e:
            self._log_error(f"Action failed", e)
            return (
                "The action you have chosen cannot be exected. "
                "Please double-check if you have selected the wrong Numerical Label or Action or Action format. "
                "Then provide the revised Thought and Action."
            )

        await asyncio.sleep(0.5)
        return "SUCCESS"

    async def _execute_action_safely(self, action_key: str, info: dict, step: int) -> str:
        try:
            exec_obs = await self.exec_action(action_key, info, step)
            return "" if exec_obs == "SUCCESS" else exec_obs
        except Exception as e:
            self._log_error(f"Action {action_key} at step {step} failed", e)
            return f"Action execution error: {str(e)}"

    def _record_step(self, action_key: str, info: dict, obs: str, step: int, state: dict):
        self.history_messages.append(copy.deepcopy(self.messages))
        history_messages_path = os.path.join(self.args.task_dir, "messages.json")
        self._save_json(self.history_messages, "messages.json")

        info["SoM"] = info["SoM"]["format_ele_text"]
        self.history_action_info.append(
            {
                "action": action_key,
                "info": info,
                "it": step,
                "obs": obs,
                "timestamp": time.time()
            }
        )
        self._save_json(self.history_action_info, "action_info.json")

    def _clean_surrogates(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(obj, list):
            return [self._clean_surrogates(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._clean_surrogates(v) for k, v in obj.items()}
        else:
            return obj

    def _save_json(self, data, filename: str):
        try:
            path = os.path.join(self.args.task_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._clean_surrogates(data), f, ensure_ascii=False, indent=4)
        except Exception as e:
            self._log_error(f"Failed to save {filename}", e)

    def _log_error(self, msg: str, exc: Exception):
        error_detail = f"{msg} | {type(exc).__name__}: {str(exc)}"
        print(f"[ERROR] {error_detail}", file=sys.stderr)
        traceback.print_exc()
