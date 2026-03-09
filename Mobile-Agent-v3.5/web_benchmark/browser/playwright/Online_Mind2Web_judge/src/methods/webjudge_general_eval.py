from utils import encode_image
from PIL import Image
import re
import asyncio
MAX_IMAGE =50

async def identify_key_points(task, input_image_paths, model):
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    
    prompt = """Task: {task}"""
    text = prompt.format(task=task)

    input_images_msg = []

    if input_image_paths != None:
        for input_image_path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(input_image_path))
            input_images_msg.append(
                                        {
                                            'type': 'image_url',
                                            'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                        }
                                    )

    messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]+ input_images_msg,
            }
        ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]

async def judge_image(task, input_image_paths, image_path, key_points, model):
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
### Reasoning**: [Your explanation]  
### Score**: [1-5]"""


    prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
    text = prompt.format(task=task,key_points=key_points)

    input_images_msg = []
    if input_image_paths != None:
        for input_image_path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(input_image_path))
            input_images_msg.append(
                                        {
                                            'type': 'image_url',
                                            'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                        }
                                    )
    messages = [{"role": "system", "content": system_msg}]

    if input_images_msg:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg
        })
    
    jpg_base64_str = encode_image(Image.open(image_path))
    messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                    },
                ]
            }
        )

    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def WebJudge_general_eval(task, input_image_paths, action_thoughts, last_actions, images_path, model, score_threshold):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), it should be considered a failure.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
5: Some tasks require a submission action or a display of results to be considered successful. Repeat actions or actions that do not lead to a visible result should be considered a failure.
6: If the agent loops through a sequence of actions that do not make progress toward the goal (including failing to click "Save" or "Submit," etc.), it should be considered a failure.

Format your response into two lines as shown below:
Thoughts: <your thoughts and reasoning process should base on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""


    key_points = await identify_key_points(task, input_image_paths, model)
    key_points = key_points.replace("\n\n", "\n")

    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    
    tasks = [judge_image(task, input_image_paths, image_path, key_points, model) for image_path in images_path]
    image_responses = await asyncio.gather(*tasks)

    input_images_msg = []
    whole_content_img = []
    whole_thoughts = []
    record = []
    pattern = r"[1-5]"
    for response, image_path in zip(image_responses, images_path):
        try:
            score_text = response.split("### Score")[1]
            thought = response.split("### Reasoning:")[-1].strip().lstrip("\n").split("### Score")[0].replace('\n',' ')
            score = re.findall(pattern, score_text)[0]
            record.append({"Response": response, "Score": int(score)})
        except Exception as e:
            print(f"Error processing response: {e}")
            score = 0
            record.append({"Response": response, "Score": 0})

        if int(score) >= score_threshold:
            jpg_base64_str = encode_image(Image.open(image_path))
            whole_content_img.append(
                {
                    'type': 'image_url',
                    'image_url': {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}
                }
            )
            if thought != "":
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]
    if len(whole_content_img) == 0:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""

    if action_thoughts != None:
        text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}. Reasoning: {action_thought}" for i, (action, action_thought) in enumerate(zip(last_actions,action_thoughts))), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))

    else:
        text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))

    input_images_msg = []
    if input_image_paths is not None:
        for path in input_image_paths:
            input_images_jpg_base64_str = encode_image(Image.open(path))
            input_images_msg.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
            })

    messages = [{"role": "system", "content": system_msg}]

    if input_images_msg:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": text}] + whole_content_img
    })
    
    return messages, text, system_msg, record, key_points