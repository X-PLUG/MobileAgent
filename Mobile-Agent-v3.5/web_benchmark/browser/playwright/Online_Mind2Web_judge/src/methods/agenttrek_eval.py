from utils import encode_image
from PIL import Image

def AgentTrek_eval(task, last_actions, thoughts, images_path):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task goal, the agent's trajectory, your goal is to decide whether the agent's execution is successful or not.

*Evaluation Criteria* 
Whether the agent's trajectory is effective and corresponding to the goal 

*Instructions* 1. Review the agent's actions and reasoning processes step by step. 
2. if the agent is stuck in the very first login stage, which means it fails to log into target website at the beginning, that's a failure. 
3. Determine if the agent has achieved the task goal based on the trajectory. A task can be considered successful if most trajectory is effective. 
4. the agent sometimes can't stop after finishing a task and continue doing repeated actions. these actions may be some failed attempt after a series of correct actions. the task should be regarded as successful if the correct actions are effective and almost reach the goal. 
5. if the agent is stuck in the loop at the early stage of the task, which means they don't even get close to the goal before they get stuck in the loop, that's a failure. for example, the agent begin to get stuck before third step. 
6. when the task is to change the google account password, it can't be regarded as successful when agent finish at trying to click "manage your account". 
7. if there are over 8 correct action in the trajectory, it can be regard as a successful agent. 
8. final saving action is not a must. the task is successful if the agent does most things right and just forget to save the change at last. 
9. if the original task has 2 subtasks, the agent only complete one of them, that's still a success. e.g. the task is to update name and birthday, but agent only update name, that's fine. 
10. if the task is to post a review, the agent can be considered successful when it finish writing the review and reach the step to post it, don't have to click the post button. 
11. Since we don't have a printer, some printing related task can be considered successful if the agent reach the step to click print button. 
12. if the task is finished at the initial state and the agent do nothing because of it, it should also be regarded as successful.

*IMPORTANT* 
1. in the trajectory, an action always follows a corresponding reasoning, which shows the observation and thought of the agent. 
2. your response should be contain: 
Thoughts: <your thoughts and reasoning process> 
Status: "success" or "failure"
"""
    prompt = """The goal of the task: {task}

Trajectory:
{thoughts_and_actions}

The last snapshot of the web page is shown in the image."""
    thoughts_and_actions = ""
    for idx, (thought, action) in enumerate(zip(thoughts, last_actions)):
        thought = thought.replace("\n\n", " ")
        action = action.replace("\n\n", " ")
        thoughts_and_actions += f"Thought {idx+1}: {thought}\nAction {idx+1}: {action}\n\n"
    text = prompt.format(task=task, thoughts_and_actions=thoughts_and_actions.strip("\n\n"))

    jpg_base64_str = encode_image(Image.open(images_path))
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                },
            ],
        }
    ]
    return messages, text, system_msg