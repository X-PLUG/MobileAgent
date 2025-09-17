
import copy
import json

from x.data.agent.base import STD_THINKING_KEY, BaseFormatAbs
from x.data.text import parse_tags
from x.io import read_json
from x.io.json import read_json
from x.qwen.image import make_qwen_image_item

MOBILE_USE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format

{}

## Action Space

{}

## Note

- Planing the task and explain your reasoning step-by-step in `think` part.
- Write your action in the `action` part according to the action space.
""" 

OUTPUT_FORMAT = {
    'only_action': """```
<action> ... </action>
```""",
    'thought_action': """```
<think> ... </think>

<action> ... </action>
```""",
}

STEP_PREDICTION = """
Please act as a GUI operation prediction model. You will be given a pre-operation image and a post-operation image, along with the possible action space. Please infer what operation occurred.

## Action Space

{}


## Output Format
```
<action> ... </action>
```"""


import json


def generate_prompt(action_space, add_thought=True, actions_only=None):
    prompt_parts = []
    # Describe available actions
    prompt_parts.append("You can perform the following actions:")
    action_space_new = {}
    arguments_only = []
    if actions_only != None:
        for action in action_space['action_space']:
            if action["action"] in actions_only:
                action_space_new["action_space"].appned(action)
                for argument in action["arguments"]:
                    arguments_only.append(argument)
    else:
        action_space_new = action_space
    for action in action_space_new['action_space']:
        if actions_only == None or action['action'] in actions_only:
            prompt_parts.append(f"- {action['action']}: {action['action_desc']}")

    # Describe available arguments
    prompt_parts.append("\nThe arguments you can use are:")
    for argument in action_space_new['argument_space']:
        if actions_only == None or argument['argument'] in arguments_only:
            enum_part = f" Possible values: {', '.join(argument['enum'])}." if argument['enum'] else ""
            prompt_parts.append(f"- {argument['argument']}: {argument['argument_desc']}{enum_part}")

    # Provide instructions for formatting output
    prompt_parts.append("\nFormat your output as a JSON object with the selected action and its arguments at the same level.")
    examples = []
    for action in action_space_new['action_space'][:2]:  # Using top two actions as examples
        example = {"action": action['action']}
        for argument in action['arguments']:
            example[argument] = "<value>"
        if add_thought:
            examples.append(f"<think>\n...\n</think>\n<action>\n{json.dumps(example,ensure_ascii=False)}\n</action>")
        else:
            examples.append(f"<action>\n{json.dumps(example, ensure_ascii=False)}\n</action>")
    
    prompt_parts.append("\nExample outputs:")
    prompt_parts.extend(examples)
    
    return "\n".join(prompt_parts)

class JsonFormat(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, use_step_instruction=False, repeat_query=False,actions_only=None,hint=False):
        super().__init__(space_file, add_thought, force_add_thought)
        self.use_step_instruction = use_step_instruction
        self.repeat_query = repeat_query
        self.actions_only=actions_only
        self.hint = hint


    def format_action(self, action_content, image_ele):
        action_content = self._format_action_base(action_content, image_ele)
        return json.dumps(action_content, ensure_ascii=False)

    def parse_action(self, action_str, restrict_mode=False):
        return json.loads(action_str)

    def parse_response(self, model_response, restrict_mode=False):
        result = {}
        result = parse_tags(model_response, ['think', 'action'])
        action_content = self.parse_action(result['action'], restrict_mode=True)
        result['action_content'] = action_content
        return result

    def format_response(self, step, image_ele, add_thought=True):
        action_content = step['action_content']
        model_result = {
            'action': f"<action>\n{self.format_action(action_content, image_ele)}\n</action>",
            STD_THINKING_KEY: f"<think>\n{step[STD_THINKING_KEY]}\n</think>" if step.get(STD_THINKING_KEY, None) is not None else None,
           
        }
        model_response = ""
        if add_thought and STD_THINKING_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_THINKING_KEY]+'\n'
            
        model_response += model_result['action']
        return model_response
    
    def gen_next_round(self, line, state, previous_model_response=None):
        '''
        用于构建静态评测
        '''
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            if line_can_thought:
                _format = 'thought_action'
            else:
                _format = 'only_action'
            system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(self.space,self.actions_only))
            # like ui-tars
            messages = [{
                'role': 'system',
                "content": [{"text": system_prompt}]
            }]
            state['_si'] = 0
            state['messages'] = messages
            state['line_can_thought'] = line_can_thought
            state['_format'] = _format
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None
        _format = state['_format']
        step = line['steps'][si]
        messages = state['messages']
        line_can_thought = state['line_can_thought']

        if previous_model_response:
            messages.append({
                'role': "assistant",
                "content": [{"text": previous_model_response}]
            })
        else:
            if si-1>=0:
                previous_step = line['steps'][si-1]
                model_response = self.format_response(previous_step, state['screenshot_ele'], add_thought=line_can_thought)
                messages.append({
                    'role': "assistant",
                    "content": [{"text": model_response}]
                })
                   
        messages.append({
                'role': "user",
                "content": []
            })
        format_instruct = "Output Format: {}".format(OUTPUT_FORMAT[_format])
        if si==0:
            messages[-1]['content'].append({"text": f"User Instruction: {line['goal']}\n{format_instruct}"})
            messages[-1]['content'].append({"text": "If the query asks a question, please answer the question through the answer action before terminating the process.\n"})
        elif self.repeat_query:
            messages[-1]['content'].append({"text": f"User Instruction: {line['goal']}\n{format_instruct}"})
        else:
            messages[-1]['content'].append({"text": f"{format_instruct}"})
        
        try:
            image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
        except:
            print(step['screenshot'])
            raise
        
        messages[-1]['content'].append(image_ele)

        messages_with_response = copy.deepcopy(messages)
        if self.hint:
            messages[-1]['content'].append({"text": f"Action Hint: {self.format_response(step, image_ele, add_thought=False)}\n"})
        model_response = ''
        if step.get('action_content', None):
            # 兼容fake line
            model_response = self.format_response(step, image_ele, add_thought=line_can_thought)
        messages_with_response.append({
            'role': "assistant",
            "content": [{"text": model_response}]
        })
        if 'thought' in step and step['thought'] != "":
            state['thought'] = step['thought']
        elif 'motivation' in step and step['motivation'] != "":
            state['thought'] = step['motivation']
        else:
            state['thought'] = ""
        # state['thought'] = step['thought']
        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)
        return state

    

    def to_multiround(self, line):
        '''
        用于构建多步训练数据
        '''
        line_can_thought = self.can_thought(line)

        if line_can_thought:
            _format = 'thought_action'
        else:
            _format = 'only_action'
        system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(self.space,self.actions_only))

        # like ui-tars
        messages = [{
            'role': 'system',
            "content": [{"text": system_prompt}]
        }]
        
        for si, step in enumerate(line['steps']):
            messages.append({
                'role': "user",
                "content": []
            })
            if si==0:
                messages[-1]['content'].append({"text": f"User Instruction: {line['goal']}"})
            try:
                image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
            except:
                print(step['screenshot'])
                raise
            messages[-1]['content'].append(image_ele)
                    

            model_result = {
                'action': f"<action>\n{self.format_action(step['action_content'], image_ele)}\n</action>",
                STD_THINKING_KEY: f"<think>\n{step[STD_THINKING_KEY]}\n</think>" if STD_THINKING_KEY in step else None
            }
            model_response = ""
            if line_can_thought and STD_THINKING_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
                model_response += model_result[STD_THINKING_KEY]+'\n'
                
            model_response += model_result['action']
            messages.append({
                'role': "assistant",
                "content": [{"text": model_response}]
            })
 
        return messages

    def to_step_prediction_message(self, line):
        '''
        用于动态环境交互
        '''
        for si, step in enumerate(line['steps']):
            if si>=len(line['steps'])-1:
                break
            next_step = line['steps'][si+1]
            messages = [{
                'role': 'system',
                "content": [{"text": STEP_PREDICTION.format(generate_prompt(self.space, add_thought=self.add_thought,actions_only=self.actions_only))}]
            }]
            image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
            messages.append({
                'role': "user",
                "content": [
                    {"text": f"Screenshot before action:\n"},
                    image_ele,
                    {"text": f"Screenshot after action:\n"},
                    make_qwen_image_item(next_step['screenshot'], image=next_step.get('screenshot_pil', None)),]
            })
            messages.append({
                'role': "assistant",
                "content": [{"text": f"<action>\n{self.format_action(step['action_content'], image_ele)}\n</action>"}]
            })
            yield messages