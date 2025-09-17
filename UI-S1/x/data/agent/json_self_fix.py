
import copy
import json

from x.data.agent.base import STD_THINKING_KEY, BaseFormatAbs
from x.data.text import parse_tags
from x.io import read_json
from x.io.json import read_json
from x.qwen.image import make_qwen_image_item

prompt_template = '''End-to-End Model Thought Integration  



## Integration Requirements

* Write the thought process from a global goal, the action history, thought history and screenshot history.

* The reasoning logic must satisfy:  
  - Begin by reviewing the global task objective.  
  - Inherit the context and decisions from historical steps.  
  - Incorporate the manager’s planning logic.  
  - Derive actions that fully align with the operator’s output.  

## Output Format

<think>
[A coherent reasoning process, reflecting task decomposition, environmental observation, and iterative decision-making]  
</think>

## Output Example

<think>
The current task requires checking the order status of DeepSeek. Access to the official website and locating the login entry have been completed. Based on the page loading result, the login form is ready. Authentication information needs to be filled: the username has already been entered as "DeepSeek," and now the password must be entered.  
</think>


## Key Design Notes  

* Explicitly require the global task objective to ensure the end-to-end model always anchors to the core goal.  

* Enforce structured historical records to prevent information loss.  

* Logic consistency mechanism.  

* The thought process should naturally connect historical conclusions with the current manager’s planning.  

* Transform the manager’s planning into autonomous decisions phrased as "According to the requirements, determine..."  

* Translate operator actions into imperative statements phrased as "Execute..."  

* Do not mention any coordinates in `<think> ... </think>`.

## Global Task Objective
{instruction}

- If this isn't the target app for your operation, you can use open operation to navigate to the correct application.

You can use Next Action Hint to guide the think process, but within the think section, you must conceal the fact that hints were received.


Please integration the thought of current manager and operation into <think> ... </think> in English. 
'''

import json


class JsonFormatSF(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, use_step_instruction=False, repeat_query=False,actions_only=None):
        super().__init__(space_file, add_thought, force_add_thought)
        self.use_step_instruction = use_step_instruction
        self.repeat_query = repeat_query
        self.actions_only=actions_only


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
            system_prompt = prompt_template.format(instruction=line['goal'])

            # like ui-tars
            messages = [{
                'role': 'system',
                "content": [{"text": system_prompt}]
            }]
            state['_si'] = 0
            state['messages'] = messages
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None
        step = line['steps'][si]
        messages = state['messages']

        if previous_model_response:
            messages.append({
                'role': "assistant",
                "content": [{"text": previous_model_response}]
            })
        else:
            if si-1>=0:
                previous_step = line['steps'][si-1]
                model_response = self.format_response(previous_step, state['screenshot_ele'], add_thought=True)
                messages.append({
                    'role': "assistant",
                    "content": [{"text": model_response}]
                })
                   
        messages.append({
                'role': "user",
                "content": []
            })
        format_instruct = "Output Format: <think> ... </think>"
        next_action = step['action_content']

        if si==0:
            messages[-1]['content'].append({"text": f"User Instruction: {line['goal']} ## Next Action Hint: {next_action}\n{format_instruct}"})
        else:
            messages[-1]['content'].append({"text": f"## Next Action Hint: {next_action}\n{format_instruct}"})
        try:
            image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
        except:
            print(step['screenshot'])
            raise
        
        messages[-1]['content'].append(image_ele)

        messages_with_response = copy.deepcopy(messages)
        model_response = ''
        if step.get('action_content', None):
            # 兼容fake line
            model_response = self.format_response(step, image_ele, add_thought=True)
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

    