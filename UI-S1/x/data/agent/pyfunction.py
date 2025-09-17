from x.data.agent.base import STD_THINKING_KEY, BaseFormatAbs
from x.data.text import parse_tags
from x.io.json import read_json
from x.qwen.image import make_qwen_image_item
import copy
import traceback



MOBILE_USE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format

{}

## Action Space

### Action Defination

{}

### Arguments Defination

{}

## Output Format Requirements



## Note


- Planing the task and explain your reasoning step-by-step in `Thought` part.
- Write your action in the `Action` part according to the action space.
- Must output exactly one action per turn.
- Actions must follow Python function call syntax.
- Python function call example format:
  click(coordinate=[123, 456])
  type(text='abc')
- Always use single quotes for string values
""" 

OUTPUT_FORMAT = {
    'only_action': """```
<action> a python function call </action>
```""",
    'thought_action': """```
<thinking> step by step reasoning content in natural language </thinking>
<action> a python function call </action>
```""",
}

import json



class PythonCallFormat(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False):
        super().__init__(space_file, add_thought, force_add_thought)
        
    def build_system_prompt(self, line):
        line_can_thought = self.can_thought(line)
        if line_can_thought:
            _format = 'thought_action'
        else:
            _format = 'only_action'

        return MOBILE_USE.format(OUTPUT_FORMAT[_format], self.space['action_space'], self.space['argument_space'])

    def format_action(self, action_content, image_ele):
        action_content = self._format_action_base(action_content, image_ele)

        action_name = action_content["action"]
        arguments = {key: value for key, value in action_content.items() if key != "action"}
        argument_str = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
        function_call_str = f"{action_name}({argument_str})"
        return function_call_str

    def parse_action(self, action_str, restrict_mode=False):
        action_str = action_str.strip()
        # 找到括号的位置
        start_paren = action_str.find("(")
        end_paren = action_str.rfind(")")

        # 提取动作名称和参数字符串
        action_name = action_str[:start_paren].strip()
        argument_str = action_str[start_paren + 1:end_paren].strip()

        # 使用eval安全地解析参数
        if argument_str:
            try:
                arguments = eval(f"dict({argument_str})")
            except Exception as e:
                raise ValueError(f"解析失败: 检查输入字符串格式。\n[Error ARGS]{argument_str}[Error ARGS]") from e
        else:
            arguments = {}

        # 初始化结果字典
        action_data = {"action": action_name}
        action_data.update(arguments)

        return action_data

    def parse_response(self, model_response, restrict_mode=False):
        result = {}
        result = parse_tags(model_response, ['thinking', 'action'])
        action_content = self.parse_action(result['action'], restrict_mode=True)
        result['action_content'] = action_content
        return result

    def format_response(self, step, image_ele, add_thought=True):
        action_content = step['action_content']
        model_result = {
            'action': f"Action:\n<action>\n{self.format_action(action_content, image_ele)}\n</action>",
            STD_THINKING_KEY: f"Thought:\n<thinking>\n{step[STD_THINKING_KEY]}\n</thinking>" if step.get(STD_THINKING_KEY, None) is not None else None,
            # 'conclusion': f"Conclusion:\n<conclusion>\n{previous_step['conclusion']}\n</conclusion>" if previous_step.get('conclusion', None) is not None else None,
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
            system_prompt = self.build_system_prompt(line)
            # like ui-tars
            messages = [{
                'role': 'system',
                "content": [{"text": system_prompt}]
            }]
            state['_si'] = 0
            state['messages'] = messages
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None
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
        
        if si==0:
            messages[-1]['content'].append({"text": f"User Instruction: {line['goal']}"})
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
            model_response = self.format_response(step, image_ele, add_thought=line_can_thought)
        messages_with_response.append({
            'role': "assistant",
            "content": [{"text": model_response}]
        })
        
        
        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)
        return state

    def to_multiround(self, line):
        line_can_thought = self.can_thought(line)

        system_prompt = self.build_system_prompt(line)

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
                STD_THINKING_KEY: f"<thinking>\n{step[STD_THINKING_KEY]}\n</thinking>" if STD_THINKING_KEY in step else None
            }
            model_response = ""
            if line_can_thought and 'thought' in step: # 这里支持模型在没有thought内容的时候生成thought prompt
                model_response += model_result['thought']+'\n'
                
            model_response += model_result['action']
            messages.append({
                'role': "assistant",
                "content": [{"text": model_response}]
            })
 
        return messages