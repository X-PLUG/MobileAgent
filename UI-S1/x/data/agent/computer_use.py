
from x.data.agent.base import STD_CONCLUSION_KEY, STD_THINKING_KEY, BaseFormatAbs

from x.data.text import parse_tags
import json

from x.qwen.image import make_qwen_image_item
import copy
import traceback
from x.qwen.agent import get_computer_system_prompt
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
       
class ComputerUseSingleTurnFormat(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, custom_use=None):
        super().__init__(space_file, add_thought, force_add_thought)
        self.suffix = ''
        self.custom_use = custom_use
        if self.add_thought:
            self.suffix += '_add_thought'
      
    def format_action(self, action_content, image_ele):
        action_content = self._format_action_base(action_content, image_ele)
        return json.dumps({"name": "mobile_use", "arguments": action_content}, ensure_ascii=False)

    def parse_action(self, action_str, restrict_mode=False):
        d = json.loads(action_str)
        return d['arguments']

    def parse_response(self, model_response, restrict_mode=False):
        result = {}
        result = parse_tags(model_response, ['thinking', 'tool_call', 'conclusion'])
        action_content = self.parse_action(result['tool_call'], restrict_mode=True)
        result['action_content'] = action_content
        return result

    def format_response(self, step, image_ele, add_thought=True):
        action_content = step['action_content']
        model_result = {
            'action': f"<tool_call>\n{self.format_action(action_content, image_ele)}\n</tool_call>",
            STD_THINKING_KEY: f"<thinking>\n{step[STD_THINKING_KEY]}\n</thinking>" if step.get(STD_THINKING_KEY, None) is not None else None,
            STD_CONCLUSION_KEY: f"<conclusion>\n{step[STD_CONCLUSION_KEY]}\n</conclusion>" if step.get(STD_CONCLUSION_KEY, None) is not None else None,
        }
        model_response = ""
        if add_thought and STD_THINKING_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_THINKING_KEY]+'\n'
            
        model_response += model_result['action'] + '\n'
        if add_thought and STD_CONCLUSION_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_CONCLUSION_KEY]+'\n'
                    
        return model_response
    
    def gen_next_round(self, line, state, previous_model_response=None):
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            image_ele_0 = make_qwen_image_item(line['steps'][0]['screenshot'], image=line['steps'][0].get('screenshot_pil', None))

            # TODO 使用动态动作空间

            state['_si'] = 0
            state['system'] = [get_computer_system_prompt(height=image_ele_0['resized_height'], width=image_ele_0['resized_width'], custom_use=self.custom_use)]
            state['step_history'] = []
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        history = state['step_history']
        if si >= len(line['steps']):
            return None

        line_can_thought = state['line_can_thought']

        if previous_model_response:
            history.append(self.parse_response(previous_model_response)['conclusion'])
        else:
            if si-1>=0:
                previous_step = line['steps'][si-1]
                history.append(previous_step['conclusion'])
              

        messages = []
        messages.extend(state['system'])
        history = '\n\n'.join([f"Step {hsi+1}: {_}" for hsi, _ in enumerate(history)])
        query = line['goal']
        user_prompt = '''Please generate the next move according to the UI screenshot, instruction and previous actions.
Instruction: {instruction}
Previous actions:
{history}
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'''.format(instruction=query, history=history)

        step = line['steps'][si]

        image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
        messages.append({
            "role": "user", 
            "content": [
                {"text": user_prompt},
                image_ele
                ]
        })

        messages_with_response = copy.deepcopy(messages)
        model_response = ''
        if 'action_content' in step:
            # 兼容fake line
            model_response = self.format_response(step, image_ele, add_thought=line_can_thought)
        messages_with_response.append({
            'role': "assistant",
            "content": [{"text": model_response}]
        })
        
        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)
        return state

class ComputerUseMultiTurnFormat(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, use_step_instruction=False, custom_use=None):
        super().__init__(space_file, add_thought, force_add_thought)
        self.suffix = ''
        self.custom_use = custom_use
        self.use_step_instruction = use_step_instruction
        if self.add_thought:
            self.suffix += '_add_thought'
        if self.use_step_instruction:
            self.suffix += '_step_instruction'
    
    def format_action(self, action_content, image_ele):
        action_content = self._format_action_base(action_content, image_ele)
        return json.dumps({"name": "mobile_use", "arguments": action_content}, ensure_ascii=False)

    def parse_action(self, action_str, restrict_mode=False):
        d = json.loads(action_str)
        return d['arguments']

    def parse_response(self, model_response, restrict_mode=False):
        result = {}
        result = parse_tags(model_response, ['thinking', 'tool_call', 'conclusion'])
        action_content = self.parse_action(result['tool_call'], restrict_mode=True)
        result['action_content'] = action_content
        return result

    def format_response(self, step, image_ele, add_thought=True):
        action_content = step['action_content']
        model_result = {
            'action': f"<tool_call>\n{self.format_action(action_content, image_ele)}\n</tool_call>",
            STD_THINKING_KEY: f"<thinking>\n{step[STD_THINKING_KEY]}\n</thinking>" if step.get(STD_THINKING_KEY, None) is not None else None,
            STD_CONCLUSION_KEY: f"<conclusion>\n{step[STD_CONCLUSION_KEY]}\n</conclusion>" if step.get(STD_CONCLUSION_KEY, None) is not None else None,
        }
        model_response = ""
        if add_thought and STD_THINKING_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_THINKING_KEY]+'\n'
            
        model_response += model_result['action'] + '\n'
        if add_thought and STD_CONCLUSION_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_CONCLUSION_KEY]+'\n'
                    
        return model_response

    def gen_next_round(self, line, state, previous_model_response=None):
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            image_ele_0 = make_qwen_image_item(line['steps'][0]['screenshot'], image=line['steps'][0].get('screenshot_pil', None))

            # TODO 使用动态动作空间
            state['_si'] = 0
            messages = [get_computer_system_prompt(height=image_ele_0['resized_height'], width=image_ele_0['resized_width'], custom_use=self.custom_use)]
            state['messages'] = messages
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None

        line_can_thought = state['line_can_thought']
        messages = state['messages']
        step = line['steps'][si]
        image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))

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
        assert self.use_step_instruction == False
        
        

        thought_instruction = ''
        if state['line_can_thought']:
            thought_instruction += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
            thought_instruction += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'

        if si==0:
            query = line['goal']
            messages.append({
                "role": "user", 
                "content": [
                    {"text": f"""Please generate the next move according to the UI screenshot, instruction and previous actions.
Instruction: {query}
{thought_instruction}
"""},
                    image_ele
                    ]
            })
        else:
            messages.append({
                "role": "user", 
                "content": [
                    {"text": f"{thought_instruction}"},
                    image_ele
                    ]
            })

        messages_with_response = copy.deepcopy(messages)
        model_response = ''
        if 'action_content' in step:
            # 兼容fake line
            model_response = self.format_response(step, image_ele, add_thought=line_can_thought)
        messages_with_response.append({
            'role': "assistant",
            "content": [{"text": model_response}]
        })
        
        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)
        return state