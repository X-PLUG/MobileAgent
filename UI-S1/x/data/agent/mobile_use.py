
from x.data.agent.base import STD_CONCLUSION_KEY, STD_THINKING_KEY, BaseFormatAbs

from x.data.text import parse_tags
import json

from x.qwen.image import make_qwen_image_item
import copy
import traceback
from x.qwen.agent import get_mobile_system_prompt
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
class MobileUseSingleTurnFormat(BaseFormatAbs):
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
        if add_thought and 'conclusion' in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result['conclusion']+'\n'
                    
        return model_response

    def gen_next_round(self, line, state, previous_model_response=None):
        # assert previous_model_response is None
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            if line_can_thought:
                _format = 'thought_action'
            else:
                _format = 'only_action'
            image_ele_0 = make_qwen_image_item(line['steps'][0]['screenshot'], image=line['steps'][0].get('screenshot_pil', None))

            # TODO 使用动态动作空间
           
            state['_si'] = 0
            state['system'] = get_mobile_system_prompt(height=image_ele_0['resized_height'], width=image_ele_0['resized_width'], custom_use=self.custom_use)
            state['step_history'] = []
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None


        step_history = state['step_history']
        messages = []
        messages.append(state['system'])
        query = line['goal']
        step = line['steps'][si]

        if state['line_can_thought']:
            query += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
            if self.force_add_thought or STD_CONCLUSION_KEY in step:
                query += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'
        
        step_history_str = ''.join([f"Step {j+1}: {_}; " for j,_ in enumerate(step_history)])
        step_history_str = f"\nTask progress (You have done the following operation on the current device): {step_history_str}"

        if self.use_step_instruction:
            step_instriction = step['step_instruction']
            step_instriction = f"\nCurrent step query: {step_instriction}"
        else:
            step_instriction = ''
        
        image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
        messages.append({
            "role": "user", 
            "content": [
                {"text": f"The user query:  {query}{step_instriction}{step_history_str}"},
                image_ele
                ]
        })
        if self.use_step_instruction:
            step_history_content =  step['step_instruction']
        elif 'conclusion' in step:
            step_history_content =  step['conclusion']
        else:
            step_history_content = self.format_action(step['action_content'], image_ele)
        step_history.append(
            step_history_content
        )
        
        messages_with_response = copy.deepcopy(messages)
        model_response = ''
        if 'action_content' in step:
            # 兼容fake line
            model_response = self.format_response(step, image_ele, add_thought=state['line_can_thought'])
        messages_with_response.append({
            'role': "assistant",
            "content": [{"text": model_response}]
        })

        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)
        
        return state
       

class MobileUseMultiTurnFormat(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, use_step_instruction=False, repeat_query=False, use_add_info=False, custom_use=None):
        super().__init__(space_file, add_thought, force_add_thought)
        self.suffix = ''
        self.custom_use = custom_use
        self.use_step_instruction = use_step_instruction
        self.repeat_query = repeat_query
        self.use_add_info = use_add_info

        if self.repeat_query: # TODO useless maybe delete
            self.suffix += '_repeat_query'
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
        model_response = model_response.strip() 
        return model_response

    def gen_next_round(self, line, state, previous_model_response=None):
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            if line_can_thought:
                _format = 'thought_action'
            else:
                _format = 'only_action'
            image_ele_0 = make_qwen_image_item(line['steps'][0]['screenshot'], image=line['steps'][0].get('screenshot_pil', None))

            # TODO 使用动态动作空间
            state['_si'] = 0
            messages = [get_mobile_system_prompt(height=image_ele_0['resized_height'], width=image_ele_0['resized_width'], custom_use=self.custom_use)]
            state['messages'] = messages
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None

        line_can_thought = state['line_can_thought']
        messages = state['messages']
        image_ele = make_qwen_image_item(line['steps'][si]['screenshot'], image=line['steps'][si].get('screenshot_pil', None))

        if previous_model_response:
            messages.append({
                'role': "assistant",
                "content": [{"text": previous_model_response}]
            })
        else:
            if si-1>=0:
                previous_step = line['steps'][si-1]
                model_response = self.format_response(previous_step, image_ele, add_thought=line_can_thought)
                messages.append({
                    'role': "assistant",
                    "content": [{"text": model_response}]
                })
        
        step = line['steps'][si]
        if self.use_step_instruction:
            step_instriction = line['steps'][si]['step_instruction']
            step_instriction = f"\nCurrent step query: {step_instriction}"
        else:
            step_instriction = ''

        thought_instruction = ''
        if state['line_can_thought']:
            thought_instruction += '\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.'
            if self.force_add_thought or STD_CONCLUSION_KEY in step:
                thought_instruction += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'

        add_info_content = ""
        if self.use_add_info and line.get('add_info',''):
           add_info_content = "\nTips: \n"+line['add_info']+"\nThe above are some prompts that guide you to make decisions at specific steps."
        if si==0:
            query = line['goal']
            
            messages.append({
                "role": "user", 
                "content": [
                    {"text": f"The user query:  {query}{step_instriction}{add_info_content}{thought_instruction}"},
                    image_ele
                    ]
            })
        else:
            if self.repeat_query:
                query = line['goal']
                use_prompt = f"The user query:  {query}{step_instriction}{add_info_content}{thought_instruction}"
            else:
                use_prompt = f"{step_instriction}{thought_instruction}"
            messages.append({
                "role": "user", 
                "content": [
                    {"text": use_prompt},
                    image_ele
                    ]
            })

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

class MobileUseMultiTurnFormatV2(BaseFormatAbs):
    def __init__(self, space_file, add_thought, force_add_thought=False, use_step_instruction=False, repeat_query=False, use_add_info=False, custom_use=None):
        super().__init__(space_file, add_thought, force_add_thought)
        self.suffix = ''
        self.custom_use = custom_use
        self.use_step_instruction = use_step_instruction
        self.repeat_query = repeat_query
        self.use_add_info = use_add_info

        if self.repeat_query: # TODO useless maybe delete
            self.suffix += '_repeat_query'
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
        result = parse_tags(model_response, ['think', 'tool_call', 'conclusion'])
        action_content = self.parse_action(result['tool_call'], restrict_mode=True)
        result['action_content'] = action_content
        return result

    def format_response(self, step, image_ele, add_thought=True):
        action_content = step['action_content']
        model_result = {
            'action': f"<tool_call>\n{self.format_action(action_content, image_ele)}\n</tool_call>",
            STD_THINKING_KEY: f"<think>\n{step[STD_THINKING_KEY]}\n</think>" if step.get(STD_THINKING_KEY, None) is not None else None,
            STD_CONCLUSION_KEY: f"<conclusion>\n{step[STD_CONCLUSION_KEY]}\n</conclusion>" if step.get(STD_CONCLUSION_KEY, None) is not None else None,
        }
        model_response = ""
        if add_thought and STD_THINKING_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_THINKING_KEY]+'\n'
            
        model_response += model_result['action'] + '\n'
        if add_thought and STD_CONCLUSION_KEY in step: # 这里支持模型在没有thought内容的时候生成thought prompt
            model_response += model_result[STD_CONCLUSION_KEY]+'\n'
        model_response = model_response.strip() 
        return model_response

    def gen_next_round(self, line, state, previous_model_response=None):
        if state == None:
            state = {}
            # 构造system
            line_can_thought = self.can_thought(line)
            if line_can_thought:
                _format = 'thought_action'
            else:
                _format = 'only_action'
            image_ele_0 = make_qwen_image_item(line['steps'][0]['screenshot'], image=line['steps'][0].get('screenshot_pil', None))

            # TODO 使用动态动作空间
            state['_si'] = 0
            messages = [get_mobile_system_prompt(height=image_ele_0['resized_height'], width=image_ele_0['resized_width'], custom_use=self.custom_use)]
            state['messages'] = messages
            state['line_can_thought'] = line_can_thought
        else:
            state = copy.deepcopy(state)
        
        si = state['_si']
        if si >= len(line['steps']):
            return None

        line_can_thought = state['line_can_thought']
        messages = state['messages']
        image_ele = make_qwen_image_item(line['steps'][si]['screenshot'], image=line['steps'][si].get('screenshot_pil', None))

        if previous_model_response:
            messages.append({
                'role': "assistant",
                "content": [{"text": previous_model_response}]
            })
        else:
            if si-1>=0:
                previous_step = line['steps'][si-1]
                model_response = self.format_response(previous_step, image_ele, add_thought=line_can_thought)
                messages.append({
                    'role': "assistant",
                    "content": [{"text": model_response}]
                })
        
        step = line['steps'][si]
        if self.use_step_instruction:
            step_instriction = line['steps'][si]['step_instruction']
            step_instriction = f"\nCurrent step query: {step_instriction}"
        else:
            step_instriction = ''

        thought_instruction = ''
        if state['line_can_thought']:
            thought_instruction += '\nBefore answering, explain your reasoning step-by-step in <think></think> tags, and insert them before the <tool_call></tool_call> XML tags.'
            if self.force_add_thought or STD_CONCLUSION_KEY in step:
                thought_instruction += '\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.'

        add_info_content = ""
        if self.use_add_info and line.get('add_info',''):
           add_info_content = "\nTips: \n"+line['add_info']+"\nThe above are some prompts that guide you to make decisions at specific steps."
        if si==0:
            query = line['goal']
            
            messages.append({
                "role": "user", 
                "content": [
                    {"text": f"The user query:  {query}{step_instriction}{add_info_content}{thought_instruction}"},
                    image_ele
                    ]
            })
        else:
            if self.repeat_query:
                query = line['goal']
                use_prompt = f"The user query:  {query}{step_instriction}{add_info_content}{thought_instruction}"
            else:
                use_prompt = f"{step_instriction}{thought_instruction}"
            messages.append({
                "role": "user", 
                "content": [
                    {"text": use_prompt},
                    image_ele
                    ]
            })

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
