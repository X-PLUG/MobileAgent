
import copy
from x.data.text import parse_tags
from x.io.json import read_json
from x.qwen.data_format import slim_messages
from x.qwen.image import resize_coordinate

def deal_with_coordinate(action_content, image_ele):
    width, height = image_ele['width'], image_ele['height']
    resized_width, resized_height = image_ele['resized_width'], image_ele['resized_height']
    
    if 'coordinate' in action_content:
        action_content['coordinate'] = resize_coordinate(action_content['coordinate'], (width, height), (resized_width, resized_height))
        action_content['coordinate'] = list(map(round, action_content['coordinate']))
    if 'coordinate2' in action_content:
        action_content['coordinate2'] = resize_coordinate(action_content['coordinate2'], (width, height), (resized_width, resized_height)) 
        action_content['coordinate2'] = list(map(round, action_content['coordinate2']))
    return action_content

STD_THINKING_KEY = 'thought'
STD_CONCLUSION_KEY = 'conclusion'


def collect_multi_rounds(line, fm_mr):
    state = None
    messages_list = []
    while True:
        state = fm_mr.gen_next_round(line, state)
        if state == None:
            break
        messages_list.append(slim_messages(state['messages_with_response'], num_image_limit=2))
    return messages_list

def collect_multi_rounds_for_static_training(line, fm_mr):
    state = None
    messages_list = []
    model_response = None
    while True:

        state = fm_mr.gen_next_round(line, state, previous_model_response=model_response)
        
        if state == None:
            break
        model_response = state['messages_with_response'][-1]['content'][-1]['text']
        model_action = parse_tags(model_response, ['tool_call'])['tool_call']
        assert model_action
        messages_list.append(slim_messages(state['messages_with_response'], num_image_limit=2))
    return messages_list

def collect_single_rounds(line, fm_sr):
    state = None
    messages_list = []
    while True:
        state = fm_sr.gen_next_round(line, state)
        if state == None:
            break
        messages_list.append(state['messages_with_response'])
    return messages_list

class BaseFormatAbs():
    def __init__(self, space_file, add_thought, force_add_thought=False):
        if isinstance(space_file, dict):
            self.space = space_file
        else:
            self.space = read_json(space_file)
        self.add_thought = add_thought
        self.force_add_thought = force_add_thought
        
    def can_thought(self, line):
        if self.force_add_thought:
            return True

        if not self.add_thought:
            return False
        for step in line['steps']:
            if STD_THINKING_KEY not in step:
                return False
        return True

    def _gen_round_post(self, line, state, image_ele, si, messages, messages_with_response):
        
        state['step_id'] = si
        state['screenshot_ele'] = image_ele
        state['action_content'] = line['steps'][si]['action_content']
        if 'check_options' in line['steps'][si]:
            state['check_options'] = line['steps'][si]['check_options']
        state['_si'] = si + 1
        state['messages'] = messages
        state['messages_with_response'] = messages_with_response

    def _format_action_base(self, action_content, image_ele):
        action = copy.deepcopy(action_content)
        return deal_with_coordinate(action, image_ele)
    
    def format_action(self, action_content, image_ele):
        # return self._format_action_base(action_content, image_ele)
        raise NotImplementedError

    def format_response(self, step, image_ele, add_thought=True):
        raise NotImplementedError
    def parse_response(self, model_response, restrict_mode=False):
        raise NotImplementedError

    def parse_action(self, action_str, restrict_mode=False):
        raise NotImplementedError

    def build_system_prompt(self, line):
        raise NotImplementedError