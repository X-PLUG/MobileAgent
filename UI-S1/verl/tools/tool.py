class Tool:
    def __init__(self, task_id,instruction, memory=None):
        self.task_id = task_id
        self.instruction = instruction
        self.memory = memory

    

    def get_prompt(self, instruction):
        # 可以根据实际需求实现
        prompt = f"""
            You are a GUI Agent Tool. Given a user instruction, your task is to extract the underlying task and output the appropriate tool name along with its arguments.

            ## User Instruction:
            {instruction}

            ## Tools Available:
            - manager: A tool for managing multiple subtasks when a task needs to be broken down. For example, use this when the instruction contains several repeated subtasks.
            - recorder: A tool for recording observed information during the execution. For example, use this when the task needs storing intermediate information. 
            Init the argument as empty at the beginning of the task.

            ## Tool Arguments:
            - list: A list of subtasks or observed information.
            - dict: A dictionary containing observed information.

            Format your output as a JSON object with the chosen tool and its arguments at the same level.

            ### Example outputs:
            <tool>
            {{"tool": "manager", "list": ["A", "B", "C"]}}
            </tool>

            <tool>
            {{"tool": "recorder", "dict": {{"A": 100, "B": 200, "C": ""}}}}
            </tool>
        """.strip()
        return prompt


class Manager(Tool):
    def __init__(self, task_id,instruction, memory=None):
        super().__init__(task_id,instruction, memory)
        self.current_idx = 0
        self.memory = 
    def extract_memory(self):
        return self.memory
    def get_observation(self):
        Observation = f"""
        # Finished Task: {" ,".join(self.memory[:self.current_idx])}
        # Current Task: You are now dealing with {self.memory[self.current_idx]}
        """
        # 可以根据实际需求实现
        return Observation
    def next(self):
        self.current_idx += 1
    # 可以扩展Manager专属方法


class Calculator(Tool):
    def __init__(self, task_id,instruction, memory=None):
        super().__init__(task_id,instruction, memory)

    # 可以扩展Calculator专属方法


class Recorder(Tool):
    def __init__(self, task_id,instruction, memory=None):
        super().__init__(task_id,instruction, memory)
    
    def 
    # 可以扩展Recorder专属方法


# 用法举例
if __name__ == "__main__":
    mgr = Manager(task_id='123', memory={"todos": []})
    calc = Calculator(task_id='124')
    rec = Recorder(task_id='125', memory={'logs': []})

    print(mgr.get_observation())
    print(calc.get_prompt("What is 2+2?"))
    print(rec.memory)
