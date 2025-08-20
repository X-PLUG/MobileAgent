# PC
def get_subtask_prompt_cn(instruction):
    func_prompt = '''多模态agent通过执行点击、输入等一系列操作来完成用户的指令。
用户指令可能由跨越多个应用程序的数个子任务组成，我希望你能将这个复杂的指令，分解为一些子任务，子任务有4种类型：
1. 常规字符串形式：例如“在系统设置中，打开深色模式”；
2. 包含字典内容的字符串：当前子任务的执行结果需要以字典方式传递给其他子任务，例如“在Outlook中，查看‘Paul’发来的邮件信息，以dict形式输出{'contact': 'Paul', 'mail_content': 'content of the email'}”；
3. 格式化字符串：利用前序子任务传递的信息，补全当前子任务后，能够完全独立执行，例如“将{mail_content}通过短信发送给‘Joey’”；
4. 包含字典内容的格式化字符串：既需要前序子任务传递的信息，以补全当前子任务，同时执行的结果也需要以字典方式传递给其他子任务，例如“在谷歌中搜索{question}，并将相关信息以dict形式输出{'info': 'related information'}”。

举例来说，复合指令“系统设置中打开深色模式，在微信中查看‘John’发来的问题，在Chrome中搜索问题的答案，将答案添加到一个新建word文档中，保存为‘作业.docx’，然后发送给‘John’。”可以被分解为：
{
"subtask 1": "在系统设置中，打开深色模式",
"subtask 2": "在微信中，查看‘John’发来的问题，将问题以dict形式输出{'John_question': 'content of the question'}",
"subtask 3": "在Chrome中，搜索{John_question}，将搜索到的答案以dict形式输出{'John_question_answer': 'answer to the question'}",
"subtask 4": "在Word中，新建一个文档，写入{John_question_answer}，并保存为‘作业.docx’",
"subtask 5": "在微信中，发送‘作业.docx’给‘John’"
}

需要注意：
1. 包含字典内容的字符串或格式化字符串，需要尽可能详细地说明dict中各个key的含义，即将哪些内容以dict的形式输出；
2. 每个格式化字符串形式的子任务中包含的key，在前序子任务中要有对应的dict形式输出，也就是说，前序子任务执行完成后，保证当前子任务能够通过参数传递得到补全，从而可以独立执行。
3. 必须保证，每个子任务，无论是常规字符串，还是补全之后的格式化字符串，能够完全脱离其他子任务独立执行。例如“在Word中新建一个文档，写入{John_question_answer}”可以独立执行，但“将修改后的Word文档通过邮件发送给{name}”则因为‘Word文档’指代不明确无法独立执行。
4. 拆解后的每个子任务要有明确的应用程序，例如‘在Chrome中’、‘在Word中’等。一般而言，docx格式文档用Word程序打开，xlsx格式表格用Excel程序打开。此外，需要打开文件时，要明确文件的名字。
'''
    
    inst_prompt = '''
User Instruction:
{}
'''

    format_prompt = '''
请你按照如下格式输出拆分后的子任务：
{
"subtask 1": ,
"subtask 2": ,
...
}
'''
    prompt = func_prompt + inst_prompt.format(instruction) + format_prompt
    return prompt



def get_subtask_prompt(instruction):
    func_prompt = '''A multi-modal agent completes a user's instruction by performing a series of actions such as clicking and typing. A user's instruction may consist of multiple subtasks across different applications. I want you to break down this complex instruction into several subtasks, which are of four types:

1. Regular string: For example, "Open dark mode in system settings";
2. String containing dictionary content: The result of the current subtask needs to be passed to other subtasks in a dictionary format, for example, "Check the emails from 'Paul' in Outlook and output the email details in a dict format like {'contact': 'Paul', 'mail_content': 'content of the email'}";
3. Formatted string containing the keys from previous subtasks: Use the information from previous subtasks to complete and independently execute the current subtask, for example, "Send {mail_content} via SMS to 'Joey'". Note: Note: The text in the first "{""}" must be a key from the output of a previous subtask, and there should be no "''";
4. Formatted string containing the keys from previous subtasks and the dictionary content: This requires both information from previous subtasks to complete the current subtask and the result also needs to be passed to other subtasks in a dictionary format, for example, "Search for {question} on Google and output the relevant information in a dict format like {'info': 'related information'}". Note: The text in the first "{""}" must be a key from the output of a previous subtask, and there should be no "''".


For example, the compound instruction "Open dark mode in system settings, check the two questions sent by 'John' in WeChat, search for answers to these two questions in Chrome, add the answers to a new Word document, save it as 'assignment.docx', and then send it to 'John'." can be broken down into:
{
"subtask 1": "Open dark mode in system settings",
"subtask 2": "Check the questions sent by 'John' in WeChat and output the questions in a dict format {'John_question_1': 'content of John\'s question_1', 'John_question_2': 'content of John\'s question_2'}",
"subtask 3": "Search for {John_question_1} in Chrome and output the found answer in a dict format {'John_question_1_answer': 'answer to the question_1'}",
"subtask 4": "Search for {John_question_2} in Chrome and output the found answer in a dict format {'John_question_2_answer': 'answer to the question_2'}",
"subtask 5": "Create a new document in Word, write {John_question_1_answer} and {John_question_2_answer} sequentially, then save it as 'assignment.docx'",
"subtask 6": "Send 'assignment.docx' to 'John' via WeChat"
}

Notes:
1. Strings or formatted strings containing dictionary content should explain as detailed as possible the meaning of each key in the dict, i.e., what content should be output in dict form;
2. Each key in a formatted string subtask must have a corresponding dict output in preceding subtasks, ensuring that after a preceding subtask is completed, the current subtask can be fully completed through parameter passing and thus executed independently.
3. Ensure each subtask, whether as a regular string or a completed formatted string, can be executed independently of other subtasks. For example, "Create a new document in Word and write {John_question_answer}" can be executed independently, but "Send the modified Word document via email to {name}" cannot because "Word document" is ambiguous and cannot be executed independently.
4. Each subtask must specify a clear application, such as 'in Chrome' or 'in Word'. Generally, docx formatted documents are opened with Word, and xlsx spreadsheets are opened with Excel. Additionally, when opening a file, clearly state the file name.
5. Note that if a subtask contains a dict, ensure that the values in the dictionary do not contain single quote characters to avoid format errors.
'''
    
    inst_prompt = '''
User Instruction:
{}
'''

    format_prompt = '''
Please output the split subtasks in the following format:
{
"subtask 1": ,
"subtask 2": ,
...
}
'''
    prompt = func_prompt + inst_prompt.format(instruction) + format_prompt
    return prompt




def get_select_prompt(content):
    prompt_template = '''
Analyze the specified text range {} and output the first line and last line of the specified range separately. 
How to identify paragraphs: There are 2 spaces at the beginning of each paragraph. Define the title as the single line at the top. 
If the content has only one line (such as title), it is both the first and last line.'''

    prompt_format = '''
You should respond in the following format:
<first>The content of the first line</first>
<last>The content of the last line</last>
'''
    prompt = prompt_template.format(content)+prompt_format
    return prompt




def get_select_prompt_simple(content):
    prompt_template = '''
Analyze the text range of this part of the current Word document: {}, and output the content of the first and last lines separately.
If the content has only one line in total, this line is the first line and also the last line.'''

    prompt_format = '''
You should respond in the following format:
<first>The content of the first line</first>
<last>The content of the last line</last>
'''
    prompt = prompt_template.format(content)+prompt_format
    return prompt



def get_select_prompt_backup(content):
    prompt_template = '''
Directly output the first line and the last line of the content: {} in the current shown Microsoft Word document. If the content has only one line, output this line twice.'''

    prompt_format = '''
You should respond in the following format:
<first>The content of the first line</first>
<last>The content of the last line</last>
'''
    prompt = prompt_template.format(content)+prompt_format
    return prompt


def get_action_prompt(instruction, clickable_infos, width, height, thought_history, summary_history, action_history, reflection_history, last_summary, last_action, reflection_thought, add_info, error_flag, completed_content, memory):
    prompt = "### Background ###\n"
    prompt += f"This image is a computer screenshot where icons are marked with numbers. Its width is {width} pixels and its height is {height} pixels. The user\'s instruction is: {instruction}.\n\n"

    prompt += add_info
    prompt += "\n\n"
    
    prompt += "### Screenshot information ###\n"
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information of the current screenshot. "
    prompt += "This information consists of two parts: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom; "


    prompt += "the content is a text or 'icon' respectively. "
    prompt += "The information is as follow:\n"

    for clickable_info in clickable_infos:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    
    
    if len(action_history) > 0:
        prompt += "### History operations ###\n"
        prompt += "Before arriving at the current screenshot, you have completed the following operations:\n"
        for i in range(len(action_history)):
            if len(reflection_history) > 0:
                prompt += f"Step-{i+1}: [Operation: " + summary_history[i].split(' to ')[0].strip() + "; Action: " + action_history[i] + "; Reflection: " + reflection_history[i] + "]\n"
            else:
                prompt += f"Step-{i+1}: [Operation: " + summary_history[i].split(' to ')[0].strip() + "; Action: " + action_history[i] + "]\n"
        prompt += "\n"
    
    if completed_content != "":
        prompt += "### Progress ###\n"
        prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        prompt += "Completed contents:\n" + completed_content + "\n\n"
    
    if memory != "":
        prompt += "### Memory ###\n"
        prompt += "During the operations, you record the following contents on the screenshot for use in subsequent operations:\n"
        prompt += "Memory:\n" + memory + "\n"
    

    # 禁用
    if error_flag:
        prompt += "### Last operation ###\n"
        prompt += f"You previously wanted to perform the operation \"{last_summary}\" on this page and executed the Action \"{last_action}\". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time."
        prompt += "\n\n"
        print(f"You previously wanted to perform the operation \"{last_summary}\" on this page and executed the Action \"{last_action}\". But you find that this operation does not meet your expectation. You need to reflect and revise your operation this time.")
    
    prompt += "### Task requirements ###\n"
    prompt += "In order to meet the user\'s requirements, you need to select one of the following operations to operate on the current screen:\n"
    prompt += "Note that to open an app, use the Open App action, rather than tapping the app's icon. "
    prompt += "For certain items that require selection, such as font and font size, direct input is more efficient than scrolling through choices."
    prompt += "You must choose one of the actions below:\n"
    prompt += "Open App (app name): If you want to open an app, you should use this action to open the app named 'app name'."
    prompt += "Right Tap (x, y): Right tap the position (x, y) in current page. This can be used to create a new file.\n"
    prompt += "Tap (x, y): Tap the position (x, y) in current page. This can be used to select an item.\n"
    prompt += "Double Tap (x, y): Double tap the position (x, y) in the current page. This can be used to open a file. If Tap (x, y) in the last step doesn't work, you can try double tap the position (x, y) in the current page.\n"


    prompt += '''
    Shortcut (key1, key2): There are several shortcuts (key1+key2) you may use.
    For example, if you can't find the download button, use command+s to save the page or download the file.
    To select all, you can use command+a.
    To create a new file in Word/Excel, you can use command+n.
    To create a new tab for starting a new search in Chrome, you can use command+t.
    To copy an item, you can first select it and then use command+c.
    To paste the copied item, you can first select the location you want to paste it to, and then use command+v.
    '''
    prompt += '''
    Press (key name): There are several keys that may help.
    For example, if you want to delete the selected content, press 'backspace'.
    You can press 'enter' to confirm, submit the input command, or insert a line break.
    Also, you can press 'up', 'down', 'left', or 'right' to scroll the page or adjust the position of the selected object.
    '''

    prompt += "Type (x, y), (text): Tap the position (x, y) and type the \"text\" in the input box and press the enter key. You should replace the \"text\" with the actual input.\n"

    prompt += "Select (content): Select the referred 'content' in the current document, such as 'title', 'the second paragraph' and 'the last two paragraphs'. This action is useful when you want to edit a certain part of the document, such as bolding, adding underlines, changing line spacing, centering text, etc.\n"
    prompt += "Replace (x, y), (text): Replace the editable content in (x, y) with the \"text\". You should replace the \"text\" with the actual input. This action is very useful when you want to start a new search in Chrome or rename a file.\n"
    prompt += "Append (x, y), (text): Append the \"text\" content after the content at (x, y) location. This action is useful when you want to append new content into a word document.\n"

    prompt += "Tell (answer): Tell me the answer of the input query.\n"
    prompt += "Stop: If all the operations to meet the user\'s requirements have been completed in ### History operation ###, use this operation to stop the whole process."
    prompt += "\n\n"

    prompt += "### Output format ###\n"
    # modified 2.10
    prompt += "You should output in the following json format:"
    prompt += '''
{"Thought": "This is your thinking about how to proceed the next operation, please output the thoughts about the history operations explicitly.", "Action": "Open App () or Tap () or Double Tap () or Triple Tap () or Shortcut () or Press() or Type () or Tell () or Stop. Only one action can be output at one time.", "Summary": "This is a one sentence summary of this operation."}
'''
    prompt += "\n\n"


    # print(prompt)

    return prompt


def get_reflect_prompt(instruction, clickable_infos1, clickable_infos2, width, height, summary, action, add_info, no_image=0):
    if no_image == 1:
        prompt = f"The computer screen's width is {width} pixels and the height is {height} pixels.\n\n"
    else:
        prompt = f"These images are two computer screenshots before and after an operation. Their widths are {width} pixels and their heights are {height} pixels.\n\n"
    
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot. "
    prompt += "The information consists of two parts, consisting of format: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom; the content is a text or an icon description respectively "
    prompt += "\n\n"
    
    prompt += "### Before the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos1:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    prompt += "\n\n"
            
    prompt += "### After the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos2:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    prompt += "\n\n"
    
    prompt += "### Current operation ###\n"
    prompt += f"The user\'s instruction is: {instruction}."
    if add_info != "":
        prompt += f"You also need to note the following requirements: {add_info}."
    prompt += "In the process of completing the requirements of instruction, an operation is performed on the computer. Below are the details of this operation:\n"
    prompt += "Operation thought: " + summary.split(" to ")[0].strip() + "\n"
    prompt += "Operation action: " + action
    prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    if no_image == 1:
        prompt += "Now you need to output the following content based on the screenshots information before and after the current operation:\n"
    else:
        prompt += "Now you need to output the following content based on the screenshots before and after the current operation:\n"
    prompt += "Whether the result of the \"Operation action\" meets your expectation of \"Operation thought\"?\n"
    prompt += "A: The result of the \"Operation action\" meets my expectation of \"Operation thought\".\n"
    prompt += "B: The \"Operation action\" results in a wrong page and I need to do something to correct this.\n"
    prompt += "C: The \"Operation action\" produces no changes."
    prompt += "\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Thought ###\nYour thought about the question\n"
    prompt += "### Answer ###\nA or B or C"
    
    return prompt


def get_memory_prompt(insight):
    if insight != "":
        prompt  = "### Important content ###\n"
        prompt += insight
        prompt += "\n\n"
    
        prompt += "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to ### Important content ### on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    else:
        prompt  = "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to user\'s instrcution on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Important content ###\nThe content or None. Please do not repeatedly output the information in ### Memory ###."
    
    return prompt

def get_process_prompt(instruction, thought_history, summary_history, action_history, completed_content, add_info, reflection_history=[]):
    prompt = "### Background ###\n"
    prompt += f"There is an user\'s instruction which is: {instruction}. You are a computer operating assistant and are operating the user\'s computer.\n\n"
    
    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the user\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"
    
    if len(thought_history) > 1:
        prompt += "### History operations ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed a series of operations. These operations are as follow:\n"
        for i in range(len(summary_history)):
            operation = summary_history[i].split(" to ")[0].strip()
            if len(reflection_history) > 0:
                prompt += f"Step-{i+1}: [Operation thought: " + operation + "; Operation action: " + action_history[i] + "; Operation reflection: " + reflection_history[i] + "]\n"
            else:
                prompt += f"Step-{i+1}: [Operation thought: " + operation + "; Operation action: " + action_history[i] + "]\n"
        prompt += "\n"

        prompt += "### Progress thinking ###\n"
        prompt += "After completing the history operations, you have the following thoughts about the progress of user\'s instruction completion:\n"
        prompt += "Completed contents:\n" + completed_content + "\n\n"
        
        prompt += "### Response requirements ###\n"
        prompt += "Now you need to update the \"Completed contents\". Completed contents is a general summary of the current contents that have been completed based on the ### History operations ###.\n\n"
        
        prompt += "### Output format ###\n"
        prompt += "Your output format is:\n"
        prompt += "### Completed contents ###\nUpdated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### History operations ###."
        
    else:
        prompt += "### Current operation ###\n"
        prompt += "To complete the requirements of user\'s instruction, you have performed an operation. Your operation thought and action of this operation are as follows:\n"
        prompt += f"Operation thought: {thought_history[-1]}\n"
        operation = summary_history[-1].split(" to ")[0].strip()
        if len(reflection_history) > 0:
            prompt += f"Operation action: {operation}\n" + "Operation reflection: " + reflection_history[-1] + "\n\n"
        else:
            prompt += f"Operation action: {operation}\n\n"
        
        # if reflection_thought is not None:
        #     prompt += "A reflection model was adopted to analyze whether the last step's operation meets the expectation, you should combine its reflection thought to produce the \"Completed contents\"."
        #     prompt += "Below is its reflection thought:\n"
        #     prompt += reflection_thought + "\n"

        prompt += "### Response requirements ###\n"
        prompt += "Now you need to combine all of the above to generate the \"Completed contents\".\n"
        prompt += "Completed contents is a general summary of the current contents that have been completed. You need to first focus on the requirements of user\'s instruction, and then summarize the contents that have been completed.\n\n"
        
        prompt += "### Output format ###\n"
        prompt += "Your output format is:\n"
        prompt += "### Completed contents ###\nGenerated Completed contents. Don\'t output the purpose of any operation. Just summarize the contents that have been actually completed in the ### Current operation ###.\n"
        prompt += "(Please use English to output)"
        
    return prompt