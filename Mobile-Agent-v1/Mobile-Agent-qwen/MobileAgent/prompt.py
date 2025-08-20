thought_prompt = "Based on this screenshot, please analyze the current progress and what the next step is."

action_prompt = '''Based on the above analysis, please select the most appropriate action from the following 9 actions:
open app (parameter): Click on the name of an App on the desktop home page. The parameter is the name of App. You can only use this action on the desktop.
tap text (parameter): The parameter is the text you need to click. If there is text at the click position, use this action in preference.
tap icon (parameter1, parameter2): The parameter1 is the description of the icon you want to click. The parameter2 selected 1 out of 5 from the top, bottom, left, right and center, represents the general location of the icon on the screenshot.
scroll up: Scroll up the current page.
scroll down: Scroll down the current page.
type (parameter): The parameter is what you want to type. Make sure you have clicked on the input box before typing.
back: Back to the previous page.
exit: Exit the app and go back to the desktop.
stop: If the instruction has been completed, stop the process.
Please select just one action with parameter if necessary.'''

format_prompt = '''Your output is not in the required format. Please note the use of parentheses. Your output must be one of the following format:
open app (parameter)
tap text (parameter)
tap icon (parameter1, parameter2)
scroll up
scroll down
type (parameter)
back
exit
stop'''