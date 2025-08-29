# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities needed for task template variation study."""

from android_world.task_evals.composite import system as system_composite
from android_world.task_evals.single import audio_recorder
from android_world.task_evals.single import expense
from android_world.task_evals.single import markor
from android_world.task_evals.single import osmand
from android_world.task_evals.single import phone
from android_world.task_evals.single import simple_gallery_pro
from android_world.task_evals.single import sms
from android_world.task_evals.single.calendar import calendar

ADDITIONAL_TASK_TEMPLATES = {
    audio_recorder.AudioRecorderRecordAudioWithFileName.__name__: [
        (
            "Utilize the Audio Recorder app to capture an audio clip and save"
            ' it under the name "{file_name}".'
        ),
        (
            "Capture an audio recording and store it with the designated name"
            ' "{file_name}" using the Audio Recorder application.'
        ),
        (
            "Record an audio clip, ensuring its saved with the filename"
            ' "{file_name}" through the Audio Recorder app.'
        ),
        (
            "Employ the Audio Recorder app to record an audio clip, making sure"
            ' to save it with the name "{file_name}".'
        ),
    ],
    calendar.SimpleCalendarAddOneEventRelativeDay.__name__: [
        (
            "Using Simple Calendar Pro, schedule an event for this"
            " {day_of_week} at {hour} o'clock titled '{event_title}' with the"
            " description '{event_description}'. Ensure the event duration is"
            " {duration_mins} minutes."
        ),
        (
            "Create a calendar entry in Simple Calendar Pro for this"
            " {day_of_week} at {hour}h labeled '{event_title}' and detailed"
            " with '{event_description}'. Set the event duration to"
            " {duration_mins} minutes."
        ),
        (
            "Employ Simple Calendar Pro to set up a calendar event for this"
            " {day_of_week} at {hour}h with the title '{event_title}' and a"
            " brief '{event_description}'. Specify the event duration as"
            " {duration_mins} minutes."
        ),
        (
            "Establish a calendar event in Simple Calendar Pro for this"
            " {day_of_week} at {hour} o'clock, naming it '{event_title}' and"
            " describing it as '{event_description}'. Make sure the event lasts"
            " for {duration_mins} minutes."
        ),
    ],
    calendar.SimpleCalendarDeleteOneEvent.__name__: [
        (
            "Remove the calendar event titled '{event_title}' at {hour} o'clock"
            " on {year}-{month}-{day} from Simple Calendar Pro."
        ),
        (
            "Utilize Simple Calendar Pro to delete the event scheduled for"
            "{hour}h on {year}-{month}-{day} labeled '{event_title}'."
        ),
        (
            "In Simple Calendar Pro, erase the calendar event titled"
            " '{event_title}' occurring at {hour} o'clock with the date"
            " {year}-{month}-{day}'."
        ),
        (
            "Delete the calendar entry labeled '{event_title}' at {hour}h on"
            " {year}-{month}-{day} in Simple Calendar Pro."
        ),
    ],
    expense.ExpenseDeleteDuplicates.__name__: [
        (
            "Remove any duplicate expenses in Arduia Pro Expense, keeping at"
            " least one unique instance of each expense type."
        ),
        (
            "Ensure only one instance of each unique expense remains in Arduia"
            " Pro Expense by deleting all exact duplicates."
        ),
        (
            "In Arduia Pro Expense, remove redundant expenses such that only a"
            " single instance of each unique expense remains."
        ),
        (
            "Eliminate all duplicate expenses from Arduia Pro Expense,"
            " preserving at least one occurrence of each distinct expense."
        ),
    ],
    markor.MarkorCreateNote.__name__: [
        (
            "Generate a new note within Markor, assigning the title"
            " '{file_name}', and input the text: '{text}'."
        ),
        (
            "Initiate a new note in Markor titled {file_name}, and include the"
            " text: '{text}'."
        ),
        (
            "In Markor, establish a note named {file_name} and populate it with"
            " the text: {text}."
        ),
        (
            "Create a note named {file_name} in Markor, filling it with the"
            " content: {text}."
        ),
    ],
    markor.MarkorDeleteNewestNote.__name__: [
        "Remove the most recent note from Markor.",
        "Delete the latest note in Markor.",
        "Erase the newest note recorded in Markor.",
        "Remove the last-created note from Markor.",
    ],
    osmand.OsmAndMarker.__name__: [
        "In the OsmAnd maps app, create a location marker at {location}.",
        (
            "Place a location marker in OsmAnd maps for the location specified"
            " as {location}."
        ),
        (
            "Mark the location {location} on the OsmAnd maps app with a"
            " location marker."
        ),
        (
            "Set a location marker in OsmAnd maps to pinpoint the location"
            " {location}."
        ),
    ],
    phone.MarkorCallApartment.__name__: [
        (
            "Place a call to the phone number associated with the apartment"
            " labeled {name}, which can be found in the apartments.md file"
            " within Markor. Confirm that the In-Call UI is visible, offering"
            " features such as 'Hold'."
        ),
        (
            "Dial the contact number linked to the apartment named {name},"
            " located within the 'apartments.md' file in Markor. Verify that"
            " the In-Call UI appears, providing options like 'Hold'."
        ),
        (
            "Initiate a phone call to the number specified for the apartment"
            " identified as {name} in the 'apartments.md' file stored in"
            " Markor. Make sure the In-Call UI is shown, including"
            " functionalities such as 'Hold'."
        ),
        (
            "Contact the phone number corresponding to the apartment denoted by"
            " {name}. Retrieve this number from the apartments.md file in"
            " Markor. Ensure that the In-Call UI appears with features like"
            " 'Hold'."
        ),
    ],
    simple_gallery_pro.SaveCopyOfReceiptTaskEval.__name__: [
        (
            "Duplicate the file named '{file_name}' from DCIM and store it with"
            " the identical name in the Download directory."
        ),
        (
            "Make a copy of the {file_name} file located in DCIM and save it"
            " under the same name within the Download folder."
        ),
        (
            "Replicate the '{file_name}' file from DCIM and ensure a duplicate"
            " is saved with the matching name in the Download directory."
        ),
        (
            "Create a duplicate of {file_name} from DCIM and place it in the"
            " Download folder with the same name."
        ),
    ],
    sms.SimpleSmsReplyMostRecent.__name__: [
        (
            "Respond to the latest text message using Simple SMS Messenger,"
            " incorporating the message: '{message}'"
        ),
        (
            "Utilize Simple SMS Messenger to reply to the most recent text"
            " message with the content: {message}"
        ),
        (
            "Craft a response to the newest text message using Simple SMS"
            " Messenger, and include the message: '{message}'"
        ),
        (
            "Reply to the most recent SMS using Simple SMS Messenger,"
            " integrating the following message: {message}"
        ),
    ],
    sms.SimpleSmsResend.__name__: [
        (
            "Forward the message I recently sent to {name} using Simple SMS"
            " Messenger."
        ),
        (
            "Resend the message that was just sent to {name} via Simple SMS"
            " Messenger."
        ),
        (
            "Send again the message I just delivered to {name} using Simple SMS"
            " Messenger."
        ),
        (
            "Use Simple SMS Messenger to resend the message I sent to {name}"
            " moments ago."
        ),
    ],
    sms.SimpleSmsSendReceivedAddress.__name__: [
        (
            "Use Simple SMS Messenger to send {name1} the event address {name2}"
            " just shared with me."
        ),
        (
            "Please forward the event address sent by {name2} to {name1}"
            " through Simple SMS Messenger."
        ),
        (
            "Send the event location shared by {name2} to {name1} using Simple"
            " SMS Messenger."
        ),
        (
            "Transmit {name2}'s shared event address to {name1} via Simple SMS"
            " Messenger."
        ),
    ],
    system_composite.TurnOnWifiAndOpenApp.__name__: [
        "Activate the WiFi, then launch the {app_name} application.",
        "Switch on the WiFi and proceed to open the {app_name} application.",
        "Enable WiFi and access the {app_name} app.",
        "Power up the WiFi and initiate the {app_name} app.",
    ],
}
