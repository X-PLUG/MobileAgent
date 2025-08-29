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

"""Tasks for Markor app."""

import dataclasses
import datetime
import random
from typing import Any

from absl import logging
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.single import vlc
from android_world.task_evals.utils import receipt_generator
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils
from android_world.utils import fuzzy_match_lib


@dataclasses.dataclass(frozen=True)
class _Note:
  name: str
  content: str


generate_random_sentence = lambda: random.choice(
    user_data_generation.RANDOM_SENTENCES
)


def _generate_random_note() -> _Note:
  """Generates a random note."""
  extensions = [".md", ".txt"]
  random_file_name = (
      user_data_generation.generate_random_file_name()
      + random.choice(extensions)
  )
  return _Note(random_file_name, generate_random_sentence())


class Markor(task_eval.TaskEval):
  app_names = ("markor",)

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class MarkorMoveNote(Markor):
  """Task for checking that a file has been moved in Markor."""

  complexity = 1.4
  schema = file_validators.MoveFile.schema
  template = (
      "In Markor, move the note {file_name} from {source_folder} to"
      " {destination_folder}."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.move_file_task = file_validators.MoveFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.move_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.move_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    subfolders = [
        "BookNotes",
        "CodeSnippets",
        "DailyNotes",
        "FitnessPlans",
        "MeetingMinutes",
        "PersonalJournal",
        "RecipeCollections",
        "StudyGuides",
        "TravelItineraries",
        "WorkProjects",
    ]
    source_folder = random.choice(subfolders)
    destination_folder = random.choice(
        [folder for folder in subfolders if folder != source_folder]
    )
    file_name = _generate_random_note().name
    return {
        "file_name": file_name,
        "source_folder": source_folder,
        "destination_folder": destination_folder,
        "noise_candidates": _NOTE_TITLES,
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.move_file_task.tear_down(env)


class MarkorCreateFolder(Markor):
  """Task for checking that a new folder in Markor has been created with a specific name."""

  complexity = 1
  schema = {
      "type": "object",
      "properties": {
          "folder_name": {"type": "string"},
      },
      "required": ["folder_name"],
  }
  template = "Create a new folder in Markor named {folder_name}."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        "file",
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    folder_name = self.params["folder_name"]

    exists = file_utils.check_file_or_folder_exists(
        folder_name, device_constants.MARKOR_DATA, env.controller
    )

    if not exists:
      logging.info("%s not found", folder_name)
      return 0.0

    return 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    random_folder_name = "folder_" + str(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return {"folder_name": random_folder_name}


class MarkorEditNote(Markor):
  """Task for editing an existing note in Markor."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "header": {"type": "string"},
          "footer": {"type": "string"},
          "replace_text": {"type": "string"},
          "edit_type": {
              "type": "string",
              "enum": ["header", "footer", "replace"],
          },
      },
      "required": ["file_name", "edit_type"],
  }

  @property
  def template(self) -> str:
    templates = {
        "header": (
            "Edit {file_name} in Markor. Add to the top of the note {header}"
        ),
        "footer": (
            "Edit {file_name} in Markor. Add to the bottom of the note {footer}"
        ),
        "replace": (
            "Edit {file_name} in Markor. Replace the text with {replace_text}"
        ),
    }

    if "edit_type" not in self.params and "edit_type" not in templates:
      return templates.get(
          self.params.get("edit_type"),
          "Invalid edit_type for {file_name} in Markor.",
      )
    return templates[self.params.get("edit_type")]

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )
    self.original_content = file_utils.create_file(
        self.params["file_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=generate_random_sentence(),
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        [
            "shell",
            "cat",
            file_utils.convert_to_posix_path(
                device_constants.MARKOR_DATA, self.params["file_name"]
            ),
        ],
        env.controller,
    )
    file_contents = res.generic.output.decode().replace("\r", "").strip()
    logging.info("Retrieved file contents: %s", file_contents)

    if self.params["edit_type"] == "header":
      expected_content = self.params["header"] + "\n" + self.original_content
    elif self.params["edit_type"] == "footer":
      expected_content = self.original_content + "\n" + self.params["footer"]
    else:
      expected_content = self.params["replace_text"]

    is_match = fuzzy_match_lib.fuzzy_match(file_contents, expected_content)
    logging.info(
        "Is content match: %s.\nFound: %s\nExpected: %s",
        is_match,
        file_contents,
        expected_content,
    )

    return 1.0 if is_match else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    extensions = [".md", ".txt"]

    random_file_name = (
        "note_"
        + user_data_generation.generate_random_string(5)
        + random.choice(extensions)
    )

    edit_type = random.choice(["header", "footer", "replace"])

    params = {
        "file_name": random_file_name,
        "edit_type": edit_type,
    }

    if edit_type == "header":
      params["header"] = generate_random_sentence()
    elif edit_type == "footer":
      params["footer"] = generate_random_sentence()
    elif edit_type == "replace":
      params["replace_text"] = "\n".join(
          [generate_random_sentence() for _ in range(3)]
      )

    return params


class MarkorDeleteNote(Markor):
  """Task for checking that a note in Markor has been deleted."""

  complexity = 1
  schema = file_validators.DeleteFile.schema
  template = "Delete the note in Markor named {file_name}."

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.delete_file_task = file_validators.DeleteFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.delete_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.delete_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    file_name = user_data_generation.generate_random_file_name()
    return {"file_name": file_name, "noise_candidates": _NOTE_TITLES}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.delete_file_task.tear_down(env)


class MarkorDeleteNewestNote(Markor):
  """Task for deleting the newest note in Markor."""

  complexity = 1
  schema = {}
  template = "Delete the newest note in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    # Generate some random notes in Markor.
    for _ in range(random.randint(2, 6)):
      note = _generate_random_note()
      file_utils.create_file(
          note.name,
          device_constants.MARKOR_DATA,
          env.controller,
          content=note.content,
      )
      # Advance system time so the change time for these initial notes can be
      # separated.
      datetime_utils.advance_system_time(
          datetime.timedelta(minutes=random.randint(-500, 500)), env.controller
      )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    self.initial_file_list_sorted = sorted(
        file_list, key=lambda f: f.change_time
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    new_file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    new_file_list_sorted = sorted(new_file_list, key=lambda f: f.change_time)
    for i in range(len(new_file_list)):
      # Both file lists are ordered by file change time, so by simply checking
      # file names and their change time are the same, we can ensure all other
      # files have not been changed.
      if not (
          new_file_list_sorted[i].file_name
          == self.initial_file_list_sorted[i].file_name
          and new_file_list_sorted[i].change_time
          == self.initial_file_list_sorted[i].change_time
      ):
        return 0.0
    one_fewer_file = (
        len(new_file_list_sorted) == len(self.initial_file_list_sorted) - 1
    )
    return 1.0 if one_fewer_file else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}


class MarkorDeleteAllNotes(Markor):
  """Task for deleting all notes in Markor."""

  # For this task's complexity, the agent may complete this task by deleting the
  # files one-by-one which envolves many steps (more than 10), but there is also
  # an optimal approach by first long pressing one file, then tapping to select
  # all others and deleting them all together.
  complexity = 1.4
  schema = {}
  template = "Delete all my notes in Markor."

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.generate_noise_files(
        user_data_generation.generate_random_string(5),
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
        random.randint(2, 6),
    )

    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )

    if not file_list:
      raise RuntimeError("Something went wrong, file was not created.")

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_list = file_utils.get_file_list_with_metadata(
        device_constants.MARKOR_DATA, env.controller
    )
    return 0.0 if file_list else 1.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {}


class MarkorCreateNote(Markor):
  """Task for checking that a new note in Markor has been created with a specific name and text."""

  app_names = ("markor",)
  complexity = 1.6
  schema = file_validators.CreateFile.schema
  template = (
      "Create a new note in Markor named {file_name} with the following text:"
      " {text}"
  )

  def __init__(self, params: dict[str, Any]):
    """See base class."""
    super().__init__(params)

    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)  # Delegate

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)  # Delegate

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    note = _generate_random_note()
    return {"file_name": note.name, "text": note.content}

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


class MarkorCreateNoteFromClipboard(Markor):
  """Task for creating a note using text in clipboard in Markor."""

  app_names = ("markor", "clipper")
  complexity = 1.4
  schema = {
      "type": "object",
      "properties": {
          "file_name": {"type": "string"},
          "file_content": {"type": "string"},
      },
      "required": ["file_name", "file_content"],
  }
  template = (
      "Create a note in Markor named {file_name}. Perform a paste operation in"
      " the note and save the note."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    if "file_content" not in params or not params["file_content"]:
      params["file_content"] = user_data_generation.generate_random_string(20)
    self.create_file_task = file_validators.CreateFile(
        {"file_name": params["file_name"], "text": params["file_content"]},
        device_constants.MARKOR_DATA,
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    adb_utils.set_clipboard_contents(
        self.params["file_content"], env.controller
    )
    if (
        adb_utils.get_clipboard_contents(env.controller)
        != self.params["file_content"]
    ):
      raise RuntimeError(
          "Something went wrong, clipboard not set up correctly."
      )
    self.create_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "file_name": _generate_random_note().name,
        "file_content": user_data_generation.generate_random_string(10),
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)


class MarkorMergeNotes(Markor):
  """Task for merging three existing notes into a new one."""

  complexity = 7.8
  schema = {
      "type": "object",
      "properties": {
          "file1_name": {"type": "string"},
          "file2_name": {"type": "string"},
          "file3_name": {"type": "string"},
          "new_file_name": {"type": "string"},
          "file1_content": {"type": "string"},
          "file2_content": {"type": "string"},
          "file3_content": {"type": "string"},
      },
      "required": [
          "file1_name",
          "file2_name",
          "file3_name",
          "new_file_name",
          "file1_content",
          "file2_content",
          "file3_content",
      ],
  }
  template = (
      "Merge the contents of Markor notes {file1_name}, {file2_name} and"
      " {file3_name} (in the same order) into a new Markor note named"
      " {new_file_name} and save it. Add a new line between the content of each"
      " note."
  )

  def __init__(self, params: dict[str, Any]):
    """Initialize the task."""
    super().__init__(params)
    self.create_file_task = file_validators.CreateFile(
        {
            "file_name": params["new_file_name"],
            # file_util.create_file with non-empty content will add a \n to the
            # end of the file.
            "text": (
                "\n\n".join([
                    self.params["file1_content"],
                    self.params["file2_content"],
                    self.params["file3_content"],
                ])
                + "\n"
            ),
        },
        device_constants.MARKOR_DATA,
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    file_utils.create_file(
        self.params["file1_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file1_content"],
    )
    file_utils.create_file(
        self.params["file2_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file2_content"],
    )
    file_utils.create_file(
        self.params["file3_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["file3_content"],
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["file1_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    file_utils.remove_single_file(
        self.params["file2_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    file_utils.remove_single_file(
        self.params["file3_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )
    self.create_file_task.tear_down(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not self.create_file_task.is_successful(env):
      return 0.0
    # The CreateFile task is using a fuzzy match in its is_successful function,
    # but here we want to explicitly check if the agent adds a blank line
    # between the notes. The following check only works based on the current way
    # we generate notes with the assumption that each file's content is a string
    # of length less than 20, consisting of letters and digits, ended with a \n.
    merged_file = (
        adb_utils.issue_generic_request(
            [
                "shell",
                "cat",
                file_utils.convert_to_posix_path(
                    device_constants.MARKOR_DATA, self.params["new_file_name"]
                ),
            ],
            env.controller,
        )
        .generic.output.decode()
        .replace("\r", "")
        .strip()
    )

    # merged_file should look like,
    # file1\n\nfile2\n\nfile3, where the first and third \n are inserted by
    # create_file in file_utils, the second and the forth \n should be inserted
    # by agent.
    content_split = merged_file.split("\n")
    are_notes_merged = (
        len(content_split) == 5
        and (not content_split[1])
        and (not content_split[3])
    )
    return 1.0 if are_notes_merged else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "file1_name": _generate_random_note().name,
        "file2_name": _generate_random_note().name,
        "file3_name": _generate_random_note().name,
        "new_file_name": user_data_generation.generate_random_string(8),
        "file1_content": user_data_generation.generate_random_string(20),
        "file2_content": user_data_generation.generate_random_string(20),
        "file3_content": user_data_generation.generate_random_string(20),
    }


class MarkorChangeNoteContent(Markor):
  """Task for changing an existing note's content and renaming it."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "original_name": {"type": "string"},
          "new_name": {"type": "string"},
          "updated_content": {"type": "string"},
      },
      "required": ["original_name", "new_name", "updated_content"],
  }
  template = (
      'Update the content of {original_name} to "{updated_content}" in Markor'
      " and change its name to {new_name}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=user_data_generation.generate_random_string(20),
    )
    user_data_generation.generate_noise_files(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )
    if not file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      raise RuntimeError("Something went wrong, file not created correctly.")

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    if not file_utils.check_file_or_folder_exists(
        self.params["new_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    content_updated = file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["new_name"]
        ),
        self.params["updated_content"],
        env.controller,
    )
    return 1.0 if content_updated else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    original = _generate_random_note().name
    new = _generate_random_note().name
    return {
        "original_name": original,
        "new_name": new,
        "updated_content": user_data_generation.generate_random_string(20),
    }


class MarkorAddNoteHeader(Markor):
  """Task for adding a header to an existing note and renaming it."""

  complexity = 1.2
  schema = {
      "type": "object",
      "properties": {
          "original_name": {"type": "string"},
          "new_name": {"type": "string"},
          "header": {"type": "string"},
          "original_content": {"type": "string"},
      },
      "required": ["original_name", "new_name", "header", "original_content"],
  }
  template = (
      "Update the Markor note {original_name} by adding the following text,"
      ' along with a new blank line before the existing content: "{header}",'
      " and rename it to {new_name}."
  )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    file_utils.create_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        content=self.params["original_content"],
    )
    user_data_generation.generate_noise_files(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
        _NOTE_TITLES,
    )

    if not file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      raise RuntimeError("Something went wrong, file not created correctly.")

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    file_utils.remove_single_file(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if file_utils.check_file_or_folder_exists(
        self.params["original_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    if not file_utils.check_file_or_folder_exists(
        self.params["new_name"],
        device_constants.MARKOR_DATA,
        env.controller,
    ):
      return 0.0
    correct = file_utils.check_file_content(
        file_utils.convert_to_posix_path(
            device_constants.MARKOR_DATA, self.params["new_name"]
        ),
        self.params["header"] + "\n\n" + self.params["original_content"] + "\n",
        env.controller,
        exact_match=True,
    )
    return 1.0 if correct else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    return {
        "original_name": _generate_random_note().name,
        "original_content": generate_random_sentence(),
        "new_name": _generate_random_note().name,
        "header": user_data_generation.generate_random_string(20),
    }


class MarkorTranscribeReceipt(task_eval.TaskEval):
  """Task for creating a markdown file from a receipt image using Simple Gallery and Markor.

  This task involves viewing a receipt image in Simple Gallery and then
  creating a markdown file in Markor with details of the transactions
  listed in the image. The file should be named 'receipt.md' and include
  transactions with the format "Date, Item, Amount".
  """

  app_names = ("simple gallery pro", "markor")
  complexity = 1.8
  template = (
      "Create a file in Markor, called receipt.md with the transactions from"
      " the receipt.png. Use Simple Gallery to view the receipt. Please enter"
      ' transactions in csv format including the header "Date, Item, Amount".'
  )

  schema = file_validators.CreateFile.schema

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.img = params.pop("img")
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initializes the task for creating a receipt markdown file."""
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    receipt_img_path = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), "receipt.png"
    )
    self.img.save(receipt_img_path)
    file_utils.copy_data_to_device(
        receipt_img_path,
        device_constants.GALLERY_DATA,
        env.controller,
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    self.create_file_task.tear_down(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    img, text = receipt_generator.create_receipt(random.randint(1, 5))
    text = "\n".join(text.split("\n")[2:])  # Remove header.
    return {
        "img": img,
        "file_name": "receipt.md",
        "text": text,
    }


class MarkorTranscribeVideo(Markor):
  """Task for transcribing a video using Markor."""

  complexity = 2
  schema = file_validators.CreateFile.schema
  app_names = ("markor", "vlc")

  template = (
      "Transcribe the contents of video {video_name} by watching it in VLC"
      " player (located in Download) and writing the sequence of strings shown"
      " on each frame to the text file {file_name} in Markor as a comma"
      ' separated list. For example, if the first frame shows the text "edna"'
      ' and the second frame shows the text "pineapple", then the text file'
      ' should contain only the following text: "edna, pineapple".'
  )

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.create_file_task = file_validators.CreateFile(
        params, device_constants.MARKOR_DATA
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)
    user_data_generation.write_video_file_to_device(
        self.params["video_name"],
        device_constants.DOWNLOAD_DATA,
        env,
        messages=self.params["messages"],
        message_display_time=8,
    )
    for file in self.params["noise_files"]:
      user_data_generation.write_video_file_to_device(
          file,
          device_constants.DOWNLOAD_DATA,
          env,
          messages=[user_data_generation.generate_random_string(10)],
          fps=1,
          message_display_time=random.randint(20, 180),
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    return self.create_file_task.is_successful(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    messages = list(
        random.sample(
            user_data_generation.COMMON_GIVEN_NAMES, random.randint(2, 4)
        )
    )
    video_name = vlc.generate_file_name()
    text_file_name = f"{video_name.split('.')[0]}_transcription.txt"
    return {
        "file_name": text_file_name,
        "text": ",".join(messages),
        # Video specific.
        "messages": messages,
        "video_name": video_name,
        "noise_files": [
            vlc.generate_file_name() for _ in range(random.randint(5, 20))
        ],
    }

_NOTE_TITLES = [
    "grocery_list_weekly.md",
    "meeting_notes_project_team.md",
    "personal_goals_2024.md",
    "reading_list_2024.md",
    "research_paper_summary.md",
    "summer_vacation_plans.md",
    "budget_home_renovation.md",
    "april_workout_routine.md",
    "birthday_gift_ideas_mom.md",
    "recipe_homemade_pizza.md",
    "weekend_todo_list.md",
    "insurance_plan_comparison.md",
    "art_project_sketches.md",
    "python_learning_goals.md",
    "trip_reflections_recent.md",
    "startup_ideas_launch.md",
    "client_meetings_schedule.md",
    "favorite_book_quotes.md",
    "garden_layout_plan.md",
    "upcoming_presentation_outline.md",
]
