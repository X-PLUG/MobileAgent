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

"""Tasks for Simple Draw Pro app."""

import random
from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import file_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class SimpleDrawProCreateDrawing(task_eval.TaskEval):
  """Task for checking that a new drawing has been created with a specific name."""

  app_names = ("simple draw pro",)
  complexity = 1.8
  schema = file_validators.CreateFile.schema
  template = (
      "Create a new drawing in Simple Draw Pro. Name it {file_name}. Save it in"
      " the Pictures folder within the sdk_gphone_x86_64 storage area."
  )

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.initialized = False
    self.create_file_task = file_validators.CreateFile(
        params,
        file_utils.convert_to_posix_path(
            device_constants.EMULATOR_DATA, "Pictures"
        ),
    )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    self.create_file_task.initialize_task(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    file_name = self.params["file_name"]
    exists = file_utils.check_file_or_folder_exists(
        file_name, self.create_file_task.data_directory, env.controller
    )
    return 1.0 if exists else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str | int]:
    words = [
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
    ]
    extensions = [".png", ".svg", ".jpg"]
    random_file_name = (
        "".join(random.choices(words, k=1))
        + "_"
        + user_data_generation.generate_random_file_name()
        + random.choice(extensions)
    )

    return {
        "file_name": random_file_name,
        "text": "",  # Unused.
    }

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    self.create_file_task.tear_down(env)
