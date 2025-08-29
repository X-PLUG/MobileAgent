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

"""Constant key names for storing episode data during evaluation runs."""

# The current step number in a given episode.
STEP_NUMBER = 'step_number'


class EpisodeConstants:
  """Episode-level constants when recording agents performing automation tasks.

  Attributes:
    EPISODE_DATA: Data for a full episode.
    GOAL: The high-level (episode-level) goal for an episode.
    INSTANCE_ID: The index of the instance in the combinations.
    TASK_TEMPLATE: A task template, e.g., MessagesSendSMS
    IS_SUCCESSFUL: Binary variable whether automated system detects success.
    RUN_TIME: Time to run episode.
    AGENT_NAME: The name of the agent. This is logged on the screen and in the
      trajectory data.
    EPISODE_LENGTH: The length of the episode.
    FINISH_DTIME: The datetime the task finished.
    SEED: The random seed to initialize the current episode's task.
    AUX_DATA: Additional data which can be passed from the task to
      process_episodes.
  """

  EPISODE_DATA = 'episode_data'
  GOAL = 'goal'
  INSTANCE_ID = 'instance_id'
  # Scripted task constants.
  TASK_TEMPLATE = 'task_template'
  IS_SUCCESSFUL = 'is_successful'
  RUN_TIME = 'run_time'
  AGENT_NAME = 'agent_name'
  EPISODE_LENGTH = 'episode_length'
  SCREEN_CONFIG = 'screen_config'
  EXCEPTION_INFO = 'exception_info'
  FINISH_DTIME = 'finish_dtime'
  SEED = 'seed'
  AUX_DATA = 'aux_data'
