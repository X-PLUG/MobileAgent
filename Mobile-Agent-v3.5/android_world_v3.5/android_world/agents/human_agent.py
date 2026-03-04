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

"""Agent for human playing."""

import sys

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


class HumanAgent(base_agent.EnvironmentInteractingAgent):
  """Human agent; wait for user to indicate they are done."""

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    del goal
    response = input(
        'Human playing! Hit enter when you are ready for evaluation (or q to'
        ' quit).'
    )
    if response == 'q':
      sys.exit()
    action_details = {'action_type': 'answer', 'text': response}
    self.env.execute_action(json_action.JSONAction(**action_details))

    state = self.get_post_transition_state()
    result = {}
    result['elements'] = state.ui_elements
    result['pixels'] = state.pixels
    return base_agent.AgentInteractionResult(True, result)

  def get_post_transition_state(self) -> interface.State:
    return self.env.get_state()
