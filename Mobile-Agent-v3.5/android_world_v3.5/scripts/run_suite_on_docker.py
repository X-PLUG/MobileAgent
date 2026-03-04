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

"""Example client for interacting with an Android environment server over HTTP.

Prerequisites:

# Build the Docker image for the Android environment server from the root
repository directory
# docker build -t android_world:latest .

# Run the Docker container
# docker run --privileged -p 5000:5000 -it android_world:latest

After running the server, you can use the client to interact with the
environment. You'll need to implement your agent logic to interact with the
environment.
"""

import json
import logging
import time
from typing import Any

from android_world.env import json_action
import numpy as np
import pydantic
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)

Params = dict[str, int | str]


class Response(pydantic.BaseModel):
  status: str
  message: str


class AndroidEnvClient:
  """Client for interacting with the Android environment server."""

  def __init__(self):
    logger.info(
        "Setting up Android environment using Docker - Initial setup may take"
        " 5-10 minutes. Please wait..."
    )
    self.base_url = "http://localhost:5000"

  def reset(self, go_home: bool) -> Response:
    """Resets the environment."""
    response = requests.post(
        f"{self.base_url}/reset", params={"go_home": go_home}
    )
    response.raise_for_status()
    return Response(**response.json())

  def get_screenshot(
      self, wait_to_stabilize: bool = False
  ) -> np.ndarray[Any, Any]:
    """Gets the current screenshot of the environment."""
    response = requests.get(
        f"{self.base_url}/screenshot",
        params={"wait_to_stabilize": wait_to_stabilize},
    )
    response.raise_for_status()
    image = response.json()
    return np.array(image["pixels"])

  def execute_action(
      self,
      action: json_action.JSONAction,
  ) -> Response:
    """Executes an action in the environment."""
    print(f"Executing action: {action.json_str()}")
    response = requests.post(
        f"{self.base_url}/execute_action", json=json.loads(action.json_str())
    )
    response.raise_for_status()
    return Response(**response.json())

  def get_suite_task_list(self, max_index: int) -> list[str]:
    """Gets the list of tasks in the suite."""
    response = requests.get(
        f"{self.base_url}/suite/task_list", params={"max_index": max_index}
    )
    response.raise_for_status()
    return response.json()["task_list"]

  def get_suite_task_length(self, task_type: str) -> int:
    """Gets the length of the suite of tasks."""
    response = requests.get(
        f"{self.base_url}/suite/task_length", params={"task_type": task_type}
    )
    response.raise_for_status()
    return response.json()["length"]

  def reinitialize_suite(
      self,
      n_task_combinations: int = 2,  # Default from initial server setup.
      seed: int = 42,  # Default from initial server setup.
      task_family: str = "android_world",  # Default from initial server setup.
  ) -> Response:
    """Reinitializes the suite of tasks."""
    response = requests.get(
        f"{self.base_url}/suite/reinitialize",
        params={
            "n_task_combinations": n_task_combinations,
            "seed": seed,
            "task_family": task_family,
        },
    )
    response.raise_for_status()
    return Response(**response.json())

  def initialize_task(self, task_type: str, task_idx: int) -> Response:
    """Initializes the task in the environment."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = requests.post(f"{self.base_url}/task/initialize", params=params)
    response.raise_for_status()
    return Response(**response.json())

  def tear_down_task(self, task_type: str, task_idx: int) -> Response:
    """Tears down the task in the environment."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = requests.post(f"{self.base_url}/task/tear_down", params=params)
    response.raise_for_status()
    return Response(**response.json())

  def get_task_score(self, task_type: str, task_idx: int) -> float:
    """Gets the score of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = requests.get(f"{self.base_url}/task/score", params=params)
    response.raise_for_status()
    return response.json()["score"]

  def get_task_goal(self, task_type: str, task_idx: int) -> str:
    """Gets the goal of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = requests.get(f"{self.base_url}/task/goal", params=params)
    response.raise_for_status()
    return response.json()["goal"]

  def get_task_template(self, task_type: str, task_idx: int) -> str:
    """Gets the template of the current task."""
    params: Params = {"task_type": task_type, "task_idx": task_idx}
    response = requests.get(f"{self.base_url}/task/template", params=params)
    response.raise_for_status()
    return response.json()["template"]

  def close(self) -> None:
    """Closes the environment."""
    response = requests.post(f"{self.base_url}/close")
    response.raise_for_status()

  def health(self) -> bool:
    """Checks the health of the environment."""
    try:
      response = requests.get(f"{self.base_url}/health")
      response.raise_for_status()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Environment is not healthy: {e}")
      return False
    return True


if __name__ == "__main__":
  client = AndroidEnvClient()

  while True:
    if not client.health():
      print("Environment is not healthy, waiting for 1 second...")
      time.sleep(1)
    else:
      break

  res = client.reset(go_home=True)
  print(f"reset response: {res}")

  screenshot = client.get_screenshot()
  print("Screen dimensions:", screenshot.shape)

  res = client.execute_action(
      json_action.JSONAction(action_type="click", x=100, y=200)
  )
  print(f"execute_action response: {res}")

  task_list = client.get_suite_task_list(max_index=-1)
  print(task_list)

  res = client.reinitialize_suite()
  print(f"reinitialize_suite response: {res}")

  for task_name in task_list:
    num_tasks = client.get_suite_task_length(task_type=task_name)
    print(f"num_tasks: {num_tasks}")

    for cur_idx in range(num_tasks):
      task_template = client.get_task_template(
          task_type=task_name, task_idx=cur_idx
      )
      print(f"task_template: {task_template}")

      task_goal = client.get_task_goal(task_type=task_name, task_idx=cur_idx)
      print(f"task_goal: {task_goal}")

      try:
        res = client.initialize_task(task_type=task_name, task_idx=cur_idx)
        print(f"initialize_task response: {res}")

        # Complete the task using your agent...

        task_score = client.get_task_score(
            task_type=task_name, task_idx=cur_idx
        )
        print(f"task_score: {task_score}")

        res = client.tear_down_task(task_type=task_name, task_idx=cur_idx)
        print(f"tear_down_task response: {res}")

      except Exception as e:  # pylint: disable=broad-exception-caught
        # Error tasks:
        # RetroPlayingQueue -> sqlite3.OperationalError: no such table:
        # playing_queue.
        # SimpleSmsReplyMostRecent -> IndexError: list index out of range
        print(f"Error initializing task {task_name} {cur_idx}: {e}")
        print("Continuing to next task...")
        continue

      res = client.reset(go_home=True)
      print(f"reset response: {res}")

  client.close()
