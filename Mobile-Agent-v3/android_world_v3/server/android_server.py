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

"""FastAPI server for managing and interacting with an Android environment.

This server exposes endpoints to control an Android emulator, execute tasks,
and manage task execution on AndroidWorld tasks.
"""

import contextlib
import typing
from typing import Any

from android_world import registry as aw_registry_module
from android_world import suite_utils
from android_world.env import env_launcher
from android_world.env import interface
from android_world.env import json_action
import fastapi
import pydantic
import uvicorn


class StateResponse(pydantic.BaseModel):
  """Pydantic model for state responses, including pixels and UI elements."""

  pixels: list[int]
  ui_elements: list[Any]


@contextlib.asynccontextmanager
async def lifespan(fast_api_app: fastapi.FastAPI):
  """Manages the lifecycle of the Android environment and task suite."""
  fast_api_app.state.app_android_env = env_launcher.load_and_setup_env(
      console_port=5554,
      emulator_setup=True,
      freeze_datetime=True,
      adb_path="/opt/android/platform-tools/adb",
  )
  task_registry = aw_registry_module.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  initial_suite = suite_utils.create_suite(
      task_registry=aw_registry,
      n_task_combinations=2,
      seed=42,  # Optional: for reproducibility
  )
  fast_api_app.state.suite = initial_suite
  fast_api_app.state.task_registry = task_registry
  yield
  # Shutdown
  if fast_api_app.state.app_android_env is not None:
    fast_api_app.state.app_android_env.close()


app = fastapi.FastAPI(lifespan=lifespan)
suite_router = fastapi.APIRouter(prefix="/suite", tags=["suite"])
task_router = fastapi.APIRouter(prefix="/task", tags=["task"])


def get_app_android_env(request: fastapi.Request) -> interface.AsyncEnv:
  """Dependency to get the application's Android environment instance."""
  return request.app.state.app_android_env


def get_app_suite(request: fastapi.Request) -> suite_utils.Suite:
  """Dependency to get the application's task suite instance."""
  return request.app.state.suite


AndroidEnv = typing.Annotated[
    interface.AsyncEnv, fastapi.Depends(get_app_android_env)
]
AndroidSuite = typing.Annotated[
    suite_utils.Suite, fastapi.Depends(get_app_suite)
]


@app.post("/reset")
async def reset(go_home: bool, app_android_env: AndroidEnv):
  """Resets the Android environment, optionally returning to the home screen."""
  app_android_env.reset(go_home=go_home)
  return {
      "status": "success",
      "message": f"Environment reset with go_home={go_home}.",
  }


@app.get("/screenshot")
async def get_screenshot(wait_to_stabilize: bool, app_android_env: AndroidEnv):
  """Captures and returns the current screenshot of the Android environment."""
  state = app_android_env.get_state(wait_to_stabilize=wait_to_stabilize)
  return {"pixels": state.pixels.tolist()}


@app.post("/execute_action")
async def execute_action(
    action_dict: dict[str, typing.Any], app_android_env: AndroidEnv
):
  """Executes a given JSON-formatted action in the Android environment."""
  action = json_action.JSONAction(**action_dict)
  app_android_env.execute_action(action)
  return {"status": "success", "message": f"Action {action} executed."}


@suite_router.get("/task_list")
async def suite_task_list(max_index: int, app_suite: AndroidSuite):
  """Returns a list of task keys from the current suite, up to max_index."""
  if max_index > len(app_suite) or max_index < 0:
    return {"task_list": list(app_suite.keys())}
  return {"task_list": list(app_suite.keys())[:max_index]}


@suite_router.get("/task_length")
async def suite_task_length(task_type: str, app_suite: AndroidSuite):
  """Returns the number of tasks for a given task type in the suite."""
  return {"length": len(app_suite[task_type])}


@suite_router.get("/reinitialize")
def reinitialize_suite(
    request: fastapi.Request,
    n_task_combinations: int = 2,  # Default from initial lifespan setup
    seed: int = 42,  # Default from initial lifespan setup
    task_family: str = "android_world",
):
  """Re-initializes the task suite with new parameters."""
  task_registry = request.app.state.task_registry
  try:
    current_aw_registry = task_registry.get_registry(task_family)
  except ValueError as exc:
    raise fastapi.HTTPException(
        status_code=400, detail=f"Invalid task family: {task_family}"
    ) from exc
  new_suite = suite_utils.create_suite(
      task_registry=current_aw_registry,
      n_task_combinations=n_task_combinations,
      seed=seed,
  )
  request.app.state.suite = new_suite
  return {
      "status": "success",
      "message": (
          "Task suite re-initialized with"
          f" n_task_combinations={n_task_combinations}, seed={seed}."
      ),
  }


@task_router.post("/initialize")
async def initialize_task(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Initializes a specific task in the Android environment."""
  app_suite[task_type][task_idx].initialize_task(app_android_env)
  return {
      "status": "success",
      "message": f"Task {task_type} {task_idx} initialized.",
  }


@task_router.post("/tear_down")
async def tear_down_task(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Tears down a specific task in the Android environment."""
  app_suite[task_type][task_idx].tear_down(app_android_env)
  return {
      "status": "success",
      "message": f"Task {task_type} {task_idx} torn down.",
  }


@task_router.get("/score")
async def get_task_score(
    task_type: str,
    task_idx: int,
    app_android_env: AndroidEnv,
    app_suite: AndroidSuite,
):
  """Gets the success status (score) of a specific task."""
  return {
      "score": app_suite[task_type][task_idx].is_successful(app_android_env)
  }


@task_router.get("/goal")
async def get_task_goal(task_type: str, task_idx: int, app_suite: AndroidSuite):
  """Gets the goal description of a specific task."""
  return {"goal": app_suite[task_type][task_idx].goal}


@task_router.get("/template")
async def get_task_template(
    task_type: str, task_idx: int, app_suite: AndroidSuite
):
  """Gets the template or configuration details of a specific task."""
  return {"template": app_suite[task_type][task_idx].template}


@app.post("/close")
async def close(app_android_env: AndroidEnv):
  """Closes the Android environment."""
  app_android_env.close()
  return {"status": "success"}


@app.get("/health")
async def health(app_android_env: AndroidEnv):
  """Checks the health of the Android environment server."""
  if isinstance(app_android_env, interface.AsyncEnv):
    return {"status": "success"}
  raise fastapi.HTTPException(
      status_code=500, detail="Environment not initialized"
  )


app.include_router(suite_router)
app.include_router(task_router)

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5000)
