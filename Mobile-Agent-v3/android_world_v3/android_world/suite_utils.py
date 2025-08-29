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

"""Utilities for evaluating automation agents."""

import collections
import datetime
import hashlib
import logging
import os
import random
import time
import traceback
from typing import Any, Callable, Type, TypeVar

from android_env import env_interface
from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import episode_runner
from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.miniwob import miniwob_base
from fuzzywuzzy import process
import numpy as np
import pandas as pd

# A fixed seed to use when use identical parameters but seed is not set.
_FIXED_SEED = 123
_TASK_TEMPLATE_COLUMN = 'task_template'
_TASK_PROMPT_COLUMN = 'task_prompt'
TaskEvalType = TypeVar('TaskEvalType', bound=task_eval.TaskEval)


class Suite(dict[str, list[task_eval.TaskEval]]):
  """A suite of tasks.

  Each key is the task name as defined in registry.py and its value is a list
  of instantiated task objects. These instances differ from each other by their
  parameter initializations; i.e. each task will have different task parameters.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._suite_family = None

  @property
  def suite_family(self) -> str:
    """Getter for suite_family."""
    if self._suite_family is None:
      raise ValueError('Suite family is not set; please first set it.')
    return self._suite_family

  @suite_family.setter
  def suite_family(self, value: str):
    """Setter for suite_family."""
    self._suite_family = value


def _log_and_print(msg: str, *args: object) -> None:
  formatted = msg % args if args else msg
  logging.info(formatted)
  print(formatted)


def _instantiate_task(
    task: Type[task_eval.TaskEval],
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    env: interface.AsyncEnv | None = None,
) -> task_eval.TaskEval:
  """Creates an instance of a task with params.

  If params is not provided, it will use random params, controlled by a seed.

  Args:
    task: The task to instantiate.
    params: Params to use.
    seed: Seed for the random number generator.
    env: The environment.

  Returns:
    An instance of a task.
  """
  task.set_device_time(env)
  if params is None:
    if seed is not None:
      random.seed(seed)
    params = task.generate_random_params()
    params[constants.EpisodeConstants.SEED] = seed
  return task(params)


def create_suite(
    task_registry: dict[str, Type[task_eval.TaskEval]],
    n_task_combinations: int = 1,
    seed: int | None = None,
    tasks: list[str] | None = None,
    use_identical_params: bool = False,
    env: interface.AsyncEnv | None = None
) -> Suite:
  """Creates task suite.

  A task suite is a set of tasks. Each task is instantiated
  `n_task_combinations` times using new parameters. For example a task suite
  could look like:

  ```python
  {
      'GoogleSearchTask': [
          GoogleSearchTask({'term': 'cute cats'}),
          GoogleSearchTask({'term': 'comfy pillows'}),
      ],
      'WifiDisable': [  # No params for WiFi task.
          WifiDisable({}),
          WifiDisable({}),
      ],
  }
  ```

  Args:
    task_registry: Maps task names to their TaskEvals.
    n_task_combinations: Number of instances to create per task. Each instance
      will have unique param combinations.
    seed: Seed for the random number generator. Setting the seed will result in
      the same sequence of params for task instantiation per each task.
    tasks: List of task types that should be in the suite. If value is `None`
      all task types and associated instances will be created.
    use_identical_params: If True, each instance of a task, for a total of
      `n_task_combinations`, will have the same params.
    env: The environment that will be run on.

  Returns:
    A mapping of task name to instances of the task.
  """

  def _get_instance_seed(name: str, i: int) -> int:
    unique_seed_str = f'{seed}_{name}_{i}'
    return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (
        2**32
    )

  suite = {}
  for name, task_type in task_registry.items():
    current = []
    for i in range(n_task_combinations):
      if use_identical_params:
        instance_seed = (
            _get_instance_seed(name, 0) if seed is not None else _FIXED_SEED
        )
      elif seed is not None:
        instance_seed = _get_instance_seed(name, i)
      else:
        instance_seed = None
      current.append(_instantiate_task(task_type, seed=instance_seed, env=env))
    suite[name] = current
  suite = _filter_tasks(suite, task_registry, tasks)

  # Sort suite alphabetically by task name.
  return Suite(sorted(suite.items()))


def _suggest_keyword(
    typo: str, keywords: list[str], threshold: int = 80
) -> str:
  """Suggests a keyword."""
  suggestion, score = process.extractOne(typo, keywords)
  if score >= threshold:
    return f" Did you mean '{suggestion}'?"
  else:
    return ''


def _filter_tasks(
    suite: dict[str, list[task_eval.TaskEval]],
    task_registry: dict[str, Type[task_eval.TaskEval]],
    tasks: list[str] | None = None,
) -> dict[str, list[task_eval.TaskEval]]:
  """Filters a suite by specific tasks.

  Args:
    suite: The suite to retrieve tasks from.
    task_registry: The task registry the suite is from.
    tasks: The tasks to retrieve. If None, just return entire suite.

  Returns:
    A "mini-suite" of tasks from suite.

  Raises:
    ValueError: If invalid task name.
  """
  if tasks is None:
    return suite
  subset = {}

  # Validate.
  for name in tasks:
    if name not in task_registry:
      raise ValueError(
          f'Task {name} not found in the task registry.'
          + _suggest_keyword(name, list(task_registry.keys()))
      )

  # Filter.
  for name, instances in suite.items():
    if name in tasks:
      subset[name] = instances
  return subset


def _run_task(
    task: TaskEvalType,
    run_episode: Callable[[TaskEvalType], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    demo_mode: bool,
) -> dict[str, Any]:
  """Runs a task.

  Args:
    task: The task.
    run_episode: Runs the agent on the task.
    env: Environment that will be run on.
    demo_mode: Whether running in demo mode; will display success overlay if so.

  Returns:
    Episode data and associated success signals.

  Raises:
    ValueError: If step data was not as expected.
  """
  start = time.time()
  try:
    task.initialize_task(env)
    _log_and_print('Running task %s with goal "%s"', task.name, task.goal)
    interaction_results = run_episode(task)
    task_successful = task.is_successful(env)
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log_and_print('%s\nSKIPPING %s.', '~' * 80, task.name)
    logging.exception(
        'Logging exception and skipping task. Will keep running. Task: %s: %s',
        task.name,
        e,
    )
    traceback.print_exc()
    return _create_failed_result(
        task.name, task.goal, traceback.format_exc(), time.time() - start
    )
  else:
    agent_successful = task_successful if interaction_results.done else 0.0
    _log_and_print(
        '%s; %s',
        'Task Successful ✅' if agent_successful > 0.5 else 'Task Failed ❌',
        f' {task.goal}',
    )

    if demo_mode:
      _display_success_overlay(env.controller, agent_successful)

    result = {
        constants.EpisodeConstants.GOAL: task.goal,
        constants.EpisodeConstants.TASK_TEMPLATE: task.name,
        constants.EpisodeConstants.EPISODE_DATA: interaction_results.step_data,
        constants.EpisodeConstants.IS_SUCCESSFUL: agent_successful,
        constants.EpisodeConstants.RUN_TIME: time.time() - start,
        constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
        constants.EpisodeConstants.EPISODE_LENGTH: len(
            interaction_results.step_data[constants.STEP_NUMBER]
        ),
        constants.EpisodeConstants.AUX_DATA: interaction_results.aux_data,
        constants.EpisodeConstants.SCREEN_CONFIG: _get_screen_config(task),
        constants.EpisodeConstants.EXCEPTION_INFO: None,
        constants.EpisodeConstants.SEED: task.params[
            constants.EpisodeConstants.SEED
        ],
    }
    task.tear_down(env)
    return result


def _get_task_info(
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
  """Gets task info from episodes.

  Args:
    episodes: Episodes to get info from.

  Returns:
    A tuple of completed and failed task lookup tables.
  """

  completed = collections.defaultdict(list)
  failed = collections.defaultdict(list)
  for episode in episodes:
    instance_name = (
        episode[constants.EpisodeConstants.TASK_TEMPLATE]
        + checkpointer_lib.INSTANCE_SEPARATOR
        + str(episode[constants.EpisodeConstants.INSTANCE_ID])
    )
    if episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is not None:
      failed[instance_name].append(episode)
    else:
      completed[instance_name].append(episode)
  return completed, failed


def _run_task_suite(
    suite: Suite,
    run_episode: Callable[[task_eval.TaskEval], episode_runner.EpisodeResult],
    env: interface.AsyncEnv,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    agent_name: str = '',
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
  """Runs e2e system on suite.

  Args:
    suite: The suite to run it on.
    run_episode: The e2e system. See run_suite.py for an example.
    env: The environment e2e system runs on.
    checkpointer: See docstring from `run`.
    demo_mode: Whether to display the scoreboard.
    agent_name: The name of the agent.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Metadata for each episode, including the scripted reward.
  """
  metadata_fields = [
      constants.EpisodeConstants.GOAL,
      constants.EpisodeConstants.TASK_TEMPLATE,
      constants.EpisodeConstants.INSTANCE_ID,
      constants.EpisodeConstants.IS_SUCCESSFUL,
      constants.EpisodeConstants.EPISODE_LENGTH,
      constants.EpisodeConstants.RUN_TIME,
      constants.EpisodeConstants.EXCEPTION_INFO,
      constants.EpisodeConstants.AUX_DATA,
  ]
  completed_tasks, failed_tasks = _get_task_info(
      checkpointer.load(fields=metadata_fields)
  )
  if process_episodes_fn is None:
    process_episodes_fn = process_episodes

  if (completed_tasks or failed_tasks) and return_full_episode_data:
    raise ValueError(
        'Cannot return full episode data when resuming from a checkpoint.'
    )
  episodes_metadata: list[dict[str, Any]] = []
  full_episode_data = []
  correct, total = 0, 0
  for name, instances in suite.items():
    msg = 'Running task: ' + name
    _log_and_print(msg + '\n' + '=' * len(msg))

    for i, instance in enumerate(instances):
      instance_name = (
          instance.name + checkpointer_lib.INSTANCE_SEPARATOR + str(i)
      )
      # Transferring from old checkpoint.
      if instance_name in completed_tasks:
        completed_episodes: list[dict[str, Any]] = completed_tasks[
            instance_name
        ]
        episodes_metadata.extend(completed_episodes)
      if instance_name in failed_tasks:
        episodes_metadata.extend(failed_tasks[instance_name])
      already_processed = (
          instance_name in completed_tasks and instance_name not in failed_tasks
      )
      if already_processed:
        _log_and_print('Skipping already processed task %s', instance_name)
        continue

      episode = _run_task(instance, run_episode, env, demo_mode=demo_mode)
      if (
          episode.get(constants.EpisodeConstants.EXCEPTION_INFO) is None
          and check_episode_fn is not None
      ):
        if not check_episode_fn(episode):
          continue
      episode[constants.EpisodeConstants.AGENT_NAME] = agent_name
      episode[constants.EpisodeConstants.INSTANCE_ID] = i
      checkpointer.save_episodes([episode], instance_name)

      if return_full_episode_data:
        full_episode_data.append(episode)

      episodes_metadata.append({k: episode[k] for k in metadata_fields})
      process_episodes_fn(episodes_metadata, print_summary=True)

      if episode[constants.EpisodeConstants.EXCEPTION_INFO] is not None:
        # Don't include episode in tally if execution/eval logic errored out.
        continue
      correct += episode[constants.EpisodeConstants.IS_SUCCESSFUL]
      total += 1
      if demo_mode:
        _update_scoreboard(correct, total, env.controller)
    print()

  return full_episode_data if return_full_episode_data else episodes_metadata


def run(
    suite: Suite,
    agent: base_agent.EnvironmentInteractingAgent,
    checkpointer: checkpointer_lib.Checkpointer = checkpointer_lib.NullCheckpointer(),
    demo_mode: bool = False,
    return_full_episode_data: bool = False,
    process_episodes_fn=None,
    check_episode_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
  """Create suite and runs eval suite.

  Args:
    suite: The suite of tasks to run on.
    agent: An agent that interacts on the environment.
    checkpointer: Checkpointer that loads from existing run and resumes from
      there. NOTE: It will resume from the last fully completed task template.
      Relatedly, data for a task template will not be saved until all instances
      are executed.
    demo_mode: Whether to run in demo mode, which displays a scoreboard and the
      task instruction as a notification.
    return_full_episode_data: Whether to return full episode data instead of
      just metadata.
    process_episodes_fn: The function to process episode data. Usually to
      compute metrics. Deafaults to process_episodes from this file.
    check_episode_fn: The function to check episode data.

  Returns:
    Step-by-step data from each episode.
  """

  def run_episode(task: task_eval.TaskEval) -> episode_runner.EpisodeResult:
    if demo_mode:
      _display_goal(agent.env, task)
    return episode_runner.run_episode(
        goal=task.goal,
        agent=agent,
        max_n_steps=_allocate_step_budget(task.complexity),
        start_on_home_screen=task.start_on_home_screen,
        termination_fn=(
            miniwob_base.is_episode_terminated
            if task.name.lower().startswith('miniwob')
            else None
        ),
    )

  if demo_mode:
    adb_utils.send_android_intent(
        'broadcast',
        'com.example.ACTION_UPDATE_SCOREBOARD',
        agent.env.controller,
        extras={'player_name': agent.name, 'scoreboard_value': '00/00'},
    )

  results = _run_task_suite(
      suite,
      run_episode,
      agent.env,
      checkpointer=checkpointer,
      demo_mode=demo_mode,
      agent_name=agent.name,
      return_full_episode_data=return_full_episode_data,
      process_episodes_fn=process_episodes_fn,
      check_episode_fn=check_episode_fn,
  )

  return results


def _allocate_step_budget(task_complexity: float) -> int:
  """Allocates number of steps dynamically based on the complexity score.

  Args:
    task_complexity: Complexity score of the task.

  Returns:
    Allocated number of steps for the task.
  """
  if task_complexity is None:
    raise ValueError('Task complexity must be provided.')
  return int(10 * (task_complexity))


def _display_message(
    header: str, body: str, env: env_interface.AndroidEnvInterface
) -> None:
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'task_type_string': header, 'goal_string': body},
  )


def _display_goal(env: interface.AsyncEnv, task: task_eval.TaskEval) -> None:
  """Displays the goal on the screen using Android World.

  Args:
    env: The environment.
    task: The current task.
  """
  adb_utils.launch_app('android world', env.controller)
  time.sleep(1.0)
  _display_message(task.goal, task.name, env.controller)
  time.sleep(6.0)
  adb_utils.press_home_button(env.controller)
  time.sleep(1.0)


def _get_screen_config(task: task_eval.TaskEval) -> dict[str, Any]:
  return {
      'width': task.width if hasattr(task, 'width') else 1080,
      'height': task.height if hasattr(task, 'height') else 2400,
      'orientation': (
          task.orientation if hasattr(task, 'orientation') else 'portrait'
      ),
      'config_name': (
          task.config_name if hasattr(task, 'config_name') else 'default'
      ),
  }


def _create_failed_result(
    name: str, goal: str, exception: str, run_time: float
) -> dict[str, Any]:
  """Creates empty result to use if the run fails for some reason."""
  return {
      constants.EpisodeConstants.GOAL: goal,
      constants.EpisodeConstants.TASK_TEMPLATE: name,
      constants.EpisodeConstants.EPISODE_DATA: np.nan,
      constants.EpisodeConstants.IS_SUCCESSFUL: np.nan,
      constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
      constants.EpisodeConstants.RUN_TIME: run_time,
      constants.EpisodeConstants.EPISODE_LENGTH: np.nan,
      constants.EpisodeConstants.EXCEPTION_INFO: exception,
      constants.EpisodeConstants.AUX_DATA: None,
  }


def _display_success_overlay(
    env: env_interface.AndroidEnvInterface, success: float
) -> None:
  """Displays success overlay."""
  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_OVERLAY',
      env,
      extras={'success_string': str(int(success))},
  )
  time.sleep(1.0)  # Let display linger.


def _update_scoreboard(
    n_correct: int, n: int, env: env_interface.AndroidEnvInterface
) -> None:
  """Updates the scoreboard."""
  percentage = (n_correct / n) * 100
  scoreboard_value = f'{n_correct}/{n} ({percentage:.1f}%)'

  adb_utils.send_android_intent(
      'broadcast',
      'com.example.ACTION_UPDATE_SCOREBOARD',
      env,
      extras={'scoreboard_value': scoreboard_value},
  )


def _extract_task_metadata() -> pd.DataFrame:
  """Extracts metadata from task_metadata.json."""
  name = 'task_metadata.json'
  filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
  df = pd.read_json(filepath)
  df.rename(columns={_TASK_TEMPLATE_COLUMN: _TASK_PROMPT_COLUMN}, inplace=True)
  df.rename(columns={'task_name': _TASK_TEMPLATE_COLUMN}, inplace=True)
  return df.set_index(_TASK_TEMPLATE_COLUMN)[
      ['difficulty', 'optimal_steps', 'tags']
  ]


def _print_results_by_tag(result_df: pd.DataFrame) -> None:
  exploded_df = result_df.explode('tags').reset_index()
  exploded_df.replace(regex={'tags': r''}, value='untagged', inplace=True)  # pytype: disable=wrong-arg-types
  return (
      exploded_df.groupby(['tags', 'difficulty'], as_index=False)
      .agg(
          num_tasks=(_TASK_TEMPLATE_COLUMN, 'count'),
          mean_success_rate=('mean_success_rate', 'mean'),
      )
      .pivot_table(
          index=['tags'],
          columns='difficulty',
          values=[
              'mean_success_rate',
          ],
      )
      .fillna('-')
      .reindex(columns=['easy', 'medium', 'hard'], level='difficulty')
  )


def process_episodes(
    episodes: list[dict[str, Any]], print_summary: bool = False
) -> pd.DataFrame:
  """Processes task suite results; i.e. the output from `run_task_suite`.

  results = run_task_suite(...)
  # Contents of results.
  results = [
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Pause the stopwatch.',
        'task_template': 'ClockStopWatchPaused',
        'episode_data': ...,
        'is_successful': False
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    },
    {
        'goal': 'Run the stopwatch.',
        'task_template': 'ClockStopWatchRunnin',
        'episode_data': ...,
        'is_successful': True
    }
  ]

  process_episodes(results)
  # Output:
  # | task_template               |   n_trials |   average_success_rate |
  # |:----------------------------|-----------:|-----------------------:|
  # | ClockStopWatchPausedVerify  |          2 |                   0.5  |
  # | ClockStopWatchRunning       |          2 |                   1    |
  # | ==========Average========== |          2 |                   0.75 |

  Args:
    episodes: Results from running `run_task_suite`.
    print_summary: Whether to print the dataframe with a summary row.

  Returns:
    A dataframe aggregating results of run.
  """

  df = pd.DataFrame(list(episodes))

  # Add exeception info for backwards compatibility.
  df = df.assign(**{
      constants.EpisodeConstants.EXCEPTION_INFO: df.get(
          constants.EpisodeConstants.EXCEPTION_INFO, np.nan
      )
  })

  result_df = df.groupby(
      constants.EpisodeConstants.TASK_TEMPLATE, dropna=True
  ).agg({
      constants.EpisodeConstants.IS_SUCCESSFUL: ['count', 'mean'],
      constants.EpisodeConstants.EPISODE_LENGTH: 'mean',
      constants.EpisodeConstants.RUN_TIME: 'sum',
      constants.EpisodeConstants.EXCEPTION_INFO: [
          ('none_count', lambda x: x.notnull().sum())
      ],
  })
  result_df = result_df.sort_index()
  result_df.columns = [
      'num_complete_trials',
      'mean_success_rate',
      'mean_episode_length',
      'total_runtime_s',
      'num_fail_trials',
  ]
  result_df['total_runtime_s'] = result_df['total_runtime_s'].map(
      lambda x: float('{:.1f}'.format(x))
  )

  # Extract metadata and merge with the results table.
  metadata_df = _extract_task_metadata()
  tagged_result_df = result_df.merge(
      metadata_df, on=[_TASK_TEMPLATE_COLUMN], how='left'
  )

  if print_summary:
    avg = result_df.mean(axis=0)
    avg.name = '========= Average ========='

    result = pd.concat([result_df, avg.to_frame().T])
    result.index.name = 'task'
    result.insert(0, 'task_num', list(range(len(result) - 1)) + [0])
    result.task_num = result.task_num.astype(int)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.width', 1000)
    _log_and_print('\n\n%s', result)  # Use lazy % formatting

    # Add a chart that shows mean success rate by tag and difficulty.
    tags_df = _print_results_by_tag(tagged_result_df)
    pd.set_option('display.precision', 2)
    _log_and_print('\n\n%s', tags_df)

  return tagged_result_df
