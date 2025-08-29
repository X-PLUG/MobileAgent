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

"""Tests for suite utils."""

import copy
import time
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from android_world import checkpointer
from android_world import constants
from android_world import episode_runner
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
from android_world.utils import test_utils
import dm_env
import numpy as np


class TestCreateSuite(parameterized.TestCase):
  """Test that entire suite can be created.

  Later tests probe specific features related to the registry.
  """

  @parameterized.named_parameters(
      dict(testcase_name='android', family='information_retrieval'),
      dict(testcase_name='miniwob', family='miniwob'),
      dict(
          testcase_name='information_retrieval', family='information_retrieval'
      ),
  )
  def test_create_suite(self, family: str):
    suite_utils.create_suite(
        registry.TaskRegistry().get_registry(family), n_task_combinations=2
    )


class TestSuite(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.task_registry = registry.TaskRegistry()
    self.original_registry = copy.deepcopy(
        self.task_registry.ANDROID_TASK_REGISTRY
    )
    self.testing_registry = {
        'Task1': test_utils.FakeCurrentStateEval,
        'Task2': test_utils.FakeAdbEval,
    }
    self.seed = 42

  def test_create_entire_suite(self):
    suite_utils.create_suite(self.original_registry, n_task_combinations=2)

  def test_create_suite(self):
    n_task_combinations = 2
    suite = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=n_task_combinations
    )
    self.assertLen(
        suite['Task1'],
        n_task_combinations,
        'Should create 2 instances for Task1',
    )
    self.assertLen(
        suite['Task2'],
        n_task_combinations,
        'Should create 2 instances for Task2',
    )

  def test_determinism_with_same_seed(self):
    suite1 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )
    suite2 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )

    self.assertEqual(
        suite1['Task1'][0].params,
        suite2['Task1'][0].params,
        'Task1 instance 1 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task1'][1].params,
        suite2['Task1'][1].params,
        'Task1 instance 2 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task2'][0].params,
        suite2['Task2'][0].params,
        'Task2 instance 1 params should match with the same seed',
    )
    self.assertEqual(
        suite1['Task2'][1].params,
        suite2['Task2'][1].params,
        'Task2 instance 2 params should match with the same seed',
    )

  def test_variation_with_different_seed(self):
    suite1 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed
    )
    suite2 = suite_utils.create_suite(
        self.testing_registry, n_task_combinations=2, seed=self.seed + 1
    )

    self.assertNotEqual(
        suite1['Task1'][0].params,
        suite2['Task1'][0].params,
        'Task1 instance 1 params should not match with different seeds',
    )
    self.assertNotEqual(
        suite1['Task2'][0].params,
        suite2['Task2'][0].params,
        'Task2 instance 1 params should not match with different seeds',
    )

  @mock.patch.object(suite_utils.random, 'seed')
  def test_no_seed_provides_randomness(self, mock_seed):
    suite_utils.create_suite(self.testing_registry, n_task_combinations=2)
    mock_seed.assert_not_called()

  def test_return_all_when_tasks_none(self):
    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'test'
    tasks = None

    result = suite_utils._filter_tasks(
        suite,
        self.task_registry.get_registry(registry.TaskRegistry.ANDROID_FAMILY),
        tasks,
    )

    self.assertEqual(
        result, suite, 'Should return the same suite when tasks is None'
    )

  def test_valid_tasks_subset(self):
    expected = [
        test_utils.FakeCurrentStateEval(
            test_utils.FakeCurrentStateEval.generate_random_params()
        )
    ]
    tasks = ['Task1']

    result = suite_utils._filter_tasks(
        {
            'Task1': expected,
            'Task2': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                )
            ],
            'Task3': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
        self.testing_registry,
        tasks,
    )

    self.assertEqual(
        result, {'Task1': expected}, 'Should return the subset of tasks'
    )

  def test_invalid_task_raises_value_error(self):
    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                )
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'test'
    tasks = ['Task1', 'Task3']

    with self.assertRaises(ValueError):
      suite_utils._filter_tasks(
          suite,
          self.task_registry.get_registry(registry.TaskRegistry.ANDROID_FAMILY),
          tasks,
      )


class SuiteUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testing_registry = {
        'Task1': test_utils.FakeCurrentStateEval,
        'Task2': test_utils.FakeAdbEval,
    }

  @parameterized.named_parameters(
      dict(testcase_name='no_demo_mode', demo_mode=False),
      dict(testcase_name='demo_mode', demo_mode=True),
  )
  @mock.patch.object(test_utils.FakeAdbEval, 'initialize_task', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(time, 'sleep', autospec=True)
  def test_run_adb_task_instances(
      self,
      mock_sleep,
      mock_send_android_intent,
      mock_env,
      mock_initialize_task,
      demo_mode,
  ):
    pixels = np.zeros((3, 3, 3))
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': pixels},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_android_env = mock.PropertyMock(return_value=mock.MagicMock())
    mock_env.controller = mock_android_env
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        True,
        {'step_number': [0]},
    )

    result = suite_utils._run_task(
        test_utils.FakeAdbEval(test_utils.FakeAdbEval.generate_random_params()),
        mock_run_e2e,
        mock_env,
        demo_mode=demo_mode,
    )

    self.assertEqual(result['is_successful'], 1)
    self.assertIn(result['goal'], 'ADB eval')
    mock_initialize_task.assert_called_once()
    if demo_mode:
      mock_send_android_intent.assert_has_calls([
          mock.call(
              'broadcast',
              'com.example.ACTION_UPDATE_OVERLAY',
              mock_android_env,
              extras={'success_string': '1'},
          ),
      ])
      mock_sleep.assert_called()

  def test_run_miniwob_task_instances_initialize(self):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        done=True,
        step_data={'step_number': [0]},
    )
    failing_instance = test_utils.FakeMiniWobTask(
        test_utils.FakeMiniWobTask.generate_random_params()
    )

    result = suite_utils._run_task(
        failing_instance, mock_run_e2e, mock.MagicMock(), demo_mode=False
    )

    self.assertIsNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  def test_run_adb_task_instances_initialize_fails(self):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        done=True,
        step_data={'step_number': [0]},
    )
    failing_instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    failing_instance.initialize_task = lambda: ValueError(
        'Something went wrong'
    )

    result = suite_utils._run_task(
        failing_instance, mock_run_e2e, mock.MagicMock(), demo_mode=False
    )
    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_adb_task_instances_is_successful_fails(self, mock_env):
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.return_value = episode_runner.EpisodeResult(
        True,
        {
            'step_number': [0],
        },
    )
    failing_instance = test_utils.FakeAdbEval(
        test_utils.FakeAdbEval.generate_random_params()
    )
    failing_instance.is_successful = lambda: ValueError('Something went wrong')

    result = suite_utils._run_task(
        failing_instance,
        mock_run_e2e,
        mock_env,
        demo_mode=False,
    )

    self.assertIsNotNone(result[constants.EpisodeConstants.EXCEPTION_INFO])

  @mock.patch.object(suite_utils, '_run_task_suite')
  @mock.patch.object(base_agent, 'EnvironmentInteractingAgent', autospec=True)
  def test_run(
      self,
      mock_agent,
      mock_run_suite,
  ):
    mock_run_suite.return_value = [{
        'goal': 'Goal',
        'is_successful': 1.0,
        'agent_name': 'AnAgent',
    }] * 2
    mock_agent.name = 'AnAgent'
    mock_agent.env = test_utils.FakeAsyncEnv()
    n_task_combinations = 1
    tasks = ['Task1']
    suite = suite_utils.create_suite(
        self.testing_registry,
        n_task_combinations=n_task_combinations,
        tasks=tasks,
    )

    results = suite_utils.run(suite, agent=mock_agent, demo_mode=False)

    mock_run_suite.assert_called_once()
    self.assertLen(results, 2)
    for result in results:
      self.assertEqual(result[constants.EpisodeConstants.GOAL], 'Goal')
      self.assertEqual(result[constants.EpisodeConstants.IS_SUCCESSFUL], True)
      self.assertEqual(result[constants.EpisodeConstants.AGENT_NAME], 'AnAgent')


class RunTaskSuiteTest(absltest.TestCase):

  def assertTaskResults(self, results: list[dict[str, Any]]) -> None:
    """Asserts that the tasks have executed as expected.

    Args:
      results: A list of dictionaries containing the result of task execution.
    """
    self.assertEqual(results[0]['is_successful'], 0)
    self.assertIn(results[0]['goal'], 'Current state eval')
    self.assertEqual(results[1]['is_successful'], 1)
    self.assertIn(results[1]['goal'], 'Current state eval')
    self.assertEqual(results[2]['is_successful'], 1)
    self.assertIn(results[2]['goal'], 'ADB eval')

  @mock.patch.object(interface, 'AsyncAndroidEnv')
  def test_run_task_suite(self, mock_env):
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        # for each, add task_template
        episode_runner.EpisodeResult(
            False,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]

    suite = suite_utils.Suite(
        **{
            'Task1': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'Task2': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'
    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, demo_mode=False
    )

    self.assertTaskResults(result)

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_resume_from_middle(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    # Simulating partially completed Task1
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0.0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
            'episode_length': 1,
            'run_time': 0,
        },
    ]
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]

    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'
    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    mock_checkpointer.load.assert_called_once()
    # Run one instance for Task1, one instance for Task2
    mock_run_e2e.assert_called()
    mock_checkpointer.save_episodes.assert_has_calls([
        mock.call(mock.ANY, 'FakeCurrentStateEval_1'),
        mock.call(mock.ANY, 'FakeAdbEval_0'),
    ])

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_start_from_beginning(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = []
    mock_env.get_state.return_value = (
        dm_env.TimeStep(
            observation={'pixels': np.zeros((3, 3, 3))},
            reward=0,
            discount=0,
            step_type=dm_env.StepType.LAST,
        ),
        [],
    )
    mock_run_e2e = mock.MagicMock()
    mock_run_e2e.side_effect = [
        episode_runner.EpisodeResult(
            False,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
        episode_runner.EpisodeResult(
            True,
            {'step_number': [0]},
        ),
    ]
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    self.assertEqual(mock_run_e2e.call_count, 3)
    mock_checkpointer.load.assert_called_once()
    mock_checkpointer.save_episodes.assert_has_calls(
        [
            mock.call(mock.ANY, 'FakeCurrentStateEval_0'),
            mock.call(mock.ANY, 'FakeCurrentStateEval_1'),
            mock.call(mock.ANY, 'FakeAdbEval_0'),
        ],
        any_order=False,
    )

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_start_from_end(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 1,
            'is_successful': 1,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 0,
            'is_successful': 1,
            'goal': 'ADB eval',
            'task_template': 'FakeAdbEval',
        },
    ]
    mock_run_e2e = mock.MagicMock()
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )

    self.assertTaskResults(result)
    mock_run_e2e.assert_not_called()
    mock_checkpointer.load.assert_called_once()
    mock_checkpointer.save.assert_not_called()

  @mock.patch.object(time, 'sleep', autospec=True)
  @mock.patch.object(interface, 'AsyncAndroidEnv')
  @mock.patch.object(adb_utils, 'send_android_intent')
  @mock.patch.object(checkpointer, 'Checkpointer')
  def test_result_suite_equal_in_number(
      self,
      mock_checkpointer,
      unused_mock_send_android_intent,
      mock_env,
      unused_mock_sleep,
  ):
    mock_checkpointer.load.return_value = [
        {
            'instance_id': 0,
            'is_successful': 0,
            'goal': 'Current state eval',
            'task_template': 'FakeCurrentStateEval',
        },
        {
            'instance_id': 0,
            'is_successful': 1,
            'goal': 'ADB eval',
            'task_template': 'FakeAdbEval',
        },
    ]
    mock_run_e2e = mock.MagicMock()
    suite = suite_utils.Suite(
        **{
            'FakeCurrentStateEval': [
                test_utils.FakeCurrentStateEval(
                    test_utils.FakeCurrentStateEval.generate_random_params()
                ),
            ],
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    suite.suite_family = 'android'

    result = suite_utils._run_task_suite(
        suite, mock_run_e2e, mock_env, mock_checkpointer
    )
    self.assertLen(result, 2)

    suite2 = suite_utils.Suite(
        **{
            'FakeAdbEval': [
                test_utils.FakeAdbEval(
                    test_utils.FakeAdbEval.generate_random_params()
                )
            ],
        },
    )
    result2 = suite_utils._run_task_suite(
        suite2, mock_run_e2e, mock_env, mock_checkpointer
    )
    self.assertLen(result2, 1)


if __name__ == '__main__':
  absltest.main()
