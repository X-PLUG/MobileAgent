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

import random
from unittest import mock
from absl.testing import absltest
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.information_retrieval import activity_app_utils
from android_world.task_evals.information_retrieval import calendar_utils
from android_world.task_evals.information_retrieval import information_retrieval
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2

DEFAULT_TASK_TEMPLATE = task_pb2.Task(
    name="test_task",
    prompt="Test {prompt} for {start_date}",
    relevant_state=task_pb2.RelevantState(
        state=state_pb2.State(
            calendar=state_pb2.Calendar(
                app_name="simple calendar pro",
                events=[
                    state_pb2.Event(
                        start_date="{start_date}",
                        start_time="10am",
                        title="{title}",
                    )
                ],
            ),
            tasks_app=state_pb2.TasksApp(
                tasks_app_tasks=[
                    state_pb2.TasksAppTask(
                        completed_date="{completed_date}",
                        importance="{importance}",
                        title="Pay Rent",
                    )
                ]
            ),
            sports_activity_app=state_pb2.SportsActivityApp(
                sports_activities=[
                    state_pb2.SportsActivity(
                        start_date="{start_date}",
                        start_time="14:00",
                        category="{category}",
                        name="Morning run",
                    )
                ]
            ),
        ),
        exclusion_conditions=[
            task_pb2.ExclusionCondition(
                operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                field="start_date",
                value="October 16 2023",
            ),
            task_pb2.ExclusionCondition(
                operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                field="completed_date",
                value="October 15 2023",
            ),
            task_pb2.ExclusionCondition(
                operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                field="category",
                value="biking",
            ),
        ],
    ),
    task_params=[
        task_pb2.TaskParams(name="prompt", possible_values=["perform {task}"]),
        task_pb2.TaskParams(
            name="start_date", possible_values=["October 15 2023"]
        ),
        task_pb2.TaskParams(name="title", possible_values=["test title"]),
        task_pb2.TaskParams(
            name="completed_date", possible_values=["October 16 2023"]
        ),
        task_pb2.TaskParams(name="importance", possible_values=["0"]),
        task_pb2.TaskParams(name="category", possible_values=["running"]),
        task_pb2.TaskParams(name="task", possible_values=["task"]),
    ],
    success_criteria=task_pb2.SuccessCriteria(
        expectations=[
            task_pb2.Expectation(
                field_transformation=task_pb2.FieldTransformation(
                    field_name="title",
                ),
                match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
            ),
        ]
    ),
)


class InformationRetrievalTaskForTest(
    information_retrieval.InformationRetrieval
):

  @classmethod
  def generate_random_params(cls):
    return {
        "prompt": "perform {task}",
        "start_date": "October 15 2023",
        "title": "test title",
        "importance": "0",
        "completed_date": "October 16 2023",
        "category": "running",
        "task": "task",
    }

  def __init__(
      self,
      params: dict[str, str],
      template: task_pb2.Task = DEFAULT_TASK_TEMPLATE,
  ):
    self._task_template = template
    super().__init__(params)

  @property
  def task_template(self) -> task_pb2.Task:
    return self._task_template


class InformationRetrievalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(interface.AsyncEnv)
    self.params = {"task_name": "test_task"}
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params()
    )
    self.mock_initialize_task = mock.patch.object(
        task_eval.TaskEval, "initialize_task"
    ).start()
    self.mock_is_successful = mock.patch.object(
        task_eval.TaskEval, "is_successful"
    ).start()
    self.mock_initialize_calendar_app = mock.patch.object(
        calendar_utils, "setup_task_state"
    ).start()
    self.mock_initialize_tasks_app = mock.patch.object(
        task_app_utils, "setup_task_state"
    ).start()
    self.mock_initialize_sports_app = mock.patch.object(
        activity_app_utils, "setup_task_state"
    ).start()

  @mock.patch.object(random, "choice")
  def test_initialize_task(self, mock_choice):
    mock_choice.return_value = "today"
    expected_task = task_pb2.Task(
        name="test_task",
        prompt="Test {prompt} for {start_date}",
        relevant_state=task_pb2.RelevantState(
            state=state_pb2.State(
                calendar=state_pb2.Calendar(
                    app_name="simple calendar pro",
                    events=[
                        state_pb2.Event(
                            start_date="October 15 2023",
                            start_time="10am",
                            title="test title",
                        )
                    ],
                ),
                tasks_app=state_pb2.TasksApp(
                    tasks_app_tasks=[
                        state_pb2.TasksAppTask(
                            completed_date="October 16 2023",
                            importance="0",
                            title="Pay Rent",
                        )
                    ]
                ),
                sports_activity_app=state_pb2.SportsActivityApp(
                    sports_activities=[
                        state_pb2.SportsActivity(
                            start_date="October 15 2023",
                            start_time="14:00",
                            category="running",
                            name="Morning run",
                        )
                    ]
                ),
            ),
            exclusion_conditions=[
                task_pb2.ExclusionCondition(
                    operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                    field="start_date",
                    value="October 16 2023",
                ),
                task_pb2.ExclusionCondition(
                    operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                    field="completed_date",
                    value="October 15 2023",
                ),
                task_pb2.ExclusionCondition(
                    operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
                    field="category",
                    value="biking",
                ),
            ],
        ),
        task_params=[
            task_pb2.TaskParams(
                name="prompt", possible_values=["perform task"]
            ),
            task_pb2.TaskParams(
                name="start_date", possible_values=["October 15 2023"]
            ),
            task_pb2.TaskParams(name="title", possible_values=["test title"]),
            task_pb2.TaskParams(
                name="completed_date", possible_values=["October 16 2023"]
            ),
            task_pb2.TaskParams(name="importance", possible_values=["0"]),
            task_pb2.TaskParams(name="category", possible_values=["running"]),
            task_pb2.TaskParams(name="task", possible_values=["task"]),
        ],
        success_criteria=task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="title",
                    ),
                    match_type=task_pb2.Expectation.MatchType.STRING_MATCH,
                ),
            ]
        ),
    )

    self.mock_task.initialize_task(self.mock_env)
    self.mock_initialize_task.assert_called_once_with(self.mock_env)
    expected_exclusion_conditions = [
        task_pb2.ExclusionCondition(
            operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
            field="start_date",
            value="October 16 2023",
        ),
        task_pb2.ExclusionCondition(
            operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
            field="completed_date",
            value="October 15 2023",
        ),
        task_pb2.ExclusionCondition(
            operation=task_pb2.ExclusionCondition.Operation.EQUAL_TO,
            field="category",
            value="biking",
        ),
    ]
    self.mock_initialize_calendar_app.assert_called_once_with(
        self.mock_task.task.relevant_state.state.calendar,
        expected_exclusion_conditions,
        self.mock_env,
    )
    self.mock_initialize_tasks_app.assert_called_once_with(
        self.mock_task.task.relevant_state.state.tasks_app,
        expected_exclusion_conditions,
        self.mock_env,
    )
    self.mock_initialize_sports_app.assert_called_once_with(
        self.mock_task.task.relevant_state.state.sports_activity_app,
        expected_exclusion_conditions,
        self.mock_env,
    )
    self.assertEqual(self.mock_task.task, expected_task)
    self.assertEqual(self.mock_task.goal, "Test perform task for today")

  def test_is_successful_string_match_succeeds(self):
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = "Test title, Pay Rent"

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1.0)

  def test_is_successful_number_match_succeeds(self):
    new_template = task_pb2.Task()
    new_template.CopyFrom(self.mock_task.task_template)
    new_template.success_criteria.CopyFrom(
        task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="importance",
                    ),
                    match_type=task_pb2.Expectation.MatchType.NUMBER_MATCH,
                ),
            ]
        )
    )
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params(), new_template
    )
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = "0"

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1.0)

  def test_is_successful_date_match_succeeds(self):
    new_template = task_pb2.Task()
    new_template.CopyFrom(self.mock_task.task_template)
    new_template.success_criteria.CopyFrom(
        task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="start_date",
                    ),
                    match_type=task_pb2.Expectation.MatchType.DATE_MATCH,
                ),
            ]
        )
    )
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params(), new_template
    )
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = "October 15 2023, October 15 2023"

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1.0)

  def test_is_successful_time_match_succeeds(self):
    new_template = task_pb2.Task()
    new_template.CopyFrom(self.mock_task.task_template)
    new_template.success_criteria.CopyFrom(
        task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="start_time",
                    ),
                    match_type=task_pb2.Expectation.MatchType.TIME_MATCH,
                ),
            ]
        )
    )
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params(), new_template
    )
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = "10:00, 14:00"

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1.0)

  def test_is_successful_datetime_match_succeeds(self):
    new_template = task_pb2.Task()
    new_template.CopyFrom(self.mock_task.task_template)
    new_template.success_criteria.CopyFrom(
        task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="start_date",
                    ),
                    match_type=task_pb2.Expectation.MatchType.DATE_MATCH,
                ),
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="start_time",
                    ),
                    match_type=task_pb2.Expectation.MatchType.TIME_MATCH,
                ),
            ]
        )
    )
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params(), new_template
    )
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = (
        "October 15 2023 10:00, October 15 2023 14:00"
    )

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 1.0)

  def test_is_successful_match_fails(self):
    new_template = task_pb2.Task()
    new_template.CopyFrom(self.mock_task.task_template)
    new_template.success_criteria.CopyFrom(
        task_pb2.SuccessCriteria(
            expectations=[
                task_pb2.Expectation(
                    field_transformation=task_pb2.FieldTransformation(
                        field_name="start_date",
                    ),
                    match_type=task_pb2.Expectation.MatchType.DATE_MATCH,
                ),
            ]
        )
    )
    self.mock_task = InformationRetrievalTaskForTest(
        InformationRetrievalTaskForTest.generate_random_params(), new_template
    )
    self.mock_task.initialize_task(self.mock_env)
    self.mock_env.interaction_cache = "October 15 2023, October 16 2023"

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 0.0)

  def test_is_successful_empty_interaction_cache_fails(self):
    self.mock_env.interaction_cache = ""
    self.mock_task.initialize_task(self.mock_env)

    self.assertEqual(self.mock_task.is_successful(self.mock_env), 0.0)


if __name__ == "__main__":
  absltest.main()
