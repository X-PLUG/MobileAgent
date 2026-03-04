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
from absl.testing import parameterized
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import recipe
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import app_snapshot
from android_world.utils import file_utils


class RecipeTestBase(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(sqlite_validators.SQLiteApp, 'initialize_task')
    )
    self.enter_context(
        mock.patch.object(
            sqlite_validators.DeleteMultipleRows, 'initialize_task'
        )
    )
    self.enter_context(
        mock.patch.object(sqlite_validators.AddMultipleRows, 'initialize_task')
    )
    self.mock_env = mock.create_autospec(interface.AsyncEnv)

  def tearDown(self):
    super().tearDown()
    self.mock_env.stop()


class RecipeDeleteMultipleRecipesTest(RecipeTestBase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.MagicMock()

  def tearDown(self):
    super().tearDown()
    self.mock_env.stop()

  def test_goal_generation(self):
    params = recipe._RecipeDeleteMultipleRecipes.generate_random_params()
    instance = recipe._RecipeDeleteMultipleRecipes(params)
    instance.params[sqlite_validators.ROW_OBJECTS] = [
        sqlite_schema_utils.Recipe(title='Recipe 1'),
        sqlite_schema_utils.Recipe(title='Recipe 2'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
    ]
    expected_goal = (
        'Delete the following recipes from Broccoli app: Recipe 1, Recipe 2,'
        ' Recipe 3.'
    )
    self.assertEqual(instance.goal, expected_goal)

  @mock.patch.object(recipe, '_generate_random_recipe')
  def test_generate_random_params(
      self,
      mock_generate_random_recipe,
  ):
    mock_generate_random_recipe.side_effect = [
        sqlite_schema_utils.Recipe(title='Recipe 1'),
        sqlite_schema_utils.Recipe(title='Recipe 2'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 4'),
        sqlite_schema_utils.Recipe(title='Recipe 5'),
    ]
    recipe._RecipeDeleteMultipleRecipes.n_rows_noise = 2
    recipe._RecipeDeleteMultipleRecipes.n_rows = 3
    params = recipe._RecipeDeleteMultipleRecipes.generate_random_params()

    expected_noise_rows = [
        sqlite_schema_utils.Recipe(title='Recipe 1'),
        sqlite_schema_utils.Recipe(title='Recipe 2'),
    ]
    expected_target_rows = [
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 4'),
        sqlite_schema_utils.Recipe(title='Recipe 5'),
    ]

    self.assertEqual(
        params[sqlite_validators.NOISE_ROW_OBJECTS], expected_noise_rows
    )
    self.assertEqual(
        params[sqlite_validators.ROW_OBJECTS], expected_target_rows
    )


# Create test for deleting duplicate recipes; here we also test the underlying
# logic from sqlite_base.DeleteMultipleDuplicateRows in lieu of tests for the
# base class.
class RecipeDeleteDuplicateRecipesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(interface.AsyncEnv)
    self.mock_env.base_env = mock.create_autospec(
        env_interface.AndroidEnvInterface
    )
    # enter context
    self.mock_list_rows = self.enter_context(
        mock.patch.object(
            sqlite_validators.SQLiteApp, 'list_rows', return_value=[]
        )
    )
    self.mock_add_rows = self.enter_context(
        mock.patch.object(sqlite_validators.SQLiteApp, 'add_rows')
    )
    self.mock_tmp_directory_from_device = self.enter_context(
        mock.patch.object(file_utils, 'tmp_directory_from_device')
    )
    self.mock_issue_generic_request = self.enter_context(
        mock.patch.object(adb_utils, 'issue_generic_request')
    )
    self.mock_remove_files = self.enter_context(
        mock.patch.object(file_utils, 'clear_directory')
    )
    self.mock_clear_db = self.enter_context(
        mock.patch.object(sqlite_validators.SQLiteApp, '_clear_db')
    )
    self.mock_restore_snapshot = self.enter_context(
        mock.patch.object(app_snapshot, 'restore_snapshot')
    )

    self.params = {
        sqlite_validators.NOISE_ROW_OBJECTS: [
            sqlite_schema_utils.Recipe(title='Unique Recipe 1', recipeId=1),
            sqlite_schema_utils.Recipe(title='Unique Recipe 2', recipeId=2),
            sqlite_schema_utils.Recipe(title='Unique Recipe 3', recipeId=3),
            sqlite_schema_utils.Recipe(title='Unique Recipe 4', recipeId=4),
            sqlite_schema_utils.Recipe(title='Unique Recipe 5', recipeId=5),
        ],
        sqlite_validators.ROW_OBJECTS: [
            sqlite_schema_utils.Recipe(title='Duplicate Recipe', recipeId=6),
            sqlite_schema_utils.Recipe(title='Duplicate Recipe', recipeId=7),
        ],
    }
    self.instance = recipe.RecipeDeleteDuplicateRecipes(self.params)

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_initialize_task(self):
    """Test initialization of the task with proper setup of duplicate rows."""
    self.mock_list_rows.return_value = (
        self.params[sqlite_validators.ROW_OBJECTS]
        + self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    )

    self.instance.initialize_task(self.mock_env)

    self.mock_add_rows.assert_has_calls([
        mock.call(
            self.params[sqlite_validators.NOISE_ROW_OBJECTS],
            self.mock_env,
        ),
        mock.call(self.params[sqlite_validators.ROW_OBJECTS], self.mock_env),
    ])

    self.assertLen(
        self.instance.rows_to_delete,
        2,
        'Should have initialized two duplicate rows',
    )
    self.instance._validate_initial_state(self.instance.before)

  def test_initialize_task_failure(self):
    """Test initialization of the task with proper setup of duplicate rows."""
    self.mock_list_rows.return_value = (
        self.params[sqlite_validators.ROW_OBJECTS][0:1]
        + self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    )

    with self.assertRaises(ValueError):
      self.instance.initialize_task(self.mock_env)

  @parameterized.named_parameters(
      ('first_index_kept', 0),
      ('second_index_kept', 1),
  )
  def test_is_successful(self, index):
    """Test the success of duplicate deletion."""
    self.mock_list_rows.side_effect = [
        self.params[sqlite_validators.NOISE_ROW_OBJECTS]
        + self.params[sqlite_validators.ROW_OBJECTS],
        self.params[sqlite_validators.NOISE_ROW_OBJECTS]
        + [self.params[sqlite_validators.ROW_OBJECTS][index]],
    ]
    self.instance.initialize_task(self.mock_env)

    success = self.instance.is_successful(self.mock_env)

    self.assertEqual(success, 1.0, 'Deletion should be successful')

  def test_is_successful_failure_both_removed(
      self,
  ):
    """Test the success of duplicate deletion."""
    self.mock_list_rows.side_effect = [
        self.params[sqlite_validators.ROW_OBJECTS]
        + self.params[sqlite_validators.NOISE_ROW_OBJECTS],
        self.params[sqlite_validators.NOISE_ROW_OBJECTS],
    ]

    self.instance.initialize_task(self.mock_env)
    success = self.instance.is_successful(self.mock_env)

    self.assertEqual(success, 0.0, 'Deletion should not be successful')

  def test_is_successful_failure_neither_removed(
      self,
  ):
    """Test the success of duplicate deletion."""
    self.mock_list_rows.side_effect = [
        self.params[sqlite_validators.ROW_OBJECTS]
        + self.params[sqlite_validators.NOISE_ROW_OBJECTS],
        self.params[sqlite_validators.NOISE_ROW_OBJECTS]
        + self.params[sqlite_validators.ROW_OBJECTS],
    ]

    self.instance.initialize_task(self.mock_env)
    success = self.instance.is_successful(self.mock_env)

    self.assertEqual(success, 0.0, 'Deletion should not be successful')

  @mock.patch.object(recipe, '_generate_random_recipe')
  def test_generate_random_params(self, mock_generate_random_recipe):
    mock_generate_random_recipe.side_effect = [
        sqlite_schema_utils.Recipe(title='Recipe 1'),
        sqlite_schema_utils.Recipe(title='Recipe 2'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 4'),
        sqlite_schema_utils.Recipe(title='Recipe 5'),
        sqlite_schema_utils.Recipe(title='Recipe 6'),
    ]
    params = recipe.RecipeDeleteDuplicateRecipes.generate_random_params()
    self.assertEqual(
        params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(title='Recipe 1'),
            sqlite_schema_utils.Recipe(title='Recipe 2'),
            sqlite_schema_utils.Recipe(title='Recipe 3'),
            sqlite_schema_utils.Recipe(title='Recipe 4'),
            sqlite_schema_utils.Recipe(title='Recipe 5'),
        ],
    )
    self.assertEqual(
        params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(title='Recipe 6'),
            sqlite_schema_utils.Recipe(title='Recipe 6'),
        ],
    )


class BroccoliDeleteDuplicateRecipesTest2(parameterized.TestCase):

  @mock.patch.object(recipe, '_generate_random_recipe')
  def test_generate_random_params(self, mock_generate_random_recipe):
    first = [
        sqlite_schema_utils.Recipe(title='Recipe 1'),
        sqlite_schema_utils.Recipe(title='Recipe 2'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 3'),
        sqlite_schema_utils.Recipe(title='Recipe 4'),
        sqlite_schema_utils.Recipe(title='Recipe 5'),
        sqlite_schema_utils.Recipe(title='Recipe 6'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
    ]
    second = [
        sqlite_schema_utils.Recipe(title='Recipe 4'),
        sqlite_schema_utils.Recipe(title='Recipe 5'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
        sqlite_schema_utils.Recipe(title='Recipe 6'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='variation 1'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='variation 2'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='variation 3'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
        sqlite_schema_utils.Recipe(title='Recipe 7', description='variation 4'),
    ]
    mock_generate_random_recipe.side_effect = first + second
    params = recipe.RecipeDeleteDuplicateRecipes2.generate_random_params()
    self.assertEqual(
        params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
            sqlite_schema_utils.Recipe(title='Recipe 7', description='target'),
        ],
    )
    self.assertEqual(
        params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(title='Recipe 1'),
            sqlite_schema_utils.Recipe(title='Recipe 2'),
            sqlite_schema_utils.Recipe(title='Recipe 3'),
            sqlite_schema_utils.Recipe(title='Recipe 4'),
            sqlite_schema_utils.Recipe(title='Recipe 5'),
            sqlite_schema_utils.Recipe(title='Recipe 6'),
            sqlite_schema_utils.Recipe(
                title='Recipe 7', description='variation 1'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 7', description='variation 2'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 7', description='variation 3'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 7', description='variation 4'
            ),
        ],
    )


class TestRecipeDeleteMultipleRecipesWithConstraint(parameterized.TestCase):

  @mock.patch.object(random, 'choice', return_value='garlic')
  @mock.patch.object(recipe, '_generate_random_recipe')
  def test_generate_random_params(
      self, mock_generate_random_recipe, unused_mock_choice
  ):
    first = [
        sqlite_schema_utils.Recipe(title='Recipe 1', directions='Stir a lot'),
        sqlite_schema_utils.Recipe(
            title='Recipe 2', directions='Contains garlic.'
        ),
        sqlite_schema_utils.Recipe(
            title='Recipe 3', directions='Also garlic here.'
        ),
        sqlite_schema_utils.Recipe(title='Recipe 4', directions='Add anchoves'),
    ]
    second = [
        sqlite_schema_utils.Recipe(
            title='Recipe 6', directions='Contains garlic.'
        ),
        sqlite_schema_utils.Recipe(title='Recipe 5', directions='Stir a lot'),
        sqlite_schema_utils.Recipe(
            title='Recipe 7', directions='Also garlic here.'
        ),
        sqlite_schema_utils.Recipe(title='Recipe 8', directions='Add anchoves'),
        sqlite_schema_utils.Recipe(
            title='Recipe 8', directions='Add anchoves again'
        ),
        sqlite_schema_utils.Recipe(
            title='Recipe 7', directions='This contains garlic.'
        ),
    ]
    mock_generate_random_recipe.side_effect = first + second

    recipe.RecipeDeleteMultipleRecipesWithConstraint.n_rows = 2
    recipe.RecipeDeleteMultipleRecipesWithConstraint.n_rows_noise = 2
    params = (
        recipe.RecipeDeleteMultipleRecipesWithConstraint.generate_random_params()
    )

    self.assertEqual(
        params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(
                title='Recipe 1', directions='Stir a lot'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 4', directions='Add anchoves'
            ),
        ],
    )
    self.assertEqual(
        params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(
                title='Recipe 6', directions='Contains garlic.'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 7', directions='Also garlic here.'
            ),
        ],
    )


class AddMultipleRecipesForTest(recipe._RecipeAddMultipleRecipes):
  n_rows = 2
  n_rows_noise = 2


class TestRecipeAddMultipleRecipes(absltest.TestCase):

  @mock.patch.object(recipe, '_generate_random_recipe')
  @mock.patch.object(random, 'choice', return_value='text_block')
  def test_generate_random_params(
      self, mock_choice, mock_generate_random_recipe
  ):
    target_rows = [
        sqlite_schema_utils.Recipe(title='Recipe 1', ingredients='Tomatoes'),
        sqlite_schema_utils.Recipe(title='Recipe 2', ingredients='Garlic'),
    ]
    noise_rows = [
        sqlite_schema_utils.Recipe(title='Noise 1', ingredients='Salt'),
        sqlite_schema_utils.Recipe(title='Noise 2', ingredients='Pepper'),
    ]
    mock_generate_random_recipe.side_effect = target_rows + noise_rows

    params = AddMultipleRecipesForTest.generate_random_params()

    mock_choice.assert_called_with(['csv', 'text_block'])

    self.assertEqual(params[sqlite_validators.ROW_OBJECTS], target_rows)
    self.assertEqual(params[sqlite_validators.NOISE_ROW_OBJECTS], noise_rows)
    self.assertEqual(params[recipe._TEXT_REPRESENTATION_TYPE], 'text_block')


class RecipeAddMultipleRecipesFromMarkor2ForTest(
    recipe.RecipeAddMultipleRecipesFromMarkor2
):
  n_rows = 2
  n_rows_noise = 2


class TestRecipeAddMultipleRecipesFromMarkor2(absltest.TestCase):

  @mock.patch.object(recipe, '_generate_random_recipe')
  @mock.patch.object(random, 'choice')
  def test_generate_random_params(
      self, mock_choice, mock_generate_random_recipe
  ):
    mock_generate_random_recipe.side_effect = [
        sqlite_schema_utils.Recipe(title='Recipe 1', preparationTime='10 mins'),
        sqlite_schema_utils.Recipe(title='Recipe 2', preparationTime='40 mins'),
        sqlite_schema_utils.Recipe(title='Recipe 3', preparationTime='10 mins'),
        sqlite_schema_utils.Recipe(
            title='Noise 1', ingredients='Salt', preparationTime='15 mins'
        ),
        sqlite_schema_utils.Recipe(
            title='Noise 2', ingredients='Salt', preparationTime='10 mins'
        ),
        sqlite_schema_utils.Recipe(
            title='Noise 3', ingredients='Salt', preparationTime='30 mins'
        ),
    ]
    mock_choice.side_effect = ['10 mins', 'text_block']

    params = RecipeAddMultipleRecipesFromMarkor2ForTest.generate_random_params()

    mock_choice.assert_called_with(['csv', 'text_block'])

    self.assertEqual(
        params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(
                title='Recipe 1', preparationTime='10 mins'
            ),
            sqlite_schema_utils.Recipe(
                title='Recipe 3', preparationTime='10 mins'
            ),
        ],
    )
    self.assertEqual(
        params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Recipe(
                title='Noise 1', ingredients='Salt', preparationTime='15 mins'
            ),
            sqlite_schema_utils.Recipe(
                title='Noise 3', ingredients='Salt', preparationTime='30 mins'
            ),
        ],
    )
    self.assertEqual(params[recipe._TEXT_REPRESENTATION_TYPE], 'text_block')


if __name__ == '__main__':
  absltest.main()
