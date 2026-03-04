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

"""Tasks for recipes app."""

import dataclasses
import random
from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


_DB_PATH = '/data/data/com.flauschcode.broccoli/databases/broccoli'
_TABLE_NAME = 'recipes'
_APP_NAME = 'broccoli app'
_DB_KEY = 'recipeId'

# How to represent recipes in text form (csv or block of text) for generated
# files.
_TEXT_REPRESENTATION_TYPE = 'text_representation_type'


class _RecipeApp(sqlite_validators.SQLiteApp):
  # From TaskEval.
  schema = {}
  app_names = (_APP_NAME,)
  template = ''  # Unused, since we directly build goal in implementations.

  # From sqlite_base.SQLiteApp
  app_name_with_db = _APP_NAME
  db_key = _DB_KEY
  db_path = _DB_PATH
  table_name = _TABLE_NAME
  row_type = sqlite_schema_utils.Recipe


class _RecipeDeleteMultipleRecipes(
    sqlite_validators.DeleteMultipleRows, _RecipeApp
):
  """Task to delete multiple recipes in Broccoli Recipe App."""

  complexity = 2
  n_rows = 3
  n_rows_noise = 0

  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    titles = [r.title for r in targets]
    titles = ', '.join(titles)
    return f'Delete the following recipes from Broccoli app: {titles}.'

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Recipe],
      after: list[sqlite_schema_utils.Recipe],
  ) -> bool:
    """Validates the integrity of the recipe deletion."""
    return sqlite_validators.validate_rows_removal_integrity(
        before, after, [r.recipeId for r in self.rows_to_delete], self.db_key
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove recipe task."""

    recipes = []
    while len(recipes) < cls.n_rows + cls.n_rows_noise:
      candidate = _generate_random_recipe()
      if not any([candidate.title == r.title for r in recipes]):
        recipes.append(candidate)

    if cls.n_rows_noise > 0:
      noise_rows = recipes[: cls.n_rows_noise]
      target_rows = recipes[cls.n_rows_noise :]
      return {
          sqlite_validators.ROW_OBJECTS: target_rows,
          sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
      }
    else:
      return {
          sqlite_validators.ROW_OBJECTS: recipes,
      }


class RecipeDeleteSingleRecipe(_RecipeDeleteMultipleRecipes):
  """Delete single recipe in Broccoli Recipe App without noise."""

  complexity = 1
  n_rows = 1
  n_rows_noise = 0


class RecipeDeleteSingleWithRecipeWithNoise(_RecipeDeleteMultipleRecipes):
  """Delete single recipe in Broccoli Recipe App with noise."""

  complexity = 2
  n_rows = 1
  n_rows_noise = 29


class RecipeDeleteMultipleRecipes(_RecipeDeleteMultipleRecipes):
  """Delete multiple recipes in Broccoli Recipe App."""

  complexity = 2.4
  n_rows = 3
  n_rows_noise = 0


class RecipeDeleteMultipleRecipesWithNoise(_RecipeDeleteMultipleRecipes):
  """Delete multiple recipes in Broccoli Recipe App with noise."""

  complexity = 3.4
  n_rows = 3
  n_rows_noise = 29


class RecipeDeleteMultipleRecipesWithConstraint(_RecipeDeleteMultipleRecipes):
  """Delete multiple recipes in Broccoli Recipe App based on ingredient."""

  complexity = 4
  n_rows = 3
  n_rows_noise = 29

  @property
  def goal(self) -> str:
    ingredient = self.params['ingredient']
    return (
        f'Delete the recipes from Broccoli app that use {ingredient} in the'
        ' directions.'
    )

  def _validate_initial_state(
      self, before: list[sqlite_schema_utils.RowType]
  ) -> None:
    del before

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove recipe task."""
    ingredient = random.choice(_COMMON_INGREDIENTS)
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: ingredient not in r.directions.lower(),
    )
    targets = []
    n_rows = cls.n_rows
    while n_rows > 0:
      try:
        targets = sqlite_schema_utils.get_random_items(
            n_rows,
            _generate_random_recipe,
            replacement=False,
            filter_fn=lambda r: ingredient in r.directions.lower(),
        )
        break
      except ValueError:
        n_rows -= 1
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        'ingredient': ingredient,
    }


class RecipeDeleteDuplicateRecipes(
    sqlite_validators.DeleteDuplicateRows, _RecipeApp
):
  """Deduplicate recipes from Broccoli Recipe App."""

  complexity = 1
  n_rows = 1
  n_rows_noise = 5

  @property
  def goal(self) -> str:
    return (
        'Delete all but one of any recipes in the Broccoli app that are exact'
        ' duplicates, ensuring at least one instance of each unique recipe'
        ' remains'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Recipe],
      after: list[sqlite_schema_utils.Recipe],
  ) -> bool:
    """Validates the integrity of the recipe deletion."""
    target1, target2 = self.rows_to_delete
    return sqlite_validators.validate_rows_removal_integrity(
        before, after, [target1.recipeId], self.db_key
    ) or sqlite_validators.validate_rows_removal_integrity(
        before, after, [target2.recipeId], self.db_key
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove recipe task."""

    rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise + cls.n_rows,
        _generate_random_recipe,
        replacement=False,
    )
    target = rows.pop()
    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: rows,
    }


class RecipeDeleteDuplicateRecipes2(RecipeDeleteDuplicateRecipes):
  """Medium hard deduplication task, with more noise events."""

  complexity = 2.4
  n_rows = 1
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return (
        'Delete all but one of any recipes in the Broccoli app that are exact'
        ' duplicates, ensuring at least one instance of each unique recipe'
        ' remains'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove recipe task."""
    noise = sqlite_schema_utils.get_random_items(
        7,
        _generate_random_recipe,
        replacement=False,
    )

    target = noise.pop()

    # Add variations of target recipe, with different properties.
    while len(noise) < cls.n_rows_noise:
      value = sqlite_schema_utils.get_random_items(
          1,
          _generate_random_recipe,
          replacement=True,
          filter_fn=lambda r: r.title == target.title,
      )[0]
      if value != target:
        noise.append(value)

    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteDuplicateRecipes3(RecipeDeleteDuplicateRecipes):
  """Harder deduplication task, with more noise events and agent must scroll."""

  complexity = 3.4
  n_rows = 1
  n_rows_noise = 30

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove recipe task."""
    noise = sqlite_schema_utils.get_random_items(
        22,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.title != 'Avocado Toast with Egg',
    )

    target = noise.pop()

    # Add noise at the top of the recipe screen, requiring agent to scroll.
    noise += sqlite_schema_utils.get_random_items(
        3,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.title == 'Avocado Toast with Egg',
    )

    # Add variations of target recipe, with different properties.
    while len(noise) < cls.n_rows_noise:
      value = sqlite_schema_utils.get_random_items(
          1,
          _generate_random_recipe,
          replacement=True,
          filter_fn=lambda r: (
              r.title == target.title and r.description == target.description
          ),
      )[0]
      if value != target:
        noise.append(value)

    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


def _get_rows_as_text(
    rows: list[sqlite_schema_utils.Recipe],
    format_type: str,
    wrap_width: int | None = None,
) -> str:
  return sqlite_schema_utils.get_text_representation_of_rows(
      rows,
      [
          'title',
          'description',
          'servings',
          'preparationTime',
          'ingredients',
          'directions',
      ],
      format_type,
      'title',
      wrap_width=wrap_width,
  )


class _RecipeAddMultipleRecipes(sqlite_validators.AddMultipleRows, _RecipeApp):
  """Task to delete multiple recipes in Broccoli Recipe App."""

  complexity = 3
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    text_repr = _get_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return f'Add the following recipes into the Broccoli app:\n{text_repr}'

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Recipe],
      after: list[sqlite_schema_utils.Recipe],
      reference_rows: list[sqlite_schema_utils.RowType],
  ) -> bool:
    """Validates the integrity of the recipe deletion."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'title',
            'description',
            'servings',
            'preparationTime',
            'source',
            'ingredients',
            'directions',
            'favorite',
        ],
        free_form_fields=[
            'title',
            'description',
            'servings',
            'preparationTime',
            'source',
            'ingredients',
            'directions',
        ],
    )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    """Currently unused."""
    return _generate_random_recipe()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add recipe task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        _generate_random_recipe,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: any([r.title != t.title for t in target_rows]),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }


class RecipeAddSingleRecipe(_RecipeAddMultipleRecipes):
  """Task to delete a single recipe in Broccoli Recipe App."""

  complexity = 2.4
  n_rows = 1
  n_rows_noise = 10


class RecipeAddMultipleRecipes(_RecipeAddMultipleRecipes):
  """Task to delete multiple recipes in Broccoli Recipe App."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10


class RecipeAddMultipleRecipesFromMarkor(_RecipeAddMultipleRecipes):
  """Task to add multiple recipes from a text file to Broccoli Recipe App."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return (
        'Add the recipes from recipes.txt in Markor to the Broccoli recipe app.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
    user_data_generation.write_to_markor(
        _get_rows_as_text(
            self.params[sqlite_validators.ROW_OBJECTS],
            self.params[_TEXT_REPRESENTATION_TYPE],
        ),
        'recipes.txt',
        env,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class RecipeAddMultipleRecipesFromMarkor2(RecipeAddMultipleRecipesFromMarkor):
  """Harder add recipe task, that involves navigating a large text file."""

  n_rows = 3
  n_rows_noise = 40
  complexity = 6

  @property
  def goal(self) -> str:
    prep_time = self.params['prep_time']
    return (
        f'Add the recipes from recipes.txt in Markor that take {prep_time} to '
        'prepare into the Broccoli recipe app.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    rows = (
        self.params[sqlite_validators.ROW_OBJECTS]
        + self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    )
    random.shuffle(rows)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
    user_data_generation.write_to_markor(
        _get_rows_as_text(
            rows,
            self.params[_TEXT_REPRESENTATION_TYPE],
        ),
        'recipes.txt',
        env,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add recipe task."""
    prep_time = random.choice(_PREP_TIME_OPTIONS)
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.preparationTime == prep_time,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        _generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.preparationTime != prep_time,
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
        'prep_time': prep_time,
    }


class RecipeAddMultipleRecipesFromImage(_RecipeAddMultipleRecipes):
  """Task to add multiple recipes from an image file to Broccoli Recipe App."""

  app_names = (_APP_NAME, 'simple gallery pro')
  complexity = 6
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return (
        'Add the recipes from recipes.jpg in Simple Gallery Pro to the Broccoli'
        ' recipe app.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    data = _get_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS], 'text_block', wrap_width=60
    )
    user_data_generation.write_to_gallery(data, 'recipes.jpg', env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)

#### Utility functions used for generating recipes #############################


def _generate_random_recipe() -> sqlite_schema_utils.Recipe:
  """Generates a random recipe."""

  descriptions = [
      'A quick and easy meal, perfect for busy weekdays.',
      'A delicious and healthy choice for any time of the day.',
      (
          'An ideal recipe for experimenting with different flavors and'
          ' ingredients.'
      ),
  ]
  directions_additions = [
      'Try adding a pinch of your favorite spices for extra flavor.',
      'Feel free to substitute with ingredients you have on hand.',
      'Garnish with fresh herbs for a more vibrant taste.',
  ]
  ingredient_descriptors = [
      'see directions',
      'as per recipe',
      'varies',
      'to preference',
      'quantities to taste',
      'as needed',
      'optional ingredients',
      'n/a',
      'various amounts',
      'adjustable',
      'to your liking',
      'flexible ingredients',
      'per individual taste',
      'as desired',
      'subject to change',
  ]

  recipe = random.choice(_RECIPES)

  return dataclasses.replace(
      recipe,
      description=random.choice(descriptions),
      servings=random.choice(_SERVINGS_OPTIONS),
      preparationTime=random.choice(_PREP_TIME_OPTIONS),
      directions=f'{recipe.directions} {random.choice(directions_additions)}',
      ingredients=random.choice(ingredient_descriptors),
  )


_RECIPES = [
    sqlite_schema_utils.Recipe(
        title='Spicy Tuna Wraps',
        directions=(
            'Mix canned tuna with mayo and sriracha. Spread on tortillas, add'
            ' lettuce and cucumber slices, roll up.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Avocado Toast with Egg',
        directions=(
            'Toast bread, top with mashed avocado, a fried egg, salt, pepper,'
            ' and chili flakes.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Greek Salad Pita Pockets',
        directions=(
            'Fill pita pockets with lettuce, cucumber, tomato, feta, olives,'
            ' and Greek dressing.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Quick Fried Rice',
        directions=(
            'Sauté cooked rice with vegetables, add soy sauce and scrambled'
            ' eggs. Toss until hot.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Pesto Pasta with Peas',
        directions=(
            'Cook pasta, stir in pesto sauce and cooked peas. Add Parmesan'
            ' cheese before serving.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='BBQ Chicken Quesadillas',
        directions=(
            'Mix shredded cooked chicken with BBQ sauce. Place on tortillas'
            ' with cheese, fold and cook until crispy.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Tomato Basil Bruschetta',
        directions=(
            'Top sliced baguette with a mix of chopped tomatoes, basil,'
            ' garlic, olive oil, salt, and pepper.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Lemon Garlic Tilapia',
        directions=(
            'Sauté tilapia in butter, add lemon juice and garlic. Serve with'
            ' steamed vegetables.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Turkey and Cheese Panini',
        directions=(
            'Layer turkey and cheese on bread, grill in a panini press until'
            ' golden.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Veggie and Hummus Sandwich',
        directions=(
            'Spread hummus on bread, add cucumber, bell pepper, carrot, and'
            ' lettuce.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Mango Chicken Curry',
        directions=(
            'Cook chicken pieces in a pan, add onions, garlic, and ginger. Stir'
            ' in curry powder, coconut milk, and mango pieces. Simmer until'
            ' chicken is cooked.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Beef Stir Fry',
        directions=(
            'Stir-fry beef slices with broccoli, bell peppers, and onions in'
            ' soy sauce and garlic. Serve over rice or noodles.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Shrimp Avocado Salad',
        directions=(
            'Mix cooked shrimp with avocado, tomatoes, cucumber, and onion.'
            ' Dress with lime juice, olive oil, salt, and pepper.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Spinach and Feta Stuffed Chicken',
        directions=(
            'Stuff chicken breasts with a mixture of spinach, feta, garlic, and'
            ' herbs. Bake until chicken is cooked through.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Zucchini Noodles with Pesto',
        directions=(
            'Spiralize zucchini into noodles, sauté with garlic, then mix with'
            ' pesto sauce. Top with grated Parmesan cheese.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Cauliflower Fried "Rice"',
        directions=(
            'Pulse cauliflower in a food processor until it resembles rice.'
            ' Sauté with vegetables, soy sauce, and add scrambled eggs.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Sweet Potato and Black Bean Tacos',
        directions=(
            'Roast sweet potato cubes, mix with black beans, and use as filling'
            ' for tacos. Top with avocado and cilantro lime sauce.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Salmon with Dill Sauce',
        directions=(
            'Bake salmon fillets and serve with a sauce made from Greek yogurt,'
            ' dill, lemon juice, and garlic.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Quinoa Salad with Vegetables',
        directions=(
            'Mix cooked quinoa with diced vegetables, feta cheese, and a lemon'
            ' olive oil dressing.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Chickpea Vegetable Soup',
        directions=(
            'Sauté onions, carrots, and celery, add broth, canned tomatoes, and'
            ' chickpeas. Simmer with spinach and seasonings.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Chicken Caesar Salad Wrap',
        directions=(
            'Toss chopped romaine lettuce with Caesar dressing, grilled chicken'
            ' strips, and Parmesan cheese. Wrap in a large tortilla.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Vegetarian Chili',
        directions=(
            'Cook onions, garlic, bell peppers, and carrots. Add canned'
            ' tomatoes, kidney beans, black beans, corn, and chili seasoning.'
            ' Simmer until vegetables are tender.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Pan-Seared Salmon with Quinoa',
        directions=(
            'Pan-sear salmon fillets until crispy. Serve over cooked quinoa'
            ' with a side of steamed asparagus.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Caprese Salad Skewers',
        directions=(
            'Thread cherry tomatoes, basil leaves, and mozzarella balls onto'
            ' skewers. Drizzle with balsamic glaze.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Chicken Alfredo Pasta',
        directions=(
            'Cook fettuccine pasta, toss with Alfredo sauce and grilled chicken'
            ' strips. Serve with a sprinkle of Parmesan cheese.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Stuffed Bell Peppers',
        directions=(
            'Mix cooked quinoa, black beans, corn, tomato sauce, and spices.'
            ' Stuff into bell peppers and bake until tender.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Eggplant Parmesan',
        directions=(
            'Slice eggplant, bread, and fry. Layer in a baking dish with'
            ' marinara sauce and mozzarella cheese. Bake until bubbly.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Thai Peanut Noodle Salad',
        directions=(
            'Toss cooked noodles with a Thai peanut sauce, sliced red bell'
            ' peppers, cabbage, carrots, and cilantro.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Butternut Squash Soup',
        directions=(
            'Sauté onions and garlic, add cubed butternut squash and broth.'
            ' Puree until smooth and season with nutmeg, salt, and pepper.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Baked Cod with Lemon and Dill',
        directions=(
            'Place cod fillets in a baking dish, season with lemon juice, dill,'
            ' salt, and pepper. Bake until fish flakes easily.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Vegetable Stir Fry with Tofu',
        directions=(
            'Stir-fry tofu cubes until golden, add assorted vegetables and a'
            ' stir-fry sauce. Serve over rice or noodles.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Classic Margherita Pizza',
        directions=(
            'Spread pizza dough with tomato sauce, top with slices of'
            ' mozzarella cheese and fresh basil leaves. Bake until crust is'
            ' golden.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Raspberry Almond Smoothie',
        directions=(
            'Blend together raspberries, almond milk, banana, and a scoop of'
            ' almond butter until smooth.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Moroccan Chickpea Stew',
        directions=(
            'Sauté onions, garlic, carrots, and spices. Add canned chickpeas,'
            ' diced tomatoes, and vegetable broth. Simmer until flavors meld.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Kale and Quinoa Salad',
        directions=(
            'Toss chopped kale, cooked quinoa, dried cranberries, sliced'
            ' almonds, and feta cheese with a lemon vinaigrette.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Grilled Cheese with Tomato and Basil',
        directions=(
            'Butter bread slices, layer with cheese, tomato slices, and basil.'
            ' Grill until bread is toasted and cheese is melted.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Sausage and Peppers Skillet',
        directions=(
            'Sauté sliced sausage, bell peppers, and onions until browned.'
            ' Serve with mustard or on a hoagie roll.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Lentil Soup',
        directions=(
            'Cook onions, carrots, celery, garlic, and lentils in vegetable'
            ' broth until lentils are tender. Season with thyme and bay leaves.'
        ),
    ),
    sqlite_schema_utils.Recipe(
        title='Garlic Butter Shrimp',
        directions=(
            'Sauté shrimp in butter and minced garlic until pink. Sprinkle with'
            ' parsley and serve with lemon wedges.'
        ),
    ),
]

_SERVINGS_OPTIONS = [
    '1 serving',
    '2 servings',
    '3-4 servings',
    '6 servings',
    '8 servings',
]
_PREP_TIME_OPTIONS = [
    '10 mins',
    '20 mins',
    '30 mins',
    '45 mins',
    '1 hrs',
    '2 hrs',
    '3 hrs',
    '4 hrs',
]

_COMMON_INGREDIENTS = [
    'tuna',
    'mayonnaise',
    'sriracha',
    'tortillas',
    'lettuce',
    'cucumber',
    'bread',
    'avocado',
    'eggs',
    'salt',
    'pepper',
    'chili flakes',
    'pita bread',
    'tomatoes',
    'feta cheese',
    'olives',
    'Greek dressing',
    'rice',
    'vegetables',
    'soy sauce',
    'pesto sauce',
    'peas',
    'Parmesan cheese',
    'chicken',
    'BBQ sauce',
    'cheese',
    'baguette',
    'basil',
    'garlic',
    'olive oil',
    'tilapia',
    'butter',
    'lemon juice',
    'turkey',
    'hummus',
    'bell peppers',
    'carrots',
    'mango',
    'curry powder',
    'coconut milk',
    'beef',
    'broccoli',
    'onions',
    'shrimp',
    'spinach',
    'herbs',
    'zucchini',
    'cauliflower',
    'sweet potato',
    'black beans',
    'cilantro',
    'Greek yogurt',
    'dill',
    'quinoa',
    'chickpeas',
    'romaine lettuce',
    'Caesar dressing',
    'Parmesan',
    'kidney beans',
    'corn',
    'chili seasoning',
    'asparagus',
    'mozzarella balls',
    'balsamic glaze',
    'fettuccine',
    'Alfredo sauce',
    'quinoa',
    'tomato sauce',
    'eggplant',
    'marinara sauce',
    'mozzarella cheese',
    'noodles',
    'Thai peanut sauce',
    'red bell peppers',
    'cabbage',
    'butternut squash',
    'nutmeg',
    'tofu',
    'pizza dough',
    'mozzarella cheese',
    'raspberries',
    'almond milk',
    'banana',
    'almond butter',
    'lentils',
    'thyme',
    'bay leaves',
    'parsley',
    'lemon wedges',
    # More exotic ingredients that are likely not in the existing recipes.
    'ghee',
    'cardamom',
    'fenugreek',
    'amchur (dry mango powder)',
    'rose water',
    'pomegranate molasses',
    'kaffir lime leaves',
    'galangal',
    'lemongrass',
    'furikake',
    'black garlic',
    'hemp seeds',
    'chia seeds',
    'açai berry',
    'maca powder',
    'spirulina',
    'cassava flour',
    'arrowroot powder',
    'seaweed',
    'escargot',
    'venison',
    'quail eggs',
    'duck fat',
    'morel mushrooms',
    'chanterelle mushrooms',
    'black truffle',
    'edible flowers',
    'salsify',
    'rutabaga',
    'celeriac',
    'finger limes',
]
