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

"""Tasks for managing expenses in an expense app."""

import abc
import dataclasses
import random
from typing import Any, Optional
from android_world.env import device_constants
from android_world.env import interface
from android_world.env.setup_device import apps
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import datetime_utils
from android_world.utils import file_utils

_DB_PATH = '/data/data/com.arduia.expense/databases/accounting.db'
_TABLE_NAME = 'expense'
_APP_NAME = 'pro expense'
_DB_KEY = 'expense_id'

# How to represent recipes in text form.
_TEXT_REPRESENTATION_TYPE = 'text_representation_type'


def _get_random_timestamp() -> int:
  """Gets a timestep in the current month, up to the current day (Oct 15)."""
  return datetime_utils.create_random_october_2023_unix_ts(
      start_day=1, end_day=15
  )


class _Expense(task_eval.TaskEval, abc.ABC):
  """Base class for expense logic task evals."""

  # From TaskEval.
  schema = {}
  app_names = (_APP_NAME,)
  template = ''  # Unused, since we directly build goal in implementations.

  # From sqlite_base.SQLiteApp
  app_name_with_db = _APP_NAME
  db_key = _DB_KEY
  db_path = _DB_PATH
  table_name = _TABLE_NAME
  row_type = sqlite_schema_utils.Expense

  def initialize_task(self, env: interface.AsyncEnv):
    if not sqlite_utils.table_exists(self.table_name, self.db_path, env):
      apps.ExpenseApp.setup(env)
    super().initialize_task(env)


class _ExpenseDeleteMultiple(_Expense, sqlite_validators.DeleteMultipleRows):
  """Task to delete multiple expenses in an expense tracking app."""

  complexity = 2
  n_rows = 3  # Default number of expenses to delete
  n_rows_noise = 0  # Default noise rows

  @property
  def goal(self) -> str:
    targets = self.params[sqlite_validators.ROW_OBJECTS]
    expense_names = [expense.name for expense in targets]
    expense_names_str = ', '.join(expense_names)
    return (
        f'Delete the following expenses from {_APP_NAME}: {expense_names_str}.'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    return sqlite_validators.validate_rows_removal_integrity(
        before,
        after,
        [expense.expense_id for expense in self.rows_to_delete],
        self.db_key,
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove expense task."""

    expenses = []
    while len(expenses) < cls.n_rows + cls.n_rows_noise:
      candidate = _generate_expense()
      if not any([candidate.name == expense.name for expense in expenses]):
        expenses.append(candidate)

    if cls.n_rows_noise > 0:
      target_rows = expenses[: cls.n_rows]
      noise_rows = expenses[cls.n_rows :]
      return {
          sqlite_validators.ROW_OBJECTS: target_rows,
          sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
      }
    else:
      return {
          sqlite_validators.ROW_OBJECTS: expenses,
      }


class ExpenseDeleteSingle(_ExpenseDeleteMultiple):
  """Task to delete a single expense in an expense tracking app."""

  complexity = 1
  n_rows = 1
  n_rows_noise = 0


class ExpenseDeleteMultiple(_ExpenseDeleteMultiple):
  """Task to delete multiple expenses in an expense tracking app."""

  complexity = 2
  n_rows = 3
  n_rows_noise = 0


class ExpenseDeleteMultiple2(_ExpenseDeleteMultiple):
  """Harder task to delete multiple expenses in an expense tracking app."""

  complexity = 3.4
  n_rows = 3
  n_rows_noise = 50


class _ExpenseDeleteDuplicates(_Expense, sqlite_validators.DeleteDuplicateRows):
  """Deduplicate expenses in the expense tracking app with some noise."""

  complexity = 1.2
  n_rows = 1  # Number of unique expenses to duplicate for the task
  n_rows_noise = 5  # Number of additional unique expenses to include as noise

  @property
  def goal(self) -> str:
    return (
        f'Delete all but one of any expenses in {_APP_NAME} that are exact'
        ' duplicates, ensuring at least one instance of each unique expense'
        ' remains.'
    )

  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense deletion."""
    target1, target2 = self.rows_to_delete
    return sqlite_validators.validate_rows_removal_integrity(
        before, after, [target1.expense_id], self.db_key
    ) or sqlite_validators.validate_rows_removal_integrity(
        before, after, [target2.expense_id], self.db_key
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove duplicate expense task."""
    rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise + cls.n_rows,
        _generate_expense,
        replacement=False,
    )
    target = rows.pop()
    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: rows,
    }


class ExpenseDeleteDuplicates(_ExpenseDeleteDuplicates):
  """Deduplicate expenses in the expense tracking app with some noise."""

  n_rows = 1
  n_rows_noise = 5


class ExpenseDeleteDuplicates2(_ExpenseDeleteDuplicates):
  """Harder task to deduplicate expenses in the expense tracking app."""

  n_rows = 1
  n_rows_noise = 40
  complexity = 1.8

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for a remove duplicate expense task."""
    assert cls.n_rows == 1
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise + cls.n_rows - 3,
        _generate_expense,
        replacement=False,
    )
    target = noise.pop()
    perturbations = random.sample(range(50, 1000), 3)
    target_varations = []
    for perturbation in perturbations:
      target_varations.append(
          dataclasses.replace(
              target,
              amount=target.amount + perturbation,
              created_date=_get_random_timestamp() * 1000,
              modified_date=_get_random_timestamp() * 1000,
          )
      )

    return {
        sqlite_validators.ROW_OBJECTS: [target, target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise + target_varations,
    }


def _get_expense_rows_as_text(
    rows: list[sqlite_schema_utils.Expense],
    format_type: str,
    wrap_width: int | None = None,
) -> str:
  return sqlite_schema_utils.get_text_representation_of_rows(
      rows,
      [
          'name',
          'amount_dollars',
          'category_name',
          'note',
      ],
      format_type,
      'name',
      wrap_width=wrap_width,
  )


class _ExpenseAddMultiple(_Expense, sqlite_validators.AddMultipleRows):
  """Task to add multiple expenses in the Expense Tracking App."""

  complexity = 3
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    text_repr = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return f'Add the following expenses into the {_APP_NAME}:\n{text_repr}'

  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.Expense],
      after: list[sqlite_schema_utils.Expense],
      reference_rows: list[sqlite_schema_utils.Expense],
  ) -> bool:
    """Validates the integrity of the expense addition."""
    return sqlite_validators.validate_rows_addition_integrity(
        before,
        after,
        reference_rows,
        compare_fields=[
            'name',
            'amount',
            'category',
            'note',
        ],
        free_form_fields=[
            'name',
            'note',
        ],
    )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    return _generate_expense()

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for an add expense task."""
    target_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        cls._get_random_target_row,
        replacement=False,
    )
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        cls._get_random_target_row,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: random.choice(['csv', 'text_block']),
    }


class ExpenseAddSingle(_ExpenseAddMultiple):
  """Task to add a single expense in the Expense Tracking App."""

  complexity = 1.2
  n_rows = 1
  n_rows_noise = 10


class ExpenseAddMultiple(_ExpenseAddMultiple):
  """Task to add multiple expenses in the Expense Tracking App."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10


class ExpenseAddMultipleFromMarkor(_ExpenseAddMultiple):
  """Task to add multiple expenses from Markor into the Expense Tracking app."""

  complexity = 6
  n_rows = 2
  n_rows_noise = 100

  @property
  def goal(self) -> str:
    return (
        'Go through the transactions in my_expenses.txt in Markor. Log the '
        f'reimbursable transactions in the {_APP_NAME}.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    targets = [
        dataclasses.replace(row, note=row.note + '. ' + 'Reimbursable.')
        for row in self.params[sqlite_validators.ROW_OBJECTS]
    ]
    rows = targets + self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    random.shuffle(rows)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
    user_data_generation.write_to_markor(
        _get_expense_rows_as_text(rows, 'csv'),
        'my_expenses.txt',
        env,
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)


class ExpenseAddMultipleFromGallery(_ExpenseAddMultiple):
  """Task to add multiple expenses from Gallery into Expense Tracking app."""

  complexity = 6
  n_rows = 3
  n_rows_noise = 10

  app_names = (_APP_NAME, 'simple gallery pro')

  @property
  def goal(self) -> str:
    return (
        'Add the expenses from expenses.jpg in Simple Gallery Pro to '
        f'{_APP_NAME}.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    data = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS], 'text_block', wrap_width=60
    )
    user_data_generation.write_to_gallery(data, 'expenses.jpg', env)
    for i in range(10):
      data = _get_expense_rows_as_text(
          self.params[sqlite_validators.NOISE_ROW_OBJECTS],
          'text_block',
          wrap_width=60,
      )
      user_data_generation.write_to_gallery(data, f'old_expenses_{i}.jpg', env)

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)


#### Generate expense data for tasks. ##########################################


def _generate_expense(
    expense_unix_time_s: Optional[int] = None,
    category_id: int | None = None,
) -> sqlite_schema_utils.Expense:
  """Generates a realistic expense entry.

  Args:
    expense_unix_time_s: The time the expense is entered into the app. This will
      be reflected in the UI.
    category_id: Optional value to override random generation.

  Returns:
      An Expense object with random realistic parameters.
  """
  if expense_unix_time_s is None:
    expense_unix_time_s = _get_random_timestamp()
  expense_unix_time_ms = expense_unix_time_s * 1000

  if category_id is None:
    category_id = random.choice(
        list(sqlite_schema_utils.Expense.category_id_to_name.keys())
    )
  name = random.choice(
      _EXPENSE_NAMES[
          sqlite_schema_utils.Expense.category_id_to_name[category_id]
      ]
  )
  amount = random.randint(
      1000, 50000
  )  # Amount in cents (e.g., $10.00 - $500.00)
  note = random.choice(_NOTES)
  return sqlite_schema_utils.Expense(
      name,
      amount,
      category_id,
      note,
      expense_unix_time_ms,
      expense_unix_time_ms,
  )


_EXPENSE_NAMES = {
    'Others': [
        'Emergency Repairs',
        'Pet Supplies',
        'Pet Care',
        'Household Items',
        'Stationery',
        'Unexpected Expenses',
        'Miscellaneous Gifts',
        'Subscriptions',
        'Membership Fees',
        'Legal Fees',
    ],
    'Income': [
        'Salary',
        'Freelance Payment',
        'Bonus',
        'Dividends',
        'Interest Income',
        'Rental Income',
        'Capital Gains',
        'Reimbursements',
        'Side Business',
        'Consulting Fees',
    ],
    'Food': [
        'Restaurant Meal',
        'Groceries',
        'Coffee',
        'Fast Food',
        'Fine Dining',
        'Bakery Items',
        'Snacks',
        'Food Delivery',
        'Specialty Foods',
        'Dining Out',
    ],
    'Housing': [
        'Rent Payment',
        'Mortgage',
        'Home Repairs',
        'Utilities',
        'Property Taxes',
        'Home Insurance',
        'Furnishing',
        'Cleaning Services',
        'Landscaping',
        'Pest Control',
    ],
    'Social': [
        'Dinner Party',
        'Gift for Friend',
        'Club Membership',
        'Wedding Gift',
        'Charity Donations',
        'Birthday Present',
        'Social Club Dues',
        'Event Tickets',
        'Night Out',
        'Party Supplies',
    ],
    'Entertainment': [
        'Concert Tickets',
        'Movie Night',
        'Theater Show',
        'Streaming Services',
        'Video Games',
        'Books',
        'Magazines',
        'Hobbies',
        'Museum Tickets',
        'Amusement Park',
    ],
    'Transportation': [
        'Taxi Fare',
        'Public Transit Pass',
        'Gas',
        'Parking Fees',
        'Car Maintenance',
        'Bike Repairs',
        'Car Insurance',
        'Public Transit',
        'Flight Tickets',
        'Ride-Sharing',
    ],
    'Clothes': [
        'New Jacket',
        'Shirt Purchase',
        'Shoes',
        'Dress',
        'Jeans',
        'Accessories',
        'Sportswear',
        'Undergarments',
        'Tailoring Services',
        'Laundry',
    ],
    'Health Care': [
        'Doctor Visits',
        'Medications',
        'Health Insurance',
        'Dental Care',
        'Eyecare',
        'Wellness Products',
        'Gym Membership',
        'Therapy Sessions',
        'Medical Tests',
    ],
    'Education': [
        'Tuition Fees',
        'School Supplies',
        'Textbooks',
        'Online Courses',
        'Seminars',
        'Workshops',
        'Educational Software',
        'Library Fees',
        'ProDev',
        'Tutoring Services',
    ],
    'Donation': [
        'Charity',
        'Fundraising Events',
        'Sponsorships',
        'Non-Profit Support',
        'Crowdfunding',
        'Religious',
        'Political',
        'Educational',
        'Medical Research',
        'Environmental',
    ],
}

_NOTES = [
    'Paid by card',
    'Urgent',
    'Monthly recurring',
    'Want to have',
    'A need',
    'Remember to transfer funds',
    'I may repeat this',
]
