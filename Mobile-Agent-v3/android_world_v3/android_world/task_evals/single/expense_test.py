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
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import expense
from android_world.task_evals.utils import sqlite_schema_utils


class TestingExpenseDeleteMultipleForTest(expense._ExpenseDeleteMultiple):
  n_rows = 2
  n_rows_noise = 3


class ExpenseDeleteDuplicateExpenses2ForTest(expense.ExpenseDeleteDuplicates2):
  n_rows = 1
  n_rows_noise = 4


class ExpenseAddMultipleForTest(expense._ExpenseAddMultiple):
  n_rows = 2
  n_rows_noise = 3


class ExpenseDeleteMultipleTest(absltest.TestCase):

  @mock.patch.object(expense, "_generate_expense")
  def test_generate_params(self, mock_get_random_row):
    mock_get_random_row.side_effect = [
        # ROW_OBJECTS
        sqlite_schema_utils.Expense("expense_1", amount=10),
        sqlite_schema_utils.Expense("expense_1", amount=60),
        sqlite_schema_utils.Expense("expense_2", amount=40),
        # ROW_OBJECTS_NOISE
        sqlite_schema_utils.Expense("expense_1", amount=80),
        sqlite_schema_utils.Expense("expense_3", amount=20),
        sqlite_schema_utils.Expense("expense_4", amount=40),
        sqlite_schema_utils.Expense("expense_5", amount=40),
    ]

    self.params = TestingExpenseDeleteMultipleForTest.generate_random_params()

    self.assertEqual(
        self.params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_1", amount=10),
            sqlite_schema_utils.Expense("expense_2", amount=40),
        ],
    )
    self.assertEqual(
        self.params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_3", amount=20),
            sqlite_schema_utils.Expense("expense_4", amount=40),
            sqlite_schema_utils.Expense("expense_5", amount=40),
        ],
    )


class ExpenseDeleteDuplicateExpenses2Test(absltest.TestCase):

  @mock.patch.object(expense, "_generate_expense")
  @mock.patch.object(expense, "_get_random_timestamp")
  @mock.patch.object(random, "sample")
  def test_generate_params(
      self, mock_sample, mock_get_random_timestamp, mock_get_random_row
  ):
    mock_get_random_row.side_effect = [
        sqlite_schema_utils.Expense("expense_1", amount=10),
        sqlite_schema_utils.Expense("expense_target", amount=40),
    ]
    mock_get_random_timestamp.side_effect = [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    mock_sample.return_value = [52, 100, 101]

    self.params = (
        ExpenseDeleteDuplicateExpenses2ForTest.generate_random_params()
    )

    self.assertEqual(
        self.params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_target", amount=40),
            sqlite_schema_utils.Expense("expense_target", amount=40),
        ],
    )
    self.assertEqual(
        self.params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_1", amount=10),
            sqlite_schema_utils.Expense(
                "expense_target", amount=92, created_date=0, modified_date=0
            ),
            sqlite_schema_utils.Expense(
                "expense_target",
                amount=140,
                created_date=1000,
                modified_date=1000,
            ),
            sqlite_schema_utils.Expense(
                "expense_target",
                amount=141,
                created_date=2000,
                modified_date=2000,
            ),
        ],
    )


class ExpenseAddMultipleTest(absltest.TestCase):

  @mock.patch.object(expense, "_generate_expense")
  def test_generate_params(self, mock_get_random_row):
    mock_get_random_row.side_effect = [
        # ROW_OBJECTS
        sqlite_schema_utils.Expense("expense_1", amount=10),
        sqlite_schema_utils.Expense("expense_1", amount=10),
        sqlite_schema_utils.Expense("expense_2", amount=40),
        # ROW_OBJECTS_NOISE
        sqlite_schema_utils.Expense("expense_1", amount=10),
        sqlite_schema_utils.Expense("expense_3", amount=20),
        sqlite_schema_utils.Expense("expense_4", amount=40),
        sqlite_schema_utils.Expense("expense_5", amount=40),
    ]

    self.params = ExpenseAddMultipleForTest.generate_random_params()

    self.assertEqual(
        self.params[sqlite_validators.ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_1", amount=10),
            sqlite_schema_utils.Expense("expense_2", amount=40),
        ],
    )
    self.assertEqual(
        self.params[sqlite_validators.NOISE_ROW_OBJECTS],
        [
            sqlite_schema_utils.Expense("expense_3", amount=20),
            sqlite_schema_utils.Expense("expense_4", amount=40),
            sqlite_schema_utils.Expense("expense_5", amount=40),
        ],
    )


if __name__ == "__main__":
  absltest.main()
