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

"""Base class for task evaluations interacting with SQLite-based Android apps."""

import abc
import dataclasses
from typing import Any
from typing import Optional
from typing import Type
from absl import logging
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import fuzzy_match_lib


def verify_playlist(
    device_playlist_rows: list[sqlite_schema_utils.PlaylistInfo],
    candidate_playlist_name: str,
    candidate_files: list[str],
) -> bool:
  """Verifies if the playlist on the device matches the expected name, files, and their order.

  Args:
    device_playlist_rows: The playlist rows queried from the device.
    candidate_playlist_name: The expected name of the playlist.
    candidate_files: The list of expected media file names in the playlist.

  Returns:
    True if the actual playlist matches the expected criteria, False otherwise.
  """
  total = sum(
      1
      for actual_item in device_playlist_rows
      if fuzzy_match_lib.fuzzy_match(
          actual_item.playlist_name, candidate_playlist_name, ignore_case=True
      )
  )

  if total != len(candidate_files):
    return False

  matched_files = 0
  for index, expected_file in enumerate(candidate_files):
    if any(
        fuzzy_match_lib.fuzzy_match(
            actual_item.playlist_name, candidate_playlist_name, ignore_case=True
        )
        and actual_item.media_file_name == expected_file
        and (actual_item.order_in_playlist == index)
        for actual_item in device_playlist_rows
    ):
      matched_files += 1
    else:
      return False

  return matched_files == len(candidate_files)


def validate_rows_removal_integrity(
    before: list[sqlite_schema_utils.RowType],
    after: list[sqlite_schema_utils.RowType],
    ids: list[int],
    id_name: str,
) -> bool:
  """Validates that specified rows have been removed correctly from the rows list and that the remaining rows are unaltered.

  This function checks that all rows with IDs in `ids` are not present
  in the `after` state and that all other rows from the `before` state remain
  unchanged. It also ensures that no new rows have been inadvertently added.

  Args:
    before: State of the rows before removal.
    after: State of the rows after attempted removal.
    ids: IDs of the rows expected to be removed.
    id_name: The name of the ID column in the database.

  Returns:
    True if specified rows are removed and the integrity of the rows list is
    maintained; False if any specified rows are not removed, if any
    non-specified rows are missing, or if new rows have been added.
  """
  for row_id in ids:
    if not any(row for row in before if getattr(row, id_name) == row_id):
      raise ValueError(f"row ID {row_id} not present in before.")

  # Validate the removal and intactness of other rows
  for row in before:
    # If the row ID is in the list of removed row IDs
    if getattr(row, id_name) in ids:
      if row in after:
        return False
    elif row not in after:
      # Make sure we didn't remove other rows.
      return False

  # Check that no new unexpected rows have been added
  for row in after:
    if row not in before:
      return False

  return True


def validate_rows_addition_integrity(
    before: list[sqlite_schema_utils.RowType],
    after: list[sqlite_schema_utils.RowType],
    reference_rows: list[sqlite_schema_utils.RowType],
    compare_fields: list[str],
    free_form_fields: list[str] | None = None,
) -> bool:
  """Validates that specific rows have been added correctly without side effects.

  Checks that `reference_rows` are present in `after` and not in `before`, and
  that the rest of the rows in `before` remain unaltered in `after`. This
  validation ensures that no unrelated rows were added, removed, or changed in
  the process.

  Args:
    before: The state of the rows before the addition.
    after: The state of the rows after the attempted addition.
    reference_rows: A list of rows that are expected to be added.
    compare_fields: Which fields to use for comparison for each row.
    free_form_fields: Free-form, text fields where fuzzy matching will be used
      for comparison.

  Returns:
      bool: True if the rows were added correctly and other rows remained
      unaltered. False otherwise.
  """
  if not compare_fields:
    raise ValueError("compare_fields must not be empty.")
  if not free_form_fields:
    free_form_fields = []

  def db_row_matches_reference(
      reference_row: sqlite_schema_utils.RowType,
      row: sqlite_schema_utils.RowType,
  ) -> bool:
    for field in compare_fields:
      reference_value = getattr(reference_row, field)
      candidate_value = getattr(row, field)
      # Fuzzy match for text fields.
      if field in free_form_fields:
        if not fuzzy_match_lib.fuzzy_match(reference_value, candidate_value):
          return False
      else:
        if reference_value != candidate_value:
          return False
    return True

  # Check if the added rows are present in the 'after' state
  for reference_row in reference_rows:
    if not any(db_row_matches_reference(reference_row, row) for row in after):
      logging.warning(
          "Expected row %s not found in the 'after' state.", reference_row
      )
      return False

  if len(after) != len(before) + len(reference_rows):
    logging.warning(
        "The length of after %i is not equal to the length of before %i +"
        " length of added rows %i",
        len(after),
        len(before),
        len(reference_rows),
    )
    return False

  # Validate that no other rows were altered or removed during the addition
  for row in before:
    if row not in after:
      logging.warning(
          "row %s from 'before' state missing or altered in the 'after' state.",
          row,
      )
      return False

  return True


# Represents row objects to be added or deleted internally.
ROW_OBJECTS = "row_objects"
NOISE_ROW_OBJECTS = "noise_row_objects"


class SQLiteApp(task_eval.TaskEval, abc.ABC):
  """Base class for tasks interacting with SQLite-based Android apps."""

  app_name_with_db: str
  db_path: str
  db_key: str
  table_name: str
  row_type: Type[sqlite_schema_utils.SQLiteRow]

  def list_rows(
      self,
      env: interface.AsyncEnv,
      timeout_sec: Optional[float] = None,
  ) -> list[sqlite_schema_utils.RowType]:
    """Lists all rows from the specified table in the app's database using ADB.

    Args:
        env: The Android environment interface.
        timeout_sec: An optional timeout for the ADB operations.

    Returns:
        A list of row objects, each representing a row from the specified table
        in the database.
    """
    return sqlite_utils.get_rows_from_remote_device(
        self.table_name, self.db_path, self.row_type, env, timeout_sec
    )

  def add_rows(
      self,
      rows: list[sqlite_schema_utils.RowType],
      env: interface.AsyncEnv,
      timeout_sec: Optional[float] = None,
  ) -> None:
    sqlite_utils.insert_rows_to_remote_db(
        rows,
        self.db_key,
        self.table_name,
        self.db_path,
        self.app_name_with_db,
        env,
        timeout_sec,
    )

  def _clear_db(self, env: interface.AsyncEnv) -> None:
    """Clears the app's SQLite database."""
    sqlite_utils.delete_all_rows_from_table(
        self.table_name, self.db_path, env, self.app_name_with_db
    )
    try:
      self.list_rows(env)
    except ValueError as e:
      raise RuntimeError(
          "After clearing the old SQLite database, a new empty database was"
          " not created."
      ) from e

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initializes the task environment."""
    self._clear_db(env)  # In case the previous run crashed.
    super().initialize_task(env)
    self._clear_db(env)
    if NOISE_ROW_OBJECTS in self.params:
      self.add_rows(self.params[NOISE_ROW_OBJECTS], env)

  def tear_down(self, env: interface.AsyncEnv):
    """Cleans up after task completion."""
    super().tear_down(env)
    self._clear_db(env)


class AddMultipleRows(SQLiteApp, abc.ABC):
  """Abstract class for tasks that involve adding multiple rows to a SQLite database."""

  n_rows: int = -1  # Number of rows to be added, to be defined in subclasses.

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.before = []

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initial setup for the task, if necessary."""
    super().initialize_task(env)
    self.before = self.list_rows(env)

  @abc.abstractmethod
  def validate_addition_integrity(
      self,
      before: list[sqlite_schema_utils.RowType],
      after: list[sqlite_schema_utils.RowType],
      reference_rows: list[sqlite_schema_utils.RowType],
  ) -> bool:
    """Validates the integrity of the rows addition.

    Args:
      before: State of database before modification.
      after: Current state of the database.
      reference_rows: The rows that we are checking if are added and in the
        current state.

    Returns:
      Whether the reference rows were successfully added.
    """

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Determine if the row addition task was successful."""
    after = self.list_rows(env)
    row_addition_successful = self.validate_addition_integrity(
        self.before, after, self.params[ROW_OBJECTS]
    )
    return 1.0 if row_addition_successful else 0.0

  @classmethod
  @abc.abstractmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.RowType:
    """Generates a random row. To be implemented in subclasses."""

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    """Generate random parameters for new row addition tasks."""
    if cls.n_rows == -1:
      raise ValueError("n_rows must be defined in subclasses.")
    random_rows = [cls._get_random_target_row() for _ in range(cls.n_rows)]
    return {ROW_OBJECTS: random_rows}


class DeleteMultipleRows(SQLiteApp, abc.ABC):
  """Abstract class for tasks that involve deleting multiple rows from a SQLite database."""

  n_rows: int  # Number of rows to be deleted, to be defined in subclasses.
  n_rows_noise: int  # Number of additional rows to add not relevant to goal.

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self.rows_to_delete = []
    self.before = []

  def _validate_initial_state(
      self, before: list[sqlite_schema_utils.RowType]
  ) -> None:
    """Validates the initial state before the deletion process starts."""
    if len(before) != (self.n_rows + self.n_rows_noise):
      raise RuntimeError(
          "Initial state validation failed. The number of rows before deletion"
          f" does not match the expected count. Found {len(before)} in DB, but"
          f" expected {self.n_rows + self.n_rows_noise}."
      )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initial setup for the task, if necessary."""
    super().initialize_task(env)
    n_rows = 0
    if ROW_OBJECTS in self.params:
      self.add_rows(self.params[ROW_OBJECTS], env)
      n_rows = len(self.params[ROW_OBJECTS])
    self.before = self.list_rows(env)
    # Newly added rows are at the end.
    self.rows_to_delete = self.before[len(self.before) - n_rows :]
    self._validate_initial_state(self.before)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    """Determine if the row deletion task was successful."""
    super().is_successful(env)

    # Get the state of the database after the deletion attempt
    after = self.list_rows(env)

    # Validate the integrity of the deletion
    deletion_successful = self.validate_deletion_integrity(self.before, after)
    return 1.0 if deletion_successful else 0.0

  @abc.abstractmethod
  def validate_deletion_integrity(
      self,
      before: list[sqlite_schema_utils.RowType],
      after: list[sqlite_schema_utils.RowType],
  ):
    """Validates the integrity of the row deletion."""


class DeleteDuplicateRows(DeleteMultipleRows):
  """Abstract class for tasks that involve deleting duplicate rows from a SQLite database."""

  def _validate_candidates(
      self, candidates: list[sqlite_schema_utils.RowType]
  ) -> None:
    """Validates the initial state before the deletion process starts."""
    if len(candidates) % 2 != 0:
      raise ValueError(
          "Initial state validation failed. Must contain exactly two rows."
      )
    val1, val2 = candidates
    for field in dataclasses.fields(val1):
      if field.name == self.db_key:
        continue
      if getattr(val1, field.name) != getattr(val2, field.name):
        raise ValueError(
            "Initial state validation failed. Doesn't contain duplicate rows."
        )

  def _validate_initial_state(
      self, before: list[sqlite_schema_utils.RowType]
  ) -> None:
    """Validates the initial state before the deletion process starts."""
    if len(before) != (2 + self.n_rows_noise):
      raise ValueError(
          "Initial state validation failed. The number of rows before deletion"
          f" does not match the expected count. Found {len(before)} in DB, but"
          f" expected {2 + self.n_rows_noise}."
      )

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    """Initial setup for the task, if necessary."""
    super().initialize_task(env)
    self._validate_candidates(self.params[ROW_OBJECTS])
    self.duplicate_rows = self.rows_to_delete
