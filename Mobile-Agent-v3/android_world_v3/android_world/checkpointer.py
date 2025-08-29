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

"""Checkpointer class."""

import abc
import datetime
import gzip
import io
import os
import pickle
from typing import Any

from absl import logging

INSTANCE_SEPARATOR = '_'

Episode = dict[str, Any]


def sort_key(filename: str) -> tuple[str, int|str]:
  """Returns the sort key for a filenames.

  Attempts to sort by the filename and then by the instance number. However,
  if the instance number is not an integer, it will be sorted as a string.

  Args:
    filename: The filename to sort.
  """
  parts = filename.split(INSTANCE_SEPARATOR, maxsplit=1)
  if len(parts) == 1:
    # When there is no instance number, sort by filename.
    return (parts[0], 0)
  name, num = parts
  try:
    num = int(num)
  except ValueError:
    num = 0
  return (name, num)


def _gzip_pickle(data: Any) -> bytes:
  """Pickle and gzip compress an object in memory.

  Args:
      data: The data to be pickled and gzipped.

  Returns:
      A bytes object containing the gzipped pickled data.
  """
  pickled_data = io.BytesIO()
  pickle.dump(data, pickled_data)

  pickled_data.seek(0)  # Reset the stream position to the beginning
  compressed_data = io.BytesIO()
  with gzip.GzipFile(
      fileobj=compressed_data, mode='wb', compresslevel=5
  ) as f_out:
    f_out.write(pickled_data.getvalue())

  return compressed_data.getvalue()


def _unzip_and_read_pickle(file_path: str) -> Any:
  """Reads a gzipped pickle file using 'with open', unzips, and unpickles it.

  Args:
      file_path: The path to the gzipped pickle file.

  Returns:
      The original Python object that was pickled and gzipped.
  """
  with open(file_path, 'rb') as f:
    compressed = f.read()

  with gzip.open(io.BytesIO(compressed), 'rb') as f_in:
    return pickle.load(f_in)


class Checkpointer(abc.ABC):
  """Saves and loads the results of an evaluation run."""

  @abc.abstractmethod
  def save_episodes(self, task_episodes: list[Episode], task_name: str) -> None:
    """Saves a task's episodes to disk."""

  @abc.abstractmethod
  def load(self, fields: list[str] | None = None) -> list[Episode]:
    """Loads all episodes from disk."""


class IncrementalCheckpointer(Checkpointer):
  """Saves and loads the results of an evaluation run.

  Designed for incremental saving of episodes for each task, enabling the
  checkpointer to save the results of an evaluation run task by task, rather
  than saving the entire dataset at once.

  Attributes:
      directory: The directory to store the task data.
  """

  def __init__(self, directory: str) -> None:
    self.directory = directory
    os.makedirs(directory, exist_ok=True)

  def save_episodes(self, task_episodes: list[Episode], task_name: str):
    """Saves a task group to disk.

    Args:
        task_episodes: The task's episodes to save.
        task_name: The unique identifier for the task group.
    """
    filename = os.path.join(self.directory, f'{task_name}.pkl.gz')
    with open(filename, 'wb') as f:
      compressed = _gzip_pickle(task_episodes)
      f.write(compressed)
    logging.info('Wrote task episodes for %s to %s', task_name, filename)

  def load(self, fields: list[str] | None = None) -> list[Episode]:
    """Loads all task groups from disk."""
    # Keep same order as runtime.
    directories = os.listdir(self.directory)
    directories.sort(key=sort_key)

    data = []
    for filename in directories:
      if filename.endswith('.pkl.gz'):
        try:
          task_group_id = filename[:-7]  # Remove ".pkl.gz" extension
          task_group = self._load_task_group(task_group_id)
          if fields is not None:
            task_group = [
                {field: episode[field] for field in fields}
                for episode in task_group
            ]
          data.extend(task_group)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.info('Unable to load %s with exception: %s', filename, e)
    return data

  def _load_task_group(self, task_group_id: str) -> list[Episode]:
    """Loads a single task group from disk."""
    filename = os.path.join(self.directory, f'{task_group_id}.pkl.gz')
    try:
      return _unzip_and_read_pickle(filename)
    except FileNotFoundError:
      logging.info(
          'File not readable: %s. It may not exist. Starting from empty state.',
          filename,
      )
      return []


class NullCheckpointer(Checkpointer):
  """Checkpointer that does nothing."""

  def __init__(self) -> None:
    """Constructor."""

  def save_episodes(self, task_episodes: list[Episode], task_name: str):
    pass

  def load(self, fields: list[str] | None = None) -> list[Episode]:
    del fields
    return []


def create_run_directory(location: str) -> str:
  """Creates the UUID directory name to save run results.

  Args:
    location: Location to write the directory.

  Returns:
    A UUID directory name.
  """
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
  return os.path.join(location, f'run_{timestamp}')


class DeprecatedCheckpointer:
  """Saves and loads the results of an evaluation run.

  Attributes:
    filename: The name of the file to write.
  """

  def __init__(self, filename: str) -> None:
    self.filename = filename

  def save(self, data: list[Episode], completed_tasks: list[str]) -> None:
    """Saves the results of an evaluation run.

    Args:
      data: The data for the run.
      completed_tasks: Metadata containing the tuple of completed tasks.
    """
    with open(self.filename, 'wb') as f:
      compressed = _gzip_pickle((data, completed_tasks))
      f.write(compressed)
    logging.info('Wrote to %s', self.filename)

  def load(
      self, fields: list[str] | None = None
  ) -> tuple[list[Episode], list[str]]:
    """Loads the results of an evaluation run."""
    del fields
    try:
      return _unzip_and_read_pickle(self.filename)
    except FileNotFoundError:
      logging.info(
          'File not readable: %s. It may not exist. Starting from empty state.',
          self.filename,
      )
      return [], []
