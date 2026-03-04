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
from android_world.task_evals.single.calendar import events_generator
from android_world.utils import datetime_utils


class EventsGeneratorTest(parameterized.TestCase):

  def test_generate_event(self):
    """Test if generate_event produces a valid event."""
    event = events_generator.generate_event(
        datetime_utils.create_random_october_2023_unix_ts()
    )
    self.assertGreater(event.end_ts, event.start_ts)
    self.assertIsInstance(event.title, str)
    self.assertIsInstance(event.description, str)
    self.assertNotEmpty(event.title)
    self.assertNotEmpty(event.description)

  @parameterized.named_parameters(
      ('name_suffix', 'Meeting with Alice', 0),
      ('subject_suffix', 'Workshop on Budget Planning', 2),
  )
  @mock.patch.object(random, 'choice')
  def test_generate_event_title(
      self,
      expected_title,
      mock_idx,
      mock_random_choice,
  ):
    """Test if generate_event_title produces a valid title."""
    mock_random_choice.side_effect = lambda x: x[mock_idx]
    title = events_generator.generate_event_title()
    self.assertEqual(title, expected_title)

  @parameterized.named_parameters(
      ('name_suffix', 'We will discuss upcoming project milestones.', 0),
      (
          'subject_suffix',
          (
              'We will finalize marketing strategies. Remember to confirm'
              ' attendance.'
          ),
          1,
      ),
  )
  @mock.patch.object(random, 'choice')
  def test_generate_event_description(
      self, expected_description, mock_idx, mock_random_choice
  ):
    """Test if generate_event_description produces a valid description."""
    mock_random_choice.side_effect = lambda x: x[mock_idx]
    description = events_generator.generate_event_description()
    self.assertEqual(description, expected_description)


if __name__ == '__main__':
  absltest.main()
