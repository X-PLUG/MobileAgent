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

from absl.testing import absltest
from android_world.env import representation_utils
from android_world.task_evals.single import clock


class TestIsTimerSet(absltest.TestCase):

  def test_valid_elements_and_activity(self):
    valid_elements = [
        representation_utils.UIElement(
            text='01h 15m 30s', content_description=None
        ),
        representation_utils.UIElement(
            text=None, content_description='1 hour, 15 minutes, 30 seconds'
        ),
    ]
    self.assertTrue(
        clock._is_timer_set(
            valid_elements, 'DeskClock', hours=1, minutes=15, seconds=30
        ),
        'Expected True for valid elements and valid activity',
    )

  def test_invalid_elements(self):
    invalid_elements = [
        representation_utils.UIElement(
            text='00h 00m 00s', content_description=None
        ),
        representation_utils.UIElement(
            text=None, content_description='0 hours, 0 minutes, 0 seconds'
        ),
    ]
    self.assertFalse(
        clock._is_timer_set(
            invalid_elements, 'DeskClock', hours=1, minutes=15, seconds=30
        ),
        'Expected False for invalid elements',
    )

  def test_invalid_activity(self):
    valid_elements = [
        representation_utils.UIElement(
            text='01h 15m 30s', content_description=None
        ),
        representation_utils.UIElement(
            text=None, content_description='1 hours, 15 minutes, 30 seconds'
        ),
    ]
    self.assertFalse(
        clock._is_timer_set(
            valid_elements, 'SomeOtherActivity', hours=1, minutes=15, seconds=30
        ),
        'Expected False for invalid activity',
    )

  def test_valid_content_description(self):
    elements = [
        representation_utils.UIElement(
            text=None, content_description='1 hours, 15 minutes, 30 seconds'
        ),
    ]
    self.assertTrue(
        clock._is_timer_set(
            elements, 'DeskClock', hours=1, minutes=15, seconds=30
        ),
        'Expected True for valid content description',
    )

  def test_valid_text(self):
    elements = [
        representation_utils.UIElement(
            text='01h 15m 30s', content_description=None
        ),
    ]
    self.assertTrue(
        clock._is_timer_set(
            elements, 'DeskClock', hours=1, minutes=15, seconds=30
        ),
        'Expected True for valid text',
    )


class TestIsStopwatchPaused(absltest.TestCase):

  def test_valid_elements(self):
    valid_elements = [
        representation_utils.UIElement(content_description='Start'),
        representation_utils.UIElement(
            text=None, content_description='Stopwatch'
        ),
        representation_utils.UIElement(text='Stopwatch'),
    ]
    self.assertTrue(
        clock._is_stopwatch_paused(valid_elements, 'DeskClock'),
        'Expected True for valid elements and valid activity',
    )

  def test_invalid_elements(self):
    invalid_elements = [
        representation_utils.UIElement(
            text=None, content_description='Stopwatch'
        ),
        representation_utils.UIElement(text=None, content_description='Lap'),
    ]
    self.assertFalse(
        clock._is_stopwatch_paused(invalid_elements, 'DeskClock'),
        'Expected False for invalid elements',
    )

  def test_invalid_activity(self):
    valid_elements = [
        representation_utils.UIElement(content_description='Start'),
        representation_utils.UIElement(
            text=None, content_description='Stopwatch'
        ),
        representation_utils.UIElement(text='Stopwatch'),
    ]
    self.assertFalse(
        clock._is_stopwatch_paused(valid_elements, 'SomeOtherActivity'),
        'Expected False for invalid activity',
    )


class TestIsStopwatchRunning(absltest.TestCase):

  def test_valid_elements(self):
    valid_elements = [
        representation_utils.UIElement(text=None, content_description='Pause'),
        representation_utils.UIElement(text=None, content_description='Lap'),
    ]
    self.assertTrue(
        clock._is_stopwatch_running(valid_elements, 'DeskClock'),
        'Expected True for valid elements and valid activity',
    )

  def test_invalid_elements(self):
    invalid_elements = [
        representation_utils.UIElement(text=None, content_description='Reset'),
        representation_utils.UIElement(text=None, content_description='Start'),
    ]
    self.assertFalse(
        clock._is_stopwatch_running(invalid_elements, 'DeskClock'),
        'Expected False for invalid elements',
    )

  def test_invalid_activity(self):
    valid_elements = [
        representation_utils.UIElement(text=None, content_description='Pause'),
        representation_utils.UIElement(text=None, content_description='Lap'),
    ]
    self.assertFalse(
        clock._is_stopwatch_running(valid_elements, 'SomeOtherActivity'),
        'Expected False for invalid activity',
    )


if __name__ == '__main__':
  absltest.main()
