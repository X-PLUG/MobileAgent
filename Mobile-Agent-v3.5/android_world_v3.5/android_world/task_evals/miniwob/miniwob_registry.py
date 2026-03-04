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

"""MiniWoB registry; it dynamically creates tasks.

In MiniWoB, a task is defined via it's HTML file. For each MiniWoB task we
dynamically create a new task class.
"""

from android_world.task_evals.miniwob import miniwob_base

TASK_REGISTRY = {}

# Subset of tasks used for evaluation.
TASK_REGISTRY_SUBSET = {}

# The names of the HTML files, modulo the .html suffix, corresponding to the
# supported tasks that can be run. All tasks have been verified to be correct
# and achievable on an Android device by a human.
_NAMES = (
    ## Original MiniWoB++ tasks:
    # keep-sorted start
    'bisect-angle',
    'book-flight',
    'choose-date',
    'choose-date-easy',
    'choose-date-medium',
    'choose-list',
    'circle-center',
    'click-button',
    'click-button-sequence',
    'click-checkboxes',
    'click-checkboxes-large',
    'click-checkboxes-soft',
    'click-checkboxes-transfer',
    'click-collapsible',
    'click-collapsible-2',
    'click-color',
    'click-dialog',
    'click-dialog-2',
    'click-link',
    'click-menu-2',
    'click-option',
    'click-pie',
    'click-scroll-list',
    'click-shades',
    'click-shape',
    'click-tab',
    'click-tab-2',
    'click-tab-2-easy',
    'click-tab-2-hard',
    'click-tab-2-medium',
    'click-test',
    'click-test-2',
    'click-test-transfer',
    'click-widget',
    'copy-paste',
    'copy-paste-2',
    'count-shape',
    'count-sides',
    'drag-box',
    # Drag sometimes is scroll, but task easily done by human in 1-shot.
    'drag-item',
    'email-inbox',
    'email-inbox-delete',
    'email-inbox-forward',
    'email-inbox-forward-nl',
    'email-inbox-forward-nl-turk',
    'email-inbox-important',
    'email-inbox-nl-turk',
    'email-inbox-noscroll',
    'email-inbox-reply',
    'email-inbox-star-reply',
    'enter-date',
    'enter-password',
    'enter-text',
    'enter-text-2',
    'enter-text-dynamic',
    'enter-time',
    'find-midpoint',
    'find-word',
    'focus-text',
    'focus-text-2',
    'grid-coordinate',
    'guess-number',
    'highlight-text',
    'highlight-text-2',
    'identify-shape',
    'login-user',
    'login-user-popup',
    'multi-layouts',
    'multi-orderings',
    'navigate-tree',
    'read-table',
    'read-table-2',
    'resize-textarea',
    'right-angle',
    'scroll-text',
    'scroll-text-2',
    'search-engine',
    'simon-says',
    'simple-algebra',
    'simple-arithmetic',
    'social-media',
    'social-media-all',
    'social-media-some',
    'terminal',
    'text-transform',
    'tic-tac-toe',
    'unicode-test',
    'use-autocomplete',
    'use-colorwheel',
    'use-colorwheel-2',
    'use-slider',
    'visual-addition',
    # keep-sorted end
    ## Removed tasks kept for posterity:
    # These tasks require near-realtime movement and are unable to be achieved
    # by humans operating on emulators, thus we exclude them.
    # 'chase-circle',
    # Too hard to click in emulator.
    # 'moving-items',
    # These tasks break or are much harder due to HTML rendering on Android
    # webview. Comments for each task:
    # Technically possible, but unnecessarily harder due to touch interface.
    # Drags will scroll screen moving task out of view.
    # 'drag-cube',
    # 'drag-items-grid',  # Elements are not interactable on Android.
    # 'drag-items',  # Elements are not interactable on Android.
    # Technically possible, but unnecessarily harder due to touch interface.
    # Drags will scroll screen moving task out of view.
    # 'drag-shapes',
    # 'drag-sort-numbers',  # Elements are not interactable on Android.
    # 'text-editor',  # Cannot underline everything. Weird glitch.
    # 'number-checkboxes',  # Not correctly rendered: Only three columns.
    # Sliders don't work with Android
    # 'use-slider-2',  # Slider implementation not working.
    # 'use-spinner',  # Slider implementation not working.
    # The menu responsiveness breaks and task does not behave as intended.
    # 'click-menu'
)

# Subset of tasks used in Synapse paper (https://arxiv.org/pdf/2306.07863.pdf).
_NAMES_SUBSET = (
    'book-flight',
    'choose-date',
    'choose-list',
    'click-button',
    'click-button-sequence',
    'click-checkboxes',
    'click-checkboxes-large',
    'click-checkboxes-soft',
    'click-checkboxes-transfer',
    'click-collapsible',
    'click-collapsible-2',
    'click-color',
    'click-dialog',
    'click-dialog-2',
    'click-link',
    'click-menu',
    'click-option',
    'click-pie',
    'click-scroll-list',
    'click-shades',
    'click-shape',
    'click-tab',
    'click-tab-2',
    'click-tab-2-hard',
    'click-test',
    'click-test-2',
    'click-widget',
    'copy-paste',
    'copy-paste-2',
    'count-shape',
    'email-inbox',
    'email-inbox-forward-nl',
    'email-inbox-forward-nl-turk',
    'email-inbox-nl-turk',
    'enter-date',
    'enter-password',
    'enter-text',
    'enter-text-dynamic',
    'enter-time',
    'find-word',
    'focus-text',
    'focus-text-2',
    'grid-coordinate',
    'guess-number',
    'identify-shape',
    'login-user',
    'login-user-popup',
    'multi-layouts',
    'multi-orderings',
    'navigate-tree',
    'read-table',
    'search-engine',
    'simple-algebra',
    'simple-arithmetic',
    'social-media',
    'social-media-all',
    'social-media-some',
    'terminal',
    'text-transform',
    'tic-tac-toe',
    'unicode-test',
    'use-autocomplete',
    'use-slider',
    'use-spinner',
)


def _create_class_name(html_file_name: str) -> str:
  """Converts a hyphen-separated string to CamelCase with 'MiniWob' suffix.

  E.g., use-slider -> UseSliderMiniWob.

  Args:
      html_file_name: A string in hyphen-separated format.

  Returns:
      A CamelCase string with 'MiniWob' prefix.
  """
  parts = html_file_name.split('-')
  camel_case = ''.join(part.capitalize() for part in parts)
  return f'MiniWob{camel_case}'


def _build_task_class(task_name: str) -> miniwob_base.MiniWoBTask:
  """Dynamically builds and returns a new subclass of MiniWoBTask.

  Args:
    task_name: The name of the task for which class is to be created.

  Returns:
    A subclass of MiniWoBTask that is dynamically created.

  Example:
    >>> BookFlightMiniWob = _build_task_class("book-flight")
    >>> isinstance(BookFlightMiniWob, miniwob_base.MiniWoBTask)
    True
  """

  @classmethod
  def generate_random_params(cls):  # pylint:disable=unused-argument
    """Sets the task: i.e. which HTML file should be loaded."""
    return {'task_name': task_name}

  return type(
      _create_class_name(name),
      (miniwob_base.MiniWoBTask,),
      {'generate_random_params': generate_random_params},
  )


for name in _NAMES:
  task_class = _build_task_class(name)
  TASK_REGISTRY[name] = task_class
  if name in _NAMES_SUBSET:
    TASK_REGISTRY_SUBSET[name] = task_class
