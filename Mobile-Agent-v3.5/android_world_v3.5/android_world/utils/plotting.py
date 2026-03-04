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

"""Plots UI elements."""

import copy
from typing import Any
from android_world.env import interface
from android_world.env import representation_utils
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np


def _get_element_text(
    element: representation_utils.UIElement, include_extra_info: bool = True
) -> str:
  """Returns a text describing the element.

  Args:
    element: The UI element to get text from.
    include_extra_info: Whether to consider extra information in deriving text,
      such as resource_id and class_name.

  Returns:
    The text description if found, '' otherwise.
  """
  if element.text:
    text = element.text
    # Special handling of single character labels.
    if (
        len(text) == 1
        and element.content_description
        and len(element.content_description) > 1
    ):
      text = element.content_description
    return text
  if element.hint_text:
    return element.hint_text
  if element.content_description:
    return element.content_description
  if element.tooltip:
    return element.tooltip
  if include_extra_info:
    if element.class_name is not None and element.class_name.endswith('Switch'):
      return 'Switch:' + ('on' if element.is_checked else 'off')
    if element.resource_id is not None:
      return element.resource_id.split('/')[-1]
    if element.class_name is not None and element.class_name.endswith(
        'EditText'
    ):
      return 'edit text'
  return ''


def _plot_element_cartoon(
    elements: list[representation_utils.UIElement],
    screenshot: np.ndarray | None = None,
    max_text_length: int = 30,
) -> tuple[plt.Axes, plt.Axes]:
  """Plots UI element bboxes and associated properties.

  It's useful for quick and dirty visualization of a screen purely using the
  UIElements.

  Args:
    elements: The elements to plot.
    screenshot: If provided: the screenshot.
    max_text_length: Maximum length of text to show.

  Returns:
    Plot of screen.
  """
  _, axs = plt.subplots(1, 2, figsize=(13, 12))
  if screenshot is not None:
    axs[1].imshow(screenshot)
  ax = axs[0]

  for element in elements:
    bbox = copy.deepcopy(element.bbox_pixels)
    if bbox is None:
      continue

    if screenshot is not None:
      # Normalize.
      bbox.x_min /= screenshot.shape[1]
      bbox.x_max /= screenshot.shape[1]
      bbox.y_min /= screenshot.shape[0]
      bbox.y_max /= screenshot.shape[0]
    else:
      # Pick reasonable values.
      bbox.x_min /= 1080
      bbox.x_max /= 1080
      bbox.y_min /= 2400
      bbox.y_max /= 2400

    if bbox:
      width = bbox.x_max - bbox.x_min
      height = bbox.y_max - bbox.y_min

      rect = patches.Rectangle(
          (bbox.x_min, bbox.y_min),
          width,
          height,
          linewidth=1,
          edgecolor='r',
          facecolor='none',
      )
      ax.add_patch(rect)

      # Check if text is present, else plot the icon_net_type.
      text = _get_element_text(element)
      if text:
        if len(text) > max_text_length:
          text = text[0:max_text_length] + '...'
        ax.text(
            bbox.x_min,
            bbox.y_min,
            text,
            fontsize=10,
            ha='left',
            va='bottom',
        )

  ax.invert_yaxis()
  return axs


def plot_ui_elements(
    state: interface.State,
    max_text_length: int = 30,
) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
  """Plots UI elements and optionally screenshot and action.

  Args:
    state: State of the environment.
    max_text_length: Maximum length of text to show.

  Returns:
    The axes from the plot.
  """
  axs = _plot_element_cartoon(
      state.ui_elements,
      screenshot=state.pixels,
      max_text_length=max_text_length,
  )
  return axs


def _plot_episode(
    screens: list[np.ndarray],
    title: str,
) -> None:
  """Plots an episode in a grid format.

  Args:
    screens: List of screen images.
    title: The title.
  """
  num_screens = len(screens)
  num_columns = 6
  num_rows = (num_screens + num_columns - 1) // num_columns
  fig, axs = plt.subplots(
      num_rows, num_columns, figsize=(4 * num_columns, num_rows * 3)
  )
  axs = np.atleast_2d(axs)
  fig.suptitle(title)

  i = 0
  for i, screen in enumerate(screens):
    row = i // num_columns
    col = i % num_columns
    ax = axs[row, col]
    ax.imshow(screen, aspect='auto')  # Use 'auto' to avoid distortion
    ax.axis('off')  # Turn off axes

  for j in range(i + 1, num_rows * num_columns):
    row = j // num_columns
    col = j % num_columns
    axs[row, col].axis('off')

  plt.tight_layout()
  plt.show()


def plot_episode(episode: dict[str, Any]) -> None:
  """Plots an episode in a grid format."""
  goal = episode['goal']
  episode_data = episode['episode_data']
  screens = (
      episode_data.get('screenshot')
      or episode_data.get('raw_screenshot')
      or episode_data.get('before_screenshot')
  )
  assert screens is not None
  _plot_episode(screens, title=goal)
