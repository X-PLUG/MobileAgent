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

"""Tools for processing and representing accessibility trees."""

import dataclasses
from typing import Any, Optional
import xml.etree.ElementTree as ET
from android_env.proto.a11y import android_accessibility_forest_pb2


@dataclasses.dataclass
class BoundingBox:
  """Class for representing a bounding box."""

  x_min: float | int
  x_max: float | int
  y_min: float | int
  y_max: float | int

  @property
  def center(self) -> tuple[float, float]:
    """Gets center of bounding box."""
    return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

  @property
  def width(self) -> float | int:
    """Gets width of bounding box."""
    return self.x_max - self.x_min

  @property
  def height(self) -> float | int:
    """Gets height of bounding box."""
    return self.y_max - self.y_min

  @property
  def area(self) -> float | int:
    return self.width * self.height


@dataclasses.dataclass
class UIElement:
  """Represents a UI element."""

  text: Optional[str] = None
  content_description: Optional[str] = None
  class_name: Optional[str] = None
  bbox: Optional[BoundingBox] = None
  bbox_pixels: Optional[BoundingBox] = None
  hint_text: Optional[str] = None
  is_checked: Optional[bool] = None
  is_checkable: Optional[bool] = None
  is_clickable: Optional[bool] = None
  is_editable: Optional[bool] = None
  is_enabled: Optional[bool] = None
  is_focused: Optional[bool] = None
  is_focusable: Optional[bool] = None
  is_long_clickable: Optional[bool] = None
  is_scrollable: Optional[bool] = None
  is_selected: Optional[bool] = None
  is_visible: Optional[bool] = None
  package_name: Optional[str] = None
  resource_name: Optional[str] = None
  tooltip: Optional[str] = None
  resource_id: Optional[str] = None
  metadata: Optional[dict[str, Any]] = None


def accessibility_node_to_ui_element(
    node: Any,
    screen_size: Optional[tuple[int, int]] = None,
) -> UIElement:
  """Converts a node from an accessibility tree to a UIElement."""

  def text_or_none(text: Optional[str]) -> Optional[str]:
    """Returns None if text is None or 0 length."""
    return text if text else None

  node_bbox = node.bounds_in_screen
  bbox_pixels = BoundingBox(
      node_bbox.left, node_bbox.right, node_bbox.top, node_bbox.bottom
  )

  if screen_size is not None:
    bbox_normalized = _normalize_bounding_box(bbox_pixels, screen_size)
  else:
    bbox_normalized = None

  return UIElement(
      text=text_or_none(node.text),
      content_description=text_or_none(node.content_description),
      class_name=text_or_none(node.class_name),
      bbox=bbox_normalized,
      bbox_pixels=bbox_pixels,
      hint_text=text_or_none(node.hint_text),
      is_checked=node.is_checked,
      is_checkable=node.is_checkable,
      is_clickable=node.is_clickable,
      is_editable=node.is_editable,
      is_enabled=node.is_enabled,
      is_focused=node.is_focused,
      is_focusable=node.is_focusable,
      is_long_clickable=node.is_long_clickable,
      is_scrollable=node.is_scrollable,
      is_selected=node.is_selected,
      is_visible=node.is_visible_to_user,
      package_name=text_or_none(node.package_name),
      resource_name=text_or_none(node.view_id_resource_name),
  )


def _normalize_bounding_box(
    node_bbox: BoundingBox,
    screen_width_height_px: tuple[int, int],
) -> BoundingBox:
  width, height = screen_width_height_px
  return BoundingBox(
      node_bbox.x_min / width,
      node_bbox.x_max / width,
      node_bbox.y_min / height,
      node_bbox.y_max / height,
  )


def forest_to_ui_elements(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = False,
    screen_size: Optional[tuple[int, int]] = None,
) -> list[UIElement]:
  """Extracts nodes from accessibility forest and converts to UI elements.

  We extract all nodes that are either leaf nodes or have content descriptions
  or is scrollable.

  Args:
    forest: The forest to extract leaf nodes from.
    exclude_invisible_elements: True if invisible elements should not be
      returned.
    screen_size: The size of the device screen in pixels (width, height).

  Returns:
    The extracted UI elements.
  """
  elements = []
  for window in forest.windows:
    for node in window.tree.nodes:
      if not node.child_ids or node.content_description or node.is_scrollable:
        if exclude_invisible_elements and not node.is_visible_to_user:
          continue
        else:
          elements.append(accessibility_node_to_ui_element(node, screen_size))
  return elements


def _parse_ui_hierarchy(xml_string: str) -> dict[str, Any]:
  """Parses the UI hierarchy XML into a dictionary structure."""
  root = ET.fromstring(xml_string)

  def parse_node(node):
    result = node.attrib
    result['children'] = [parse_node(child) for child in node]
    return result

  return parse_node(root)


def xml_dump_to_ui_elements(xml_string: str) -> list[UIElement]:
  """Converts a UI hierarchy XML dump from uiautomator dump to UIElements."""
  parsed_hierarchy = _parse_ui_hierarchy(xml_string)
  ui_elements = []

  def process_node(node, is_root):
    bounds = node.get('bounds')
    if bounds:
      x_min, y_min, x_max, y_max = map(
          int, bounds.strip('[]').replace('][', ',').split(',')
      )
      bbox = BoundingBox(x_min, x_max, y_min, y_max)
    else:
      bbox = None

    ui_element = UIElement(
        text=node.get('text'),
        content_description=node.get('content-desc'),
        class_name=node.get('class'),
        bbox=bbox,
        bbox_pixels=bbox,
        is_checked=node.get('checked') == 'true',
        is_checkable=node.get('checkable') == 'true',
        is_clickable=node.get('clickable') == 'true',
        is_enabled=node.get('enabled') == 'true',
        is_focused=node.get('focused') == 'true',
        is_focusable=node.get('focusable') == 'true',
        is_long_clickable=node.get('long-clickable') == 'true',
        is_scrollable=node.get('scrollable') == 'true',
        is_selected=node.get('selected') == 'true',
        package_name=node.get('package'),
        resource_id=node.get('resource-id'),
        is_visible=True,
    )
    if not is_root:
      ui_elements.append(ui_element)

    for child in node.get('children', []):
      process_node(child, is_root=False)

  process_node(parsed_hierarchy, is_root=True)
  return ui_elements
