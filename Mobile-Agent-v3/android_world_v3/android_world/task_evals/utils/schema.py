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

"""Helper functions for defining the schema."""

from collections.abc import Sequence
import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class Property:
  """A schema property."""

  name: str
  type: str
  is_required: bool
  options: Optional[list[str]]


def string(name: str, is_required: bool = False) -> Property:
  """Returns a string property.

  Args:
    name: The name of the property.
    is_required: If true, the property must be set in the schema.
  """
  return Property(name, type="string", is_required=is_required, options=None)


def number(name: str, is_required: bool = False) -> Property:
  """Returns a number property.

  Args:
    name: The name of the property.
    is_required: If true, the property must be set in the schema.
  """
  return Property(name, type="number", is_required=is_required, options=None)


def integer(name: str, is_required: bool = False) -> Property:
  """Returns an integer property.

  Args:
    name: The name of the property.
    is_required: If true, the property must be set in the schema.
  """
  return Property(name, type="integer", is_required=is_required, options=None)


def enum(
    name: str, options: Sequence[str], is_required: bool = False
) -> Property:
  """Returns an enum property.

  Args:
    name: The name of the property.
    options: A list of options for the enum.
    is_required: If true, the property must be set in the schema.
  """
  return Property(
      name, type="string", is_required=is_required, options=list(options)
  )


def create(properties: Sequence[Property]) -> object:
  """Returns a schema object.

  Args:
    properties: A list of properties for the schema.
  """
  def property_to_object(prop: Property) -> object:
    schema = {
        "type": prop.type,
    }
    if prop.options:
      schema["enum"] = prop.options
    return schema

  return {
      "type": "object",
      "properties": {
          property.name: property_to_object(property) for property in properties
      },
      "required": [
          property.name for property in properties if property.is_required
      ],
  }


def no_params() -> object:
  """Returns a schema object without any parameters."""
  return create([])
