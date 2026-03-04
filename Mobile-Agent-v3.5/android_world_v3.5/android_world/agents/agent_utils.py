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

"""Utilities for agents."""

import ast
import json
import re
from typing import Any


def extract_json(s: str) -> dict[str, Any] | None:
  """Extracts JSON from string.

  Tries conversion with ast and json modules.

  Args:
    s: A string with a JSON in it. E.g., "{'hello': 'world'}" or from CoT:
      "let's think step-by-step, ..., {'hello': 'world'}".

  Returns:
    JSON object.
  """
  pattern = r'\{.*?\}'
  match = re.search(pattern, s)
  if match:
    try:
      return ast.literal_eval(match.group())
    except (SyntaxError, ValueError) as error:
      try:
        # Try conversion with json module.
        return json.loads(match.group())
      except (SyntaxError, ValueError) as error2:
        print(
            'Cannot extract JSON, skipping due to errors %s and %s',
            error,
            error2,
        )
        return None
  else:
    return None
