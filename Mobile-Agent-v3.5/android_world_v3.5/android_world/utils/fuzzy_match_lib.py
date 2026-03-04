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

"""Utility functions for fuzzy matching."""

import difflib


# Threshold for determining if two strings are equal using
# difflib.SequenceMatcher(...).ratio().
_MIN_DIFF_SIMILARITY = 0.9


def fuzzy_match(text1: str, text2: str, ignore_case: bool = True) -> bool:
  """Compares two strings.

  Args:
    text1: The first text.
    text2: The second text.
    ignore_case: Whether to ignore case during comparison.

  Returns:
    Whether the two strings are approximately equal.
  """
  if text1 is None or text2 is None:
    return False
  text1 = str(text1)
  text2 = str(text2)

  def text_similarity(text1: str, text2: str, ignore_case: bool) -> float:
    """Computes similiarity between two texts."""
    if ignore_case:
      text1 = text1.lower()
      text2 = text2.lower()

    return difflib.SequenceMatcher(None, text1, text2).ratio()
  return (
      text_similarity(text1, text2, ignore_case=ignore_case)
      >= _MIN_DIFF_SIMILARITY
  )
