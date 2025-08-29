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

from absl.testing import absltest
from android_world.task_evals.single import retro_music


class TestGenerateListWithSum(absltest.TestCase):

  def test_generate_list_with_sum(self):
    trials = 10_000

    for _ in range(trials):
      n = random.randint(1, 19_000)
      m = random.randint(1, 10)

      result = retro_music._generate_list_with_sum(n, m)

      self.assertLen(result, m)
      self.assertEqual(sum(result), n)


if __name__ == "__main__":
  absltest.main()
