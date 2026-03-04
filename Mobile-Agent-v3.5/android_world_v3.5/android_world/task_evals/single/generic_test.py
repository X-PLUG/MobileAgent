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

"""Testing generic task eval for generic.py."""

import concurrent

from absl.testing import absltest
from android_world.task_evals.single import generic
from android_world.utils import test_utils


class GenericTest(test_utils.AdbEvalTestBase):

  def test_set_instance_params(self):
    task_type = generic.create_task_type(
        "NewTask", {"instruction": "test_instruction", "app_names": ("chrome",)}
    )
    self.assertEqual(
        task_type.instance_params,
        {"instruction": "test_instruction", "app_names": ("chrome",)},
    )
    task = task_type(task_type.generate_random_params())
    self.assertEqual(task.template, "test_instruction")
    self.assertEqual(task.app_names, ("chrome",))

  def test_multiple_set_instance_params(self):
    task_type1 = generic.create_task_type(
        "NewTask1",
        {"instruction": "test_instruction1", "app_names": ("chrome",)},
    )
    self.assertEqual(
        task_type1.instance_params,
        {"instruction": "test_instruction1", "app_names": ("chrome",)},
    )
    task1 = task_type1(task_type1.generate_random_params())
    task_type2 = generic.create_task_type(
        "NewTask2",
        {"instruction": "test_instruction2", "app_names": ("gmail",)},
    )
    self.assertEqual(
        task_type2.instance_params,
        {"instruction": "test_instruction2", "app_names": ("gmail",)},
    )
    task2 = task_type2(task_type2.generate_random_params())
    self.assertEqual(task1.template, "test_instruction1")
    self.assertEqual(task2.template, "test_instruction2")
    self.assertEqual(task1.app_names, ("chrome",))
    self.assertEqual(task2.app_names, ("gmail",))

  def test_concurrent_multiple_set_instance_params(self):
    num_tasks = 10

    def new_task(i):
      return generic.create_task_type(
          f"Task{i}",
          {"instruction": f"test_instruction{i}", "app_names": (f"app{i}",)},
      )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_tasks
    ) as executor:
      futures = {executor.submit(new_task, i): i for i in range(num_tasks)}
      for f in concurrent.futures.as_completed(futures):
        i = futures[f]
        task_type = f.result()
        task = task_type(task_type.generate_random_params())
        self.assertEqual(
            task.instance_params,
            {"instruction": f"test_instruction{i}", "app_names": (f"app{i}",)},
        )
        self.assertEqual(task.template, f"test_instruction{i}")
        self.assertEqual(task.app_names, (f"app{i}",))


if __name__ == "__main__":
  absltest.main()
