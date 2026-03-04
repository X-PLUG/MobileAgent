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
from android_world.task_evals.information_retrieval import information_retrieval_registry


class InformationRetrievalRegistryTest(absltest.TestCase):

  def test_read_in_tasks(self):
    tasks = information_retrieval_registry.InformationRetrievalRegistry(
    )._read_tasks()
    self.assertNotEmpty(list(tasks.tasks))
    for task in tasks.tasks:
      self.assertNotEmpty(task.name)
      self.assertNotEmpty(task.prompt)

  def test_registry(self):
    ir_registry = information_retrieval_registry.InformationRetrievalRegistry(
    )
    tasks = ir_registry._read_tasks()
    registry = ir_registry.registry
    for task in tasks.tasks:
      task_class = registry[task.name]
      self.assertIn(task.name, registry)
      self.assertEqual(task.name, task_class.__name__)
      self.assertEqual(
          task_class(task_class.generate_random_params()).task_template,
          task,
      )


if __name__ == "__main__":
  absltest.main()
