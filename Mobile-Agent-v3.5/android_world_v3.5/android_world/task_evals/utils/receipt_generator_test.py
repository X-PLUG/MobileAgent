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

import datetime
from unittest import mock

from absl.testing import absltest
from android_world.task_evals.utils import receipt_generator


class ImageUtilsTest(absltest.TestCase):

  def test_random_date(self):
    date = receipt_generator._random_date()
    self.assertIsInstance(date, datetime.date)
    self.assertBetween(
        date, datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)
    )

  def test_random_transaction(self):
    transaction = receipt_generator._random_transaction()
    self.assertIsInstance(transaction, tuple)
    self.assertIsInstance(transaction[0], datetime.date)
    self.assertIsInstance(transaction[1], str)
    self.assertTrue(transaction[2].startswith('$'))

  def test_random_company_info(self):
    company_info = receipt_generator._random_company_info()
    self.assertIsInstance(company_info, tuple)
    self.assertIsInstance(company_info[0], str)
    self.assertIsInstance(company_info[1], str)

  @mock.patch('PIL.ImageDraw.Draw')
  @mock.patch('PIL.ImageFont.truetype')
  @mock.patch('PIL.Image.new')
  def test_create_receipt(self, mock_new, mock_truetype, unused_mock_draw):
    num_transactions = 3
    _, text = receipt_generator.create_receipt(num_transactions)
    mock_new.assert_called_once()
    self.assertGreaterEqual(mock_truetype.call_count, 3)
    self.assertIsInstance(text, str)
    self.assertGreaterEqual(len(text.split('\n')), num_transactions + 3)


if __name__ == '__main__':
  absltest.main()
