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

"""Generates a simulated receipt image."""

import datetime
import random
from android_world.env import device_constants
from android_world.task_evals.utils import user_data_generation
from PIL import Image
from PIL import ImageDraw


def _random_date():
  """Generate a random date within the year 2023."""
  start_date = datetime.date(2023, 1, 1)
  end_date = device_constants.DT.date()
  time_between_dates = end_date - start_date
  days_between_dates = time_between_dates.days
  random_number_of_days = random.randrange(days_between_dates)
  return start_date + datetime.timedelta(days=random_number_of_days)


def _random_transaction():
  """Generate a random transaction with a date, item name, and price."""
  items = [
      ("USB-C Cable", (5, 20)),
      ("Wireless Mouse", (10, 50)),
      ("Bluetooth Keyboard", (30, 100)),
      ("External Hard Drive", (50, 150)),
      ("Webcam", (20, 70)),
      ("Monitor Stand", (15, 60)),
  ]
  item, price_range = random.choice(items)
  price = random.uniform(*price_range)
  return _random_date(), item, f"${price:.2f}"


def _random_company_info():
  """Generate random company information including name and slogan."""
  company_names = [
      "Tech Gadgets Inc.",
      "Innovate Solutions Ltd.",
      "Future Tech LLC",
      "Gadget Gurus Co.",
  ]
  slogans = [
      "Innovating the Future",
      "Technology for Tomorrow",
      "Bringing Ideas to Life",
      "Innovation and Excellence",
  ]
  return random.choice(company_names), random.choice(slogans)


def create_receipt(num_transactions: int = 1) -> tuple[Image.Image, str]:
  """Create a receipt image with random transactions and return the image and text.

  Args:
      num_transactions: The number of transactions to include in the receipt.

  Returns:
      The receipt image and the corresponding text.
  """
  company_name, slogan = _random_company_info()
  transactions = [_random_transaction() for _ in range(num_transactions)]

  # Adjust image size based on number of transactions
  img_height = max(250, 50 * (num_transactions + 4)) * 2
  img = Image.new("RGB", (500, img_height), color=(255, 255, 255))
  d = ImageDraw.Draw(img)

  font = user_data_generation.get_font(16)
  header_font = user_data_generation.get_font(20)
  footer_font = user_data_generation.get_font(12)

  # Add company name and slogan
  y_text = 100
  d.text(
      (10, y_text),
      f"{company_name}\n{slogan}",
      fill=(0, 0, 0),
      font=header_font,
  )

  all_text = f"{company_name}\n{slogan}\nDate, Item, Amount\n"
  y_text += 70
  for date, item, price in transactions:
    transaction_text = f"{date}, {item}, {price}"
    d.text((10, y_text), transaction_text, fill=(0, 0, 0), font=font)
    all_text += f"{transaction_text}\n"
    y_text += 30

  # Add footer
  footer_text = (
      "Thank you for your purchase!\nVisit us at: www.tech-gadgets.com"
  )
  d.text((10, img_height - 40), footer_text, fill=(0, 0, 0), font=footer_font)

  return img, all_text.strip()
