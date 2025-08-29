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

"""Fake user data; used to populate apps with data."""

import datetime
import functools
import logging
import os
import random
import re
import string
from android_env import env_interface
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.utils import file_utils
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pydub

_FONT_PATHS = [
    "arial.ttf",
    "Arial Unicode.ttf",
    "Roboto-Regular.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
]


@functools.cache
def get_font(size: int | float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  """Returns a sensible font at the requested size."""
  for font_name in _FONT_PATHS:
    try:
      return ImageFont.truetype(font_name, size=float(size))
    except IOError:
      continue
  return ImageFont.load_default(float(size))


_TMP = file_utils.get_local_tmp_directory()


def generate_random_string(length: int) -> str:
  """Generate a random string consists of English letter and digit with a given length.

  Args:
    length: The length of the string.

  Returns:
    A random string.
  """
  return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_noise_files(
    base_file_name: str,
    directory_path: str,
    env: env_interface.AndroidEnvInterface,
    variant_names: list[str],
    n: int = 20,
) -> None:
  """Creates random files that are variants of base_file_name and .

  Args:
    base_file_name: Each file will be variations of this.
    directory_path: Location to create the file.
    env: The environment to use.
    variant_names: Variant file names that will be used to create additional
      file names.
    n: Maximum number of files.
  """
  assert variant_names
  num_random_files = random.randint(1, n)
  names = set()
  while len(names) < num_random_files:
    if random.random() <= 0.85:
      selected_name = random.choice(variant_names)
      filename = generate_modified_file_name(selected_name)
    else:
      filename = generate_modified_file_name(base_file_name)

    no_extension = len(filename.split(".")) == 1
    if no_extension:
      _, extension = os.path.splitext(random.choice(variant_names))
      filename += extension
    names.add(filename)

  for filename in names:
    file_utils.create_file(filename, directory_path, env)


def generate_modified_file_name(base_file_name: str) -> str:
  """Generate a modified file name with random prefix or suffix, ensuring it is inserted before the extension."""
  modification_type = random.choice(
      ["date_prefix", "random_suffix", "fixed_suffix"]
  )
  as_prefix = random.choice([True, False])
  name_part, ext_part = os.path.splitext(base_file_name)
  if modification_type == "date_prefix":
    date_str = _generate_random_date_str()
    modification = f"{date_str}"
  elif modification_type == "random_suffix":
    random_suffix = generate_random_string(4)
    modification = f"{random_suffix}"
  else:
    meaningful_modifications = ["backup", "copy", "final", "edited"]
    meaningful_mod = random.choice(meaningful_modifications)
    modification = f"{meaningful_mod}"

  if as_prefix:
    modified_file_name = f"{modification}_{name_part}{ext_part}"
  else:
    modified_file_name = f"{name_part}_{modification}{ext_part}"

  return modified_file_name


def generate_random_file_name() -> str:
  adjective = random.choice(_ADJECTIVES)
  noun = random.choice(_NOUNS)
  base = f"{adjective}_{noun}"
  return generate_modified_file_name(base)


def _generate_random_date_str() -> str:
  start_date = datetime.date(2023, 1, 1)
  end_date = datetime.date(2023, 10, 14)
  date_format = "%Y_%m_%d"
  time_between_dates = end_date - start_date
  days_between_dates = time_between_dates.days
  random_number_of_days = random.randint(0, days_between_dates)
  random_date = start_date + datetime.timedelta(days=random_number_of_days)
  return random_date.strftime(date_format)


def write_to_gallery(
    data: str,
    file_name: str,
    env: interface.AsyncEnv,
):
  """Writes data to jpeg file in Simple Gallery directory.

  Args:
    data: Text string to display on jpeg file.
    file_name: The name of the file to write. It will appear in Simple Gallery.
    env: The environment to write to.
  """

  image = _draw_text(data)
  temp_storage_location = file_utils.convert_to_posix_path(_TMP, file_name)
  image.save(temp_storage_location)
  file_utils.copy_data_to_device(
      temp_storage_location,
      device_constants.GALLERY_DATA,
      env.controller,
  )
  try:
    os.remove(temp_storage_location)
  except FileNotFoundError:
    logging.warning(
        "Local file %s not found, so cannot remove it.", temp_storage_location
    )
  adb_utils.close_app("simple gallery", env.controller)


def _copy_data_to_device(
    data: str, file_name: str, location: str, env: interface.AsyncEnv
):
  """Copies data to device by first writing locally, then copying.."""
  temp_storage_location = file_utils.convert_to_posix_path(_TMP, file_name)
  with open(temp_storage_location, "w") as temp_file:
    temp_file.write(data)

  file_utils.copy_data_to_device(
      temp_storage_location,
      location,
      env.controller,
  )
  try:
    os.remove(temp_storage_location)
  except FileNotFoundError:
    logging.warning(
        "Local file %s not found, so cannot remove it.", temp_storage_location
    )


def write_to_markor(
    data: str,
    file_name: str,
    env: interface.AsyncEnv,
):
  """Writes data to Markor.

  Args:
    data: Text string to write to Markor directory as a new file.
    file_name: The name of the file to write. It will appear in Markor.
    env: The environment to write to.
  """
  _copy_data_to_device(data, file_name, device_constants.MARKOR_DATA, env)
  adb_utils.close_app("markor", env.controller)


def _create_mpeg_with_messages(
    file_path: str,
    messages: list[str],
    width: int = 320,
    height: int = 240,
    fps: int = 30,
    display_time: int = 1,
) -> None:
  """Create a small MPEG video file with messages displayed on each frame.

  Args:
    file_path: The output path for the video file, adjusted to .mp4 for
      compatibility.
    messages: A list of strings, where each string is a message to display.
    width: The width of the video frames.
    height: The height of the video frames.
    fps: The frames per second for the video.
    display_time: The time in seconds each message is displayed.

  Raises:
    RuntimeError: If the video file was not written to the device.
  """
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
  frames_per_message = display_time * fps
  for message in messages:
    for _ in range(frames_per_message):
      frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
      cv2.putText(
          frame,
          message,
          (50, height // 2),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 255, 255),
          2,
          cv2.LINE_AA,
      )
      out.write(frame)
  out.release()
  if not os.path.exists(file_path):
    raise RuntimeError(
        f"File {file_path} was not written to device. There was a problem with"
        " creating the video. Is ffmpeg installed?"
    )


def write_video_file_to_device(
    file_name: str,
    location: str,
    env: interface.AsyncEnv,
    messages: list[str] | None = None,
    message_display_time: int = 1,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> None:
  """Create a small MPEG video file with dummy data.

  Args:
    file_name: The name of the file to write.
    location: The path to write the file on the device.
    env: The Android environment.
    messages: A list of messages to display on the video.
    message_display_time: How long to display messages for.
    width: The width of the video frames.
    height: The height of the video frames.
    fps: The frames per second for the video.
  """
  if messages is None:
    messages = ["test" + str(random.randint(0, 1_000_000))]

  _create_mpeg_with_messages(
      file_utils.convert_to_posix_path(_TMP, file_name),
      messages,
      display_time=message_display_time,
      width=width,
      height=height,
      fps=fps,
  )

  file_utils.copy_data_to_device(
      file_utils.convert_to_posix_path(_TMP, file_name),
      location,
      env.controller,
  )


def _create_test_mp3(
    file_path: str, artist: str, title: str, duration_milliseconds: int = 1000
) -> str:
  """Creates a small MP3 file for testing purposes.

  Args:
    file_path: The path where the MP3 file will be saved.
    artist: The artist name.
    title: The title of the song.
    duration_milliseconds: The duration of the MP3 file in milliseconds.

  Returns:
    The name of the file.
  """
  tone = pydub.AudioSegment.silent(duration=duration_milliseconds)
  _ = tone.export(
      file_path, format="mp3", tags={"artist": artist, "title": title}
  )
  return file_path


def write_mp3_file_to_device(
    remote_path: str,
    env: interface.AsyncEnv,
    artist: str = "test_artist",
    title: str = "test_title",
    duration_milliseconds: int = 1000,
) -> None:
  """Copies a small MP3 file to the device.

  Args:
    remote_path: The location on the device where the
    env: The environment to write to.
    artist: The artist name.
    title: The title of the song.
    duration_milliseconds: The duration of the MP3 file in milliseconds.
  """
  local = file_utils.convert_to_posix_path(_TMP, os.path.basename(remote_path))
  _create_test_mp3(
      local,
      artist=artist,
      title=title,
      duration_milliseconds=duration_milliseconds,
  )
  file_utils.copy_data_to_device(
      local,
      remote_path,
      env.controller,
  )
  try:
    os.remove(local)
  except FileNotFoundError:
    logging.warning("Local file %s not found, so cannot remove it.", local)


def dict_to_notes(input_dict: dict[str, tuple[str, str]]) -> str:
  """Converts a dictionary of apartment details to a string for user notes.

  Args:
    input_dict: A dictionary where keys are apartment names and values are
      tuples of phone numbers and brief descriptions.

  Returns:
    A string formatted as user's notes after visiting apartments.
  """

  notes = []
  for apt_name, (phone, desc) in input_dict.items():
    note = f"Visited {apt_name}. Contact: {phone}. Impressions: {desc}."
    note += "\n"
    notes.append(note)

  return "\n".join(notes)


def generate_apartments() -> dict[str, tuple[str, str]]:
  """Generates fake data for apartments a user might have seen."""
  return {
      "EastSide Lofts": ("646-145-7468", "Studio, near subway, ground floor"),
      "GreenView Apts": (
          "332-403-8720",
          "One-bedroom, well-lit, second floor",
      ),
      "Harlem Heights": (
          "332-501-9132",
          "Three-bedroom, two baths, parking included",
      ),
      "Liberty Towers": (
          "212-990-3740",
          "Three-bedroom, garden view, pets allowed",
      ),
      "Park Lane": ("212-979-5588", "One-bedroom, pool access, third floor"),
      "Riverside Complex": (
          "917-499-4580",
          "One-bedroom, near park, first floor",
      ),
      "Skyline Condos": ("917-682-8736", "Penthouse, 3 baths, balcony"),
      "SunnySide Homes": (
          "332-934-7881",
          "Studio, modern design, rooftop access",
      ),
      "UrbanVille": ("646-770-5395", "Two-bedroom, pet-friendly, basement"),
      "WestEnd Apartments": (
          "646-885-5414",
          "Two-bedroom, gym access, top floor",
      ),
  }


def _draw_text(text: str, font_size: int = 24) -> Image.Image:
  """Create an image with the given text drawn on it.

  Args:
      text: The text to draw on the image.
      font_size: Size of the font.

  Returns:
      The image object with the text.
  """
  font = get_font(font_size)
  lines = text.split("\n")

  # Calculate dimensions using font metrics
  max_width = 0
  total_height = 0
  for line in lines:
    bbox = font.getbbox(line)
    max_width = max(max_width, bbox[2])
    if line.strip():  # For non-empty lines
      total_height += bbox[3]
    else:  # For empty lines (paragraph breaks)
      total_height += font_size // 2

  img_width = max_width + 20
  img_height = total_height + 20

  img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
  d = ImageDraw.Draw(img)

  y_text = 10
  for line in lines:
    if line.strip():
      d.text((10, y_text), line, fill=(0, 0, 0), font=font)
      bbox = font.getbbox(line)
      y_text += bbox[3]
    else:
      y_text += font_size // 2

  return img


def clear_internal_storage(env: interface.AsyncEnv) -> None:
  """Deletes all files from internal storage, leaving directory structure intact."""
  adb_command = [
      "shell",
      "find",
      device_constants.EMULATOR_DATA,
      "-mindepth",
      "1",
      "-type",
      "f",  # Regular file.
      "-delete",
  ]
  adb_utils.issue_generic_request(adb_command, env.controller)


def _clear_external_downloads(env: interface.AsyncEnv) -> None:
  """Clears all external downloads directories on device."""
  adb_utils.issue_generic_request(
      "shell content delete --uri content://media/external/downloads",
      env.controller,
      timeout_sec=20,  # This can sometimes take longer than 5s.
  )


def clear_device_storage(env: interface.AsyncEnv) -> None:
  """Clears commonly used storage locations on device."""
  clear_internal_storage(env)
  _clear_external_downloads(env)


# Family names taken verbatim from
# https://people.howstuffworks.com/culture-traditions/national-traditions/most-common-last-names-in-world.htm
_COMMON_FAMILY_NAMES = [
    # keep-sorted start
    "Ahmed",
    "Ali",
    "Alves",
    "Chen",
    "Fernandez",
    "Ferreira",
    "Garcia",
    "Gonzalez",
    "Hernandez",
    "Ibrahim",
    "Li",
    "Liu",
    "Lopez",
    "Martin",
    "Mohamed",
    "Mohammed",
    "Muller",
    "Pereira",
    "Wang",
    "Zhang",
    "da Silva",
    "dos Santos",
    # keep-sorted end
]

# Some of the most frequently listed given names from
# https://en.wikipedia.org/wiki/List_of_most_popular_given_names
COMMON_GIVEN_NAMES = [
    # keep-sorted start
    "Abdullah",
    "Adam",
    "Ahmed",
    "Alejandro",
    "Ali",
    "Alice",
    "Amelia",
    "Amina",
    "Amir",
    "Ana",
    "Anna",
    "Aria",
    "Arthur",
    "Ava",
    "Camila",
    "Carlos",
    "Charlie",
    "Charlotte",
    "Daniel",
    "David",
    "Elias",
    "Ella",
    "Ema",
    "Emil",
    "Emilia",
    "Emily",
    "Emma",
    "Eva",
    "Fatima",
    "Freya",
    "Gabriel",
    "George",
    "Grace",
    "Hana",
    "Hannah",
    "Henry",
    "Hugo",
    "Ian",
    "Ibrahim",
    "Isabella",
    "Isla",
    "Ivan",
    "Jack",
    "James",
    "Jose",
    "Juan",
    "Laura",
    "Leo",
    "Leon",
    "Leonardo",
    "Liam",
    "Lily",
    "Lina",
    "Louis",
    "Luca",
    "Lucas",
    "Luis",
    "Luka",
    "Maria",
    "Mariam",
    "Mark",
    "Martin",
    "Martina",
    "Maryam",
    "Mateo",
    "Matteo",
    "Maya",
    "Mia",
    "Miguel",
    "Mila",
    "Mohammad",
    "Muhammad",
    "Nikola",
    "Noa",
    "Noah",
    "Nora",
    "Oliver",
    "Olivia",
    "Omar",
    "Oscar",
    "Petar",
    "Samuel",
    "Santiago",
    "Sara",
    "Sarah",
    "Sofia",
    "Sofija",
    "Sophia",
    "Sophie",
    "Theo",
    "Theodore",
    "Thiago",
    "Thomas",
    "Valentina",
    "Victoria",
    "William",
    "Willow",
    # keep-sorted end
]


def generate_random_name(excluding: str = "") -> str:
  """Generates a random name from a minimally diverse distribution.

  This picks a name from an unbalanced distribution, designed only to reduce
  the chance of overfitting to static or simply patterned names.

  In particular, this also does not address variations in the representational
  forms of names (e.g. "last name, first", number of given names, etc.) and is
  known to have intrinsic regional biases.

  Args:
    excluding: Space- or comma- delimited names that should be excluded from
      output.

  Returns:
    A string representing a fake person's name.
  """
  exclude = re.split(excluding, r"[ ,]")
  family_name = random.choice(
      [n for n in _COMMON_FAMILY_NAMES if n not in exclude]
  )
  given_name = random.choice(
      [n for n in COMMON_GIVEN_NAMES if n not in exclude]
  )
  return f"{given_name} {family_name}"


def generate_random_number() -> str:
  """Generates a random +1 prefix 10 digit phone number.

  This generates a phone number roughly corresponding to what may be expected in
  North America, without attempt to capture variations in formatting or to
  represent the distribution of real world phone numbers.

  Returns:
    A string representing a fake phone number.
  """
  number = "".join(random.choice(string.digits) for _ in range(10))

  # Simple SMS Messenger will add a country code if one is not provided. Be
  # explicit to make sure this does not happen.
  number = "+1" + number
  return number


def generate_random_address() -> str:
  """Selects randomly from a small arbitrary set of real US mailing addresses.

  Returns:
    A string containing a real US address picked at random.
  """
  return random.choice([
      "123 Main St Girdwood, AK, 99587",
      "6 Elm St, Birmingham, AL, 35217",
      "789 E Oak St, Phoenix AZ 85006",
      "1011 S Maple St, Little Rock, AR, 72204",
      "1415 W Cedar Ave Denver, CO, 80223",
      "968 Spruce St, Hartford, CT, 06103",
      "1819 Birch Ct, Dover, DE, 19901",
      "2021 Poplar St, Atlanta, GA, 30340",
  ])


RANDOM_SENTENCES = [
    "Don't forget to water the plants while I'm away.",
    "Your dentist appointment is scheduled for 2 PM on Thursday.",
    "Lunch meeting with Sarah at 1 PM Cafe L'amour.",
    "The dog's vet appointment is next Monday at 11 AM.",
    "Parents' evening at school this Wednesday.",
    "Monthly budget meeting pushed to Friday.",
    "Pick up groceries: Milk and Bread and Apples.",
    "Gym membership renewal due on the 20th.",
    "The library book is due back on the 15th.",
    "Reminder to call Grandma for her birthday.",
    "Weekend plans: Hiking trip to Blue Mountain.",
    "Book club meets next Tuesday to discuss '1984'.",
    "Dry cleaning is ready for pick-up.",
    "Wedding anniversary on the 30th. Make reservations!",
    "Yoga class every Tuesday and Thursday at 6 PM.",
    "Hello, World!",
    "To be or not to be.",
    "A quick brown fox.",
    "Lorem Ipsum is simply dummy text.",
    "The night is dark and full of terrors.",
    "May the Force be with you.",
    "Elementary, my dear Watson.",
    "It's a bird, it's a plane.",
    "Winter is coming.",
    "The cake is a lie.",
    "Inconceivable!",
    "A journey of a thousand miles begins with a single step.",
    "I think, therefore I am.",
    "The early bird catches the worm.",
    "Ignorance is bliss.",
    "Actions speak louder than words.",
    "Beauty is in the eye of the beholder.",
    "Better late than never.",
    "Cleanliness is next to godliness.",
    "Don't cry over spilled milk.",
    "The pen is mightier than the sword.",
    "When in Rome, do as the Romans do.",
    "The squeaky wheel gets the grease.",
    "Where there is smoke, there is fire.",
    "You can't make an omelette without breaking a few eggs.",
]

EMULATOR_DIRECTORIES = {
    "Alarms": [
        "morning_alarm.mp3",
        "wake_up.mp3",
        "early_alarm.mp3",
        "daily_reminder.mp3",
        "weekend_alarm.mp3",
        "night_alarm.mp3",
        "early_bird.mp3",
        "fitness_reminder.mp3",
        "meditation_time.mp3",
    ],
    "Audiobooks": [
        "sci_fi_book.mp3",
        "history_lecture.mp3",
        "novel_chapter.mp3",
        "biography_audio.mp3",
        "mystery_novel.mp3",
        "self_help_guide.mp3",
        "adventure_story.mp3",
        "language_lessons.mp3",
        "childrens_fable.mp3",
    ],
    "DCIM": [
        "holiday_photos.jpg",
        "birthday_party.jpg",
        "wedding_event.jpg",
        "nature_pics.jpg",
        "road_trip.jpg",
        "graduation_ceremony.jpg",
        "first_day_school.jpg",
        "mountain_hike.jpg",
        "winter_holiday.jpg",
    ],
    "Documents": [
        "resume.pdf",
        "cover_letter.pdf",
        "annual_report.pdf",
        "meeting_notes.pdf",
        "project_plan.pdf",
        "expense_report.pdf",
        "invoice_details.pdf",
        "client_brief.pdf",
        "contract_agreement.pdf",
    ],
    "Download": [
        "setup_exe.exe",
        "sample_pdf.pdf",
        "test_download.zip",
        "image_file.png",
        "movie_trailer.mp4",
        "software_patch.exe",
        "ebook_reader.apk",
        "music_album.zip",
    ],
    "Movies": [
        "action_film.mp4",
        "romantic_comedy.mp4",
        "documentary.mp4",
        "horror_movie.mp4",
        "sci_fi_thriller.mp4",
        "animation_kids.mp4",
        "drama_series.mp4",
        "mystery_feature.mp4",
    ],
    "Music": [
        "rock_album.mp3",
        "jazz_song.mp3",
        "classical_music.mp3",
        "pop_hit.mp3",
        "electronic_dance.mp3",
        "folk_tunes.mp3",
        "hip_hop_beats.mp3",
        "opera_recordings.mp3",
    ],
    "Notifications": [
        "new_message.mp3",
        "app_alert.mp3",
        "system_notification.mp3",
        "calendar_event.mp3",
        "email_received.mp3",
        "weather_update.mp3",
        "traffic_info.mp3",
        "sports_score.mp3",
    ],
    "Pictures": [
        "selfie.jpg",
        "sunset.jpg",
        "beach_day.jpg",
        "city_night.jpg",
        "family_gathering.jpg",
        "pets_playing.jpg",
        "garden_blooms.jpg",
        "food_snapshot.jpg",
    ],
    "Podcasts": [
        "news_podcast.mp3",
        "tech_talk.mp3",
        "comedy_show.mp3",
        "health_series.mp3",
        "educational_content.mp3",
        "music_reviews.mp3",
        "political_discussion.mp3",
        "travel_tips.mp3",
    ],
    "Recordings": [
        "interview_recording.mp3",
        "lecture_capture.mp3",
        "memoir_audio.mp3",
        "meeting_audio.mp3",
        "brainstorm_session.mp3",
        "book_reading.mp3",
        "therapy_session.mp3",
        "personal_notes.mp3",
    ],
    "Ringtones": [
        "default_ringtone.mp3",
        "custom_tone.mp3",
        "vintage_bell.mp3",
        "modern_beep.mp3",
        "jazzy_ring.mp3",
        "funky_tune.mp3",
        "classic_music.mp3",
        "nature_sounds.mp3",
    ],
}

_ADJECTIVES = [
    "quick",
    "happy",
    "silly",
    "brave",
    "kind",
    "clever",
    "gentle",
    "proud",
    "friendly",
    "funny",
    "curious",
    "smart",
    "bold",
    "calm",
    "fierce",
    "wise",
    "strong",
    "bright",
    "eager",
    "fancy",
    "helpful",
    "jolly",
    "lively",
    "neat",
    "polite",
    "sharp",
    "shy",
    "super",
    "tough",
    "witty",
    "active",
    "alert",
    "best",
    "busy",
    "cool",
    "fair",
    "fancy",
    "fine",
    "glad",
    "good",
    "great",
    "hot",
    "nice",
    "pretty",
    "proud",
    "ready",
    "real",
    "safe",
    "sure",
    "warm",
]

_NOUNS = [
    "apple",
    "banana",
    "cat",
    "dog",
    "elephant",
    "fish",
    "guitar",
    "house",
    "island",
    "jacket",
    "king",
    "lion",
    "monkey",
    "nest",
    "ocean",
    "penguin",
    "queen",
    "rabbit",
    "snake",
    "tree",
    "umbrella",
    "violin",
    "watch",
    "xylophone",
    "yacht",
    "zebra",
    "ant",
    "bear",
    "cow",
    "deer",
    "eagle",
    "frog",
    "goat",
    "horse",
    "igloo",
    "jelly",
    "koala",
    "lamp",
    "mouse",
    "nurse",
    "owl",
    "pig",
    "quilt",
    "rose",
    "sun",
    "tiger",
    "unicorn",
    "vase",
    "wolf",
    "fox",
    "zebra",
]
