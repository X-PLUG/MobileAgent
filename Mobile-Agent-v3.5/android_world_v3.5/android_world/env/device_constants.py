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

"""Constants for the Pixel 6, API 33, emulator."""

import datetime

# Screen dimensions of Pixel 6.
SCREEN_HEIGHT, SCREEN_WIDTH = 2400, 1080

# Where data on emulator is stored.
EMULATOR_DATA = "/storage/emulated/0/"

# Location where app snapshots are stored.
SNAPSHOT_DATA = "/data/data/android_world/snapshots"

# keep-sorted start
AUDIORECORDER_DATA = "/storage/emulated/0/Android/data/com.dimowner.audiorecorder/files/Music/records"
DOWNLOAD_DATA = "/storage/emulated/0/Download"
GALLERY_DATA = "/sdcard/DCIM"
MARKOR_DATA = "/storage/emulated/0/Documents/Markor"
MUSIC_DATA = "/sdcard/Music"
OSMAND_DATA = "/storage/emulated/0/Android/data/net.osmand/files"
PHOTOS_DATA = "/sdcard/Pictures"
VIDEOS_DATA = "/sdcard/Movies"
# keep-sorted end

# Every task starts October 15, 2023 @ 15:34:00.
TIMEZONE = "UTC"
DT = datetime.datetime(2023, 10, 15, 15, 34, 0, tzinfo=datetime.timezone.utc)

# Format the datetime object into the Android date-time format
ANDROID_DT = DT.strftime("%m%d%H%M%y.%S")
