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

"""This module defines classes for setting up various applications in the Android World environment.

Each class represents an app and includes methods for retrieving its APK name
and performing setup tasks specific to that app using the Android Environment
Interface.
"""

import abc
import os
import time
from typing import Iterable
from absl import logging
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import tools
from android_world.task_evals.information_retrieval import joplin_app_utils
from android_world.utils import file_utils
import requests


APP_DATA = file_utils.convert_to_posix_path(os.path.dirname(__file__),
'app_data')


def download_app_data(file_name: str) -> str:
  """Downloads file from a GCS bucket, if not cached, and installs it."""
  cache_dir = file_utils.convert_to_posix_path(
      file_utils.get_local_tmp_directory(), "android_world", "app_data"
  )
  remote_url = (
      f"https://storage.googleapis.com/gresearch/android_world/{file_name}"
  )
  full_path = file_utils.convert_to_posix_path(cache_dir, file_name)
  os.makedirs(cache_dir, exist_ok=True)
  if not os.path.isfile(full_path):
    logging.info("Downloading file_name %s to cache %s", file_name, cache_dir)
    response = requests.get(remote_url)
    if response.status_code == 200:
      with open(full_path, "wb") as file:
        file.write(response.content)
    else:
      raise RuntimeError(
          f"Failed to download file_name from {remote_url}, status code:"
          f" {response.status_code}"
      )
  else:
    logging.info("File already %s exists in cache %s", file_name, cache_dir)
  return full_path


class AppSetup(abc.ABC):
  """Abstract class for setting up an app."""

  # The APK name of the app. This will assumed to be downloaded in setup.py and
  # each instance of an AppSetup will be referenced using the `apk` name as the
  # key for downloading. Some apps contain multiple APK names since different
  # versions are distributed depending on the architecture. E.g., M1 Macs
  # require different APKs for some apps.
  apk_names = ""

  # The short name of the app, as used by adb_utils.
  app_name = ""

  @classmethod
  def package_name(cls) -> str:
    return adb_utils.extract_package_name(
        adb_utils.get_adb_activity(cls.app_name)
    )

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    """Performs setup tasks specific to the app."""
    adb_utils.clear_app_data(
        adb_utils.extract_package_name(
            adb_utils.get_adb_activity(cls.app_name)
        ),
        env.controller,
    )

  @classmethod
  def _copy_data_to_device(
      cls,
      files: Iterable[str],
      device_path: str,
      env: interface.AsyncEnv,
  ) -> None:
    """Helper method for copying app data  to the device.

    Args:
      files: Names of files to copy from {APP_DATA}/app_name/ to {device_path}.
      device_path: Location on device to load the files.
      env: Android environment.
    """
    for file in files:
      copy_to_device = lambda path: adb_utils.check_ok(
          file_utils.copy_data_to_device(
              path,
              device_path,
              env.controller,
          ),
          f"Failed to copy {device_path} to device.",
      )

      full_path = download_app_data(file)
      copy_to_device(full_path)


class CameraApp(AppSetup):
  """Class for setting up pre-installed Camera app."""

  app_name = "camera"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Prevent pop-up asking for permission.
    adb_utils.grant_permissions(
        adb_utils.extract_package_name(
            adb_utils.get_adb_activity(cls.app_name)
        ),
        "android.permission.ACCESS_COARSE_LOCATION",
        env.controller,
    )

    # Click through onboarding screens during first time launch.
    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      controller.click_element("NEXT")
      time.sleep(2.0)
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class ChromeApp(AppSetup):
  """Class for setting up pre-installed Chrome app."""

  app_name = "chrome"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Click through onboarding screens during first time launch.
    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      # Welcome screen.
      controller.click_element("Accept & continue")
      time.sleep(2.0)
      # Turn on sync?
      controller.click_element("No thanks")
      time.sleep(2.0)
      # Enable notifications?
      controller.click_element("No thanks")
      time.sleep(2.0)
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class ClockApp(AppSetup):
  """Class for setting up pre-installed Clock app."""

  app_name = "clock"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Open once for initial tool tip display.
    adb_utils.launch_app(cls.app_name, env.controller)
    time.sleep(2.0)
    adb_utils.close_app(cls.app_name, env.controller)


class ContactsApp(AppSetup):
  """Class for setting up pre-installed Contacts app."""

  app_name = "contacts"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Click through onboarding screens during first time launch.
    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      # Back up & organize your contacts with Google.
      controller.click_element("Skip")
      time.sleep(2.0)
      # Allow Contacts to send you notifications?
      controller.click_element("Don't allow")
      time.sleep(2.0)
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class DialerApp(AppSetup):
  """Class for setting up pre-installed Dialer app."""

  app_name = "dialer"


class FilesApp(AppSetup):
  """Class for setting up pre-installed Files app."""

  app_name = "files"


class SettingsApp(AppSetup):
  """Class for setting up pre-installed Settings app."""

  app_name = "settings"


class MarkorApp(AppSetup):
  """Class for setting up Markor app."""

  apk_names = ("net.gsantner.markor_146.apk",)
  app_name = "markor"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      controller.click_element("NEXT")
      time.sleep(2.0)
      controller.click_element("NEXT")
      time.sleep(2.0)
      controller.click_element("NEXT")
      time.sleep(2.0)
      controller.click_element("NEXT")
      time.sleep(2.0)
      controller.click_element("DONE")
      time.sleep(2.0)

      controller.click_element("OK")
      time.sleep(2.0)
      controller.click_element("Allow access to manage all files")
      time.sleep(2.0)
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class AndroidWorldApp(AppSetup):
  """Class for setting up Android World app.

  AndroidWorld app provides on-screen visualization of tasks and rewards.
  """

  apk_names = ("androidworld.apk",)
  app_name = "android world"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.issue_generic_request(
        [
            "shell",
            "appops",
            "set",
            adb_utils.extract_package_name(
                adb_utils.get_adb_activity("android world")
            ),
            "android:system_alert_window",
            "allow",
        ],
        env.controller,
    )
    adb_utils.launch_app(cls.app_name, env.controller)
    adb_utils.close_app(cls.app_name, env.controller)


class ClipperApp(AppSetup):
  """Class for setting up clipper app."""

  apk_names = ("clipper.apk",)
  app_name = "clipper"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    controller = tools.AndroidToolController(env=env.controller)
    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      time.sleep(2.0)
      controller.click_element("Continue")
      time.sleep(2.0)
      controller.click_element("OK")
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class SimpleCalendarProApp(AppSetup):
  """Class for setting up simple calendar pro app."""

  apk_names = ("com.simplemobiletools.calendar.pro_238.apk",)
  app_name = "simple calendar pro"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.launch_app(cls.app_name, env.controller)
    adb_utils.close_app(cls.app_name, env.controller)

    # Grant permissions for calendar app.
    calendar_package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity("simple calendar pro")
    )
    adb_utils.grant_permissions(
        calendar_package,
        "android.permission.READ_CALENDAR",
        env.controller,
    )
    adb_utils.grant_permissions(
        calendar_package,
        "android.permission.WRITE_CALENDAR",
        env.controller,
    )
    adb_utils.grant_permissions(
        calendar_package,
        "android.permission.POST_NOTIFICATIONS",
        env.controller,
    )


class TasksApp(AppSetup):
  """Class for setting up Tasks app."""

  apk_names = ("org.tasks_130605.apk",)
  app_name = "tasks"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.launch_app(cls.app_name, env.controller)
    adb_utils.close_app(cls.app_name, env.controller)


class SimpleDrawProApp(AppSetup):
  """Class for setting up simple draw pro app."""

  apk_names = ("com.simplemobiletools.draw.pro_79.apk",)
  app_name = "simple draw pro"


class SimpleGalleryProApp(AppSetup):
  """Class for setting up Simple Gallery Pro app."""

  PERMISSIONS = (
      "android.permission.WRITE_EXTERNAL_STORAGE",
      "android.permission.ACCESS_MEDIA_LOCATION",
      "android.permission.READ_MEDIA_IMAGES",
      "android.permission.READ_MEDIA_VIDEO",
      "android.permission.POST_NOTIFICATIONS",
  )

  apk_names = ("com.simplemobiletools.gallery.pro_396.apk",)
  app_name = "simple gallery pro"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Grant permissions for gallery app.
    package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity(cls.app_name)
    )
    for permission in cls.PERMISSIONS:
      adb_utils.grant_permissions(package, permission, env.controller)

    adb_utils.launch_app("simple gallery pro", env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      controller.click_element("All files")
      time.sleep(2.0)
      controller.click_element("Allow access to manage all files")
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class SimpleSMSMessengerApp(AppSetup):
  """Class for setting up Simple SMS Messenger app."""

  apk_names = ("com.simplemobiletools.smsmessenger_85.apk",)
  app_name = "simple sms messenger"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Make Simple Messenger the default SMS app.
    adb_utils.set_default_app(
        "sms_default_application",
        adb_utils.extract_package_name(
            adb_utils.get_adb_activity("simple sms messenger")
        ),
        env.controller,
    )

    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      time.sleep(2.0)
      controller.click_element("SMS Messenger")
      time.sleep(2.0)
      controller.click_element("Set as default")
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class AudioRecorder(AppSetup):
  """Class for setting up Audio Recorder app."""

  apk_names = ("com.dimowner.audiorecorder_926.apk",)
  app_name = "audio recorder"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.grant_permissions(
        "com.dimowner.audiorecorder",
        "android.permission.RECORD_AUDIO",
        env.controller,
    )
    adb_utils.grant_permissions(
        "com.dimowner.audiorecorder",
        "android.permission.POST_NOTIFICATIONS",
        env.controller,
    )

    # Launch the app
    adb_utils.issue_generic_request(
        [
            "shell",
            "monkey",
            "-p",
            "com.dimowner.audiorecorder",
            "-candroid.intent.category.LAUNCHER",
            "1",
        ],
        env.controller,
    )
    time.sleep(2.0)  # Let app setup.
    adb_utils.close_app(cls.app_name, env.controller)


class MiniWobApp(AppSetup):
  """Class for setting up MiniWoB app."""

  apk_names = ("miniwobapp.apk",)
  app_name = "miniwob"


class ExpenseApp(AppSetup):
  """Class for setting up Arduia Pro Expense app."""

  apk_names = ("com.arduia.expense_11.apk",)
  app_name = "pro expense"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.launch_app(cls.app_name, env.controller)
    try:
      time.sleep(2.0)
      controller = tools.AndroidToolController(env=env.controller)
      controller.click_element("NEXT")
      time.sleep(2.0)
      controller.click_element("CONTINUE")
      time.sleep(3.0)
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class RecipeApp(AppSetup):
  """Class for setting up Broccoli Recipe app."""

  apk_names = ("com.flauschcode.broccoli_1020600.apk",)
  app_name = "broccoli app"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.launch_app(cls.app_name, env.controller)
    time.sleep(2.0)
    adb_utils.close_app(cls.app_name, env.controller)


class OsmAndApp(AppSetup):
  """Class for setting up OsmAndApp map app.

  OsmAnd handles the following intents (among others*). In addition to geo
  URIs, it can handle intents using the Google Maps API as well as a few
  other apps not listed here.

  Android geo intents:
    geo:latitude,longitude
    geo:latitude,longitude?z=zoom
    geo:0,0?q=my+street+address
    geo:0,0?q=business+near+city

  OsmAnd specific intents:
    http://download.osmand.net/go?lat=&lon=&z=
    http://osmand.net/go?lat=34&lon=-106&z=11

  Google:
    google.navigation:q=34.99393,-106.61568
    http://maps.google.com/maps?q=N34.939,W106
    http://maps.google.com/maps?f=d&saddr=My+Location&daddr=lat,lon
    http://maps.google.com/maps/@34,-106,11z
    http://maps.google.com/maps/ll=34.99393,-106.61568,z=11
    https://maps.google.com/maps?q=loc:-21.8835112,-47.7838932 (Name)
    http://maps.google.com/maps?q=34,-106
    http://www.google.com/maps/dir/Current+Location/34,-106

  * https://osmand.net/docs/technical/algorithms/osmand-intents/
  """

  PERMISSIONS = (
      "android.permission.POST_NOTIFICATIONS",
      # For other possible permissions see the manifest
      # https://github.com/osmandapp/OsmAnd/blob/master/OsmAnd/AndroidManifest.xml
  )

  DEVICE_MAPS_PATH = "/storage/emulated/0/Android/data/net.osmand/files/"

  MAP_NAMES = ("Liechtenstein_europe.obf",)

  apk_names = ("net.osmand-4.6.13.apk",)
  app_name = "osmand"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    adb_utils.launch_app(cls.app_name, env.controller)
    time.sleep(2.0)

    try:
      controller = tools.AndroidToolController(env=env.controller)
      controller.click_element("SKIP DOWNLOAD")
      time.sleep(2.0)
    except ValueError:
      logging.warn(
          "First time setup did not click through all anticipated screens."
      )
    finally:
      adb_utils.close_app(cls.app_name, env.controller)

    # Grant permissions for OsmAnd mapping app.
    package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity(cls.app_name)
    )
    for permission in cls.PERMISSIONS:
      adb_utils.grant_permissions(package, permission, env.controller)

    # Copy maps to data directory.
    cls._copy_data_to_device(cls.MAP_NAMES, cls.DEVICE_MAPS_PATH, env)

    # Make sure security context is correct so that the files can be accessed.
    for map_file in cls.MAP_NAMES:
      adb_utils.check_ok(
          adb_utils.issue_generic_request(
              [
                  "shell",
                  "chcon",
                  "u:object_r:media_rw_data_file:s0",
                  file_utils.convert_to_posix_path(
                      cls.DEVICE_MAPS_PATH, map_file
                  ),
              ],
              env.controller,
          )
      )

    adb_utils.close_app(cls.app_name, env.controller)


class OpenTracksApp(AppSetup):
  """Class for setting up OpenTracks app."""

  apk_names = ("de.dennisguse.opentracks_5705.apk",)
  app_name = "open tracks sports tracker"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    adb_utils.launch_app(cls.app_name, env.controller)
    adb_utils.close_app(cls.app_name, env.controller)

    # Grant permissions for open tracks app.
    open_tracks_package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity("open tracks")
    )
    adb_utils.grant_permissions(
        open_tracks_package,
        "android.permission.ACCESS_COARSE_LOCATION",
        env.controller,
    )
    adb_utils.grant_permissions(
        open_tracks_package,
        "android.permission.ACCESS_FINE_LOCATION",
        env.controller,
    )
    adb_utils.grant_permissions(
        open_tracks_package,
        "android.permission.POST_NOTIFICATIONS",
        env.controller,
    )
    time.sleep(2.0)
    controller = tools.AndroidToolController(env=env.controller)
    # Give permission for bluetooth, can't be done through adb.
    controller.click_element("Allow")
    adb_utils.launch_app("activity tracker", env.controller)
    adb_utils.close_app("activity tracker", env.controller)


class VlcApp(AppSetup):
  """Class for setting up VLC app."""

  videos_path = "/storage/emulated/0/VLCVideos"  # Store videos here.
  apk_names = (
      "org.videolan.vlc_13050408.apk",
      "org.videolan.vlc_13050407.apk",  # Arch86 for Mac M1/M2/etc.
  )
  app_name = "vlc"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity(cls.app_name)
    )
    adb_utils.grant_permissions(
        package, "android.permission.POST_NOTIFICATIONS", env.controller
    )
    if not file_utils.check_directory_exists(cls.videos_path, env.controller):
      file_utils.mkdir(cls.videos_path, env.controller)

    time.sleep(2.0)
    # Launch similar to opening app from app launcher. This runs setup logic not
    # available using `adb shell am start`. Specifically, it will create the
    # /data/data/org.videolan.vlc/app_db/vlc_media.db file.
    adb_utils.issue_generic_request(
        [
            "shell",
            "monkey",
            "-p",
            package,
            "-candroid.intent.category.LAUNCHER",
            "1",
        ],
        env.controller,
    )
    time.sleep(2.0)
    try:
      controller = tools.AndroidToolController(env=env.controller)
      controller.click_element("Skip")
      time.sleep(2.0)
      controller.click_element("GRANT PERMISSION")
      time.sleep(2.0)
      controller.click_element("OK")
      time.sleep(2.0)
      controller.click_element("Allow access to manage all files")
    finally:
      adb_utils.close_app(cls.app_name, env.controller)


class JoplinApp(AppSetup):
  """Class for setting up Joplin app."""

  apk_names = ("net.cozic.joplin_2097740.apk",)
  app_name = "joplin"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)

    # Grant permissions for joplin app.
    joplin_package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity(cls.app_name)
    )
    adb_utils.grant_permissions(
        joplin_package,
        "android.permission.ACCESS_COARSE_LOCATION",
        env.controller,
    )
    adb_utils.grant_permissions(
        joplin_package,
        "android.permission.ACCESS_FINE_LOCATION",
        env.controller,
    )

    # Launch the app, similar to how user launches it from App Drawer.
    adb_utils.issue_generic_request(
        [
            "shell",
            "monkey",
            "-p",
            joplin_package,
            "-candroid.intent.category.LAUNCHER",
            "1",
        ],
        env.controller,
    )
    time.sleep(10.0)
    adb_utils.close_app(cls.app_name, env.controller)
    time.sleep(10.0)

    # Calling clear_dbs() without having added a note seems to make
    # the sqlite table inaccessible. Every subsequent call to clear_dbs()
    # works fine.
    joplin_app_utils.create_note(
        folder="new folder",
        title="new_note",
        body="",
        folder_mapping={},
        env=env,
    )
    joplin_app_utils.clear_dbs(env)


class RetroMusicApp(AppSetup):
  """Class for setting up Retro Music."""

  PERMISSIONS = (
      "android.permission.READ_MEDIA_AUDIO",
      "android.permission.POST_NOTIFICATIONS",
  )

  apk_names = ("code.name.monkey.retromusic_10603.apk",)
  app_name = "retro music"

  @classmethod
  def setup(cls, env: interface.AsyncEnv) -> None:
    super().setup(env)
    package = adb_utils.extract_package_name(
        adb_utils.get_adb_activity("retro music")
    )
    for permission in cls.PERMISSIONS:
      adb_utils.grant_permissions(package, permission, env.controller)

    adb_utils.launch_app(cls.app_name, env.controller)
    time.sleep(2.0)
    adb_utils.close_app(cls.app_name, env.controller)
