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

"""Utilties to interact with the environment using adb."""

import os
import re
import time
from typing import Any, Callable, Collection, Iterable, Literal, Optional, TypeVar
import unicodedata
from absl import logging
from android_env import env_interface
from android_env.components import errors
from android_env.proto import adb_pb2
import immutabledict

T = TypeVar('T')

_DEFAULT_TIMEOUT_SECS = 10

# pylint: disable=line-too-long
# Maps app names to the activity that should be launched to open the app.
_PATTERN_TO_ACTIVITY = immutabledict.immutabledict({
    'google chrome|chrome': (
        'com.android.chrome/com.google.android.apps.chrome.Main'
    ),
    'google chat': 'com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity',
    'settings|system settings': 'com.android.settings/.Settings',
    'youtube|yt': 'com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity',
    'google play|play store|gps': (
        'com.android.vending/com.google.android.finsky.activities.MainActivity'
    ),
    'gmail|gemail|google mail|google email|google mail client': (
        'com.google.android.gm/.ConversationListActivityGmail'
    ),
    'google maps|gmaps|maps|google map': (
        'com.google.android.apps.maps/com.google.android.maps.MapsActivity'
    ),
    'google photos|gphotos|photos|google photo|google pics|google images': 'com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity',
    'google calendar|gcal': (
        'com.google.android.calendar/com.android.calendar.AllInOneActivity'
    ),
    'camera': 'com.android.camera2/com.android.camera.CameraLauncher',
    'audio recorder': 'com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity',
    'google drive|gdrive|drive': (
        'com.google.android.apps.docs/.drive.startup.StartupActivity'
    ),
    'google keep|gkeep|keep': (
        'com.google.android.keep/.activities.BrowseActivity'
    ),
    'grubhub': (
        'com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity'
    ),
    'tripadvisor': 'com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity',
    'starbucks': 'com.starbucks.mobilecard/.main.activity.LandingPageActivity',
    'google docs|gdocs|docs': 'com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'google sheets|gsheets|sheets': 'com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'google slides|gslides|slides': 'com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity',
    'clock': 'com.google.android.deskclock/com.android.deskclock.DeskClock',
    'google search|google': 'com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity',
    'contacts': 'com.google.android.contacts/com.android.contacts.activities.PeopleActivity',
    'facebook|fb': 'com.facebook.katana/com.facebook.katana.LoginActivity',
    'whatsapp|wa': 'com.whatsapp/com.whatsapp.Main',
    'instagram|ig': (
        'com.instagram.android/com.instagram.mainactivity.MainActivity'
    ),
    'twitter|tweet': 'com.twitter.android/com.twitter.app.main.MainActivity',
    'snapchat|sc': 'com.snapchat.android/com.snap.mushroom.MainActivity',
    'telegram|tg': 'org.telegram.messenger/org.telegram.ui.LaunchActivity',
    'linkedin': (
        'com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity'
    ),
    'spotify|spot': 'com.spotify.music/com.spotify.music.MainActivity',
    'netflix': 'com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity',
    'amazon shopping|amazon|amzn': (
        'com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity'
    ),
    'tiktok|tt': 'com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity',
    'discord': 'com.discord/com.discord.app.AppActivity$Main',
    'reddit': 'com.reddit.frontpage/com.reddit.frontpage.MainActivity',
    'pinterest': 'com.pinterest/com.pinterest.activity.PinterestActivity',
    'android world': 'com.example.androidworld/.MainActivity',
    'files': 'com.google.android.documentsui/com.android.documentsui.files.FilesActivity',
    'markor': 'net.gsantner.markor/net.gsantner.markor.activity.MainActivity',
    'clipper': 'ca.zgrs.clipper/ca.zgrs.clipper.Main',
    'messages': 'com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity',
    'simple sms messenger|simple sms': 'com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity',
    'dialer|phone': 'com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity',
    'simple calendar pro|simple calendar': 'com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity',
    'simple gallery pro|simple gallery': 'com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity',
    'miniwob': 'com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity',
    'simple draw pro': 'com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity',
    'pro expense|pro expense app': (
        'com.arduia.expense/com.arduia.expense.ui.MainActivity'
    ),
    'broccoli|broccoli app|broccoli recipe app|recipe app': (
        'com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity'
    ),
    'caa|caa test|context aware access': 'com.google.ccc.hosted.contextawareaccess.thirdpartyapp/.ChooserActivity',
    'osmand': 'net.osmand/net.osmand.plus.activities.MapActivity',
    'tasks|tasks app|tasks.org:': (
        'org.tasks/com.todoroo.astrid.activity.MainActivity'
    ),
    'open tracks sports tracker|activity tracker|open tracks|opentracks': (
        'de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity'
    ),
    'joplin|joplin app': 'net.cozic.joplin/.MainActivity',
    'vlc|vlc app|vlc player': 'org.videolan.vlc/.gui.MainActivity',
    'retro music|retro|retro player': (
        'code.name.monkey.retromusic/.activities.MainActivity'
    ),
})
# pylint: enable=line-too-long

_ORIENTATIONS = {
    'portrait': '0',
    'landscape': '1',
    'portrait_reversed': '2',
    'landscape_reversed': '3',
}

# Special app names that will trigger opening the default app.
_DEFAULT_URIS: dict[str, str] = {
    'calendar': 'content://com.android.calendar',
    'browser': 'http://',
    'contacts': 'content://contacts/people/',
    'email': 'mailto:',
    'gallery': 'content://media/external/images/media/',
}


def check_ok(response: adb_pb2.AdbResponse, message=None) -> None:
  """Check an ADB response and raise RuntimeError if not OK.

  Args:
    response: AdbResponse to check.
    message: Error message to raise on non-ok response. If not specified, a
      generic "ADB command failed" error message is used.

  Raises:
    RuntimeError: If response status is not OK.
  """
  if response.status != adb_pb2.AdbResponse.Status.OK:
    if message is not None:
      raise RuntimeError(message)
    else:
      raise RuntimeError(
          f'ADB command failed with status {response.status}:'
          f' {response.generic.output.decode()}.'
      )


def start_activity(
    activity: str,
    extra_args: Optional[Collection[str]],
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to launch the given activity.

  Args:
    activity: The activity to launch in standard android_package/activity_name
      format.
    extra_args: Optional set of arguments to be issued with the ABD broadcast.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attempting to launch %r', activity)
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          start_activity=adb_pb2.AdbRequest.StartActivity(
              full_activity=activity, extra_args=extra_args
          ),
          timeout_sec=timeout_sec,
      )
  )
  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to launch activity: %r', activity)
    return response

  logging.debug('Launch package output %r', response.generic.output)
  return response


def get_current_activity(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> tuple[Optional[str], adb_pb2.AdbResponse]:
  """Returns the full activity name that is currently opened to the user.

  Args:
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    A tuple (current_activity_name, adb_response) containing the string with
      the current activity or None if no current activity can be
      extracted, and the adb response received after issuing the request.
  """
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          get_current_activity=adb_pb2.AdbRequest.GetCurrentActivity(),
          timeout_sec=timeout_sec,
      )
  )
  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.warning(
        'Failed to obtain visible task. error_message: %r',
        response.error_message,
    )
    return (None, response)

  activity = response.get_current_activity.full_activity
  return (activity, response)


def tap_screen(
    x: int,
    y: int,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to tap the screen at the specified point.

  Args:
    x: X coordinate on the screen, in pixels.
    y: Y coordinate on the screen, in pixels.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attemting to tap the screen at (%d, %d)', x, y)
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          tap=adb_pb2.AdbRequest.Tap(x=x, y=y), timeout_sec=timeout_sec
      )
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to tap the screen')

  return response


def double_tap(
    x: int,
    y: int,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues two AdbRequests to double tap the screen at the specified point.

  Args:
    x: X coordinate on the screen, in pixels.
    y: Y coordinate on the screen, in pixels.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the second tap request.
  """
  logging.info('Attempting to double tap the screen at (%d, %d)', x, y)
  first_tap = tap_screen(x, y, env, timeout_sec=0)
  second_tap = tap_screen(x, y, env, timeout_sec=timeout_sec)
  logging.info('First tap: %s', first_tap)
  logging.info('Second tap: %s', second_tap)
  return second_tap


def long_press(
    x: int,
    y: int,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to long press the screen at the specified point.

  Args:
    x: X coordinate on the screen, in pixels.
    y: Y coordinate on the screen, in pixels.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing long press.
  """
  logging.info('Attempting to long press the screen at (%d, %d)', x, y)
  return issue_generic_request(
      ['shell', 'input', 'swipe', str(x), str(y), str(x), str(y), '1000'],
      env,
      timeout_sec,
  )


def press_home_button(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to press the HOME button in the nav bar.

  Args:
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attempting to press the HOME button')
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          press_button=adb_pb2.AdbRequest.PressButton(
              button=adb_pb2.AdbRequest.PressButton.HOME
          ),
          timeout_sec=timeout_sec,
      )
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to press the HOME button')
  return response


def press_back_button(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to press the BACK button in the nav bar.

  Args:
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attemting to press the BACK button')
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          press_button=adb_pb2.AdbRequest.PressButton(
              button=adb_pb2.AdbRequest.PressButton.BACK
          ),
          timeout_sec=timeout_sec,
      )
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to press the BACK button')

  return response


def press_enter_button(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to press the ENTER button in the nav bar.

  Args:
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attemting to press the ENTER button')
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          press_button=adb_pb2.AdbRequest.PressButton(
              button=adb_pb2.AdbRequest.PressButton.ENTER
          ),
          timeout_sec=timeout_sec,
      )
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to press the ENTER button')

  return response


def press_keyboard_generic(
    keycode: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues an AdbRequest to press any button in the keyboard.

  Args:
    keycode: The keycode to press.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  logging.info('Attemting to press the keyboard button: %s', keycode)

  response = issue_generic_request(
      ['shell', 'input', 'keyevent', keycode],
      env,
      timeout_sec,
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to press the keyboard button: %s', keycode)

  return response


def _adb_text_format(text: str) -> str:
  """Prepares text for use with adb."""
  to_escape = [
      '\\',
      ';',
      '|',
      '`',
      '\r',
      ' ',
      "'",
      '"',
      '&',
      '<',
      '>',
      '(',
      ')',
      '#',
      '$',
  ]
  for char in to_escape:
    text = text.replace(char, '\\' + char)
  normalized_text = unicodedata.normalize('NFKD', text)
  return normalized_text.encode('ascii', 'ignore').decode('ascii')


def _split_words_and_newlines(text: str) -> Iterable[str]:
  """Split lines of text into individual words and newline chars."""
  lines = text.split('\n')
  for i, line in enumerate(lines):
    words = line.split(' ')
    for j, word in enumerate(words):
      if word:
        yield word
      if j < len(words) - 1:
        yield '%s'
    if i < len(lines) - 1:
      yield '\n'


def type_text(
    text: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> None:
  """Issues an AdbRequest to type the specified text string word-by-word.

  It types word-by-word to fix issue where sometimes long text strings can be
  typed out of order at the character level. Additionally, long strings can time
  out and word-by-word fixes this, while allowing us to keep a lot timeout per
  word.

  Args:
    text: The text string to be typed.
    env: The environment.
    timeout_sec: A timeout to use for this operation. Note: For longer texts,
      this should be longer as it takes longer to type.
  """
  words = _split_words_and_newlines(text)
  for word in words:
    if word == '\n':
      logging.info('Found \\n, pressing enter button.')
      press_enter_button(env)
      continue
    formatted = _adb_text_format(word)
    logging.info('Attempting to type word: %r', formatted)
    response = env.execute_adb_call(
        adb_pb2.AdbRequest(
            input_text=adb_pb2.AdbRequest.InputText(text=formatted),
            timeout_sec=timeout_sec,
        )
    )

    if response.status != adb_pb2.AdbResponse.Status.OK:
      logging.error('Failed to type word: %r', formatted)


def issue_generic_request(
    args: Collection[str] | str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Issues a generic adb command.

  Example:
  ~~~~~~~

  issue_generic_request(['shell', 'ls'], env)
  # or
  issue_generic_request('shell ls', env)

  Args:
    args: Set of arguments to be issued with the ABD broadcast. Can also be a
      string.
    env: The environment.
    timeout_sec: A timeout to use for this operation.

  Returns:
    The adb response received after issuing the request.
  """
  if isinstance(args, str):
    args_str = args
    args = args.split(' ')
  else:
    args_str = ' '.join(args)

  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          generic=adb_pb2.AdbRequest.GenericRequest(args=args),
          timeout_sec=timeout_sec,
      )
  )
  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to issue generic adb request: %r', args_str)

  return response


def get_adb_activity(app_name: str) -> Optional[str]:
  """Get a mapping of regex patterns to ADB activities top Android apps."""
  for pattern, activity in _PATTERN_TO_ACTIVITY.items():
    if re.match(pattern.lower(), app_name.lower()):
      return activity


def get_all_package_names(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> list[str]:
  """Returns all packages installed on the device.

  Args:
    env: The AndroidEnv interface.
    timeout_sec: A timeout to use for this operation.

  Returns:
    A list of installed package names.
  """
  response = env.execute_adb_call(
      adb_pb2.AdbRequest(
          package_manager=adb_pb2.AdbRequest.PackageManagerRequest(
              list=adb_pb2.AdbRequest.PackageManagerRequest.List(
                  packages=adb_pb2.AdbRequest.PackageManagerRequest.List.Packages()
              )
          ),
          timeout_sec=timeout_sec,
      )
  )
  if response.status != adb_pb2.AdbResponse.Status.OK:
    logging.error('Failed to issue package manager request.')

  package_names = list(response.package_manager.list.items)
  return package_names


def get_all_apps(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> list[str]:
  """Returns all apps installed on the device.

  Note: the output list will not be exhaustive as it is currently based on a
  mapping we define, so any apps not included in that mapping will not be
  output here.

  Args:
    env: The AndroidEnv interface.
    timeout_sec: A timeout to use for this operation. If not set the default
      timeout will be used.

  Returns:
    A list of app names.
  """
  packages = get_all_package_names(env, timeout_sec)
  package_to_app = {
      v.split('/')[0]: k.split('|')[0] for k, v in _PATTERN_TO_ACTIVITY.items()
  }
  app_names = []
  for package in packages:
    if package in package_to_app:
      app_names.append(package_to_app[package])

  return app_names


def _launch_default_app(
    app_key: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Launches a default application with a predefined data URI."""
  if app_key not in _DEFAULT_URIS:
    raise ValueError(
        f'Unrecognized app key: {app_key}. Must be one of'
        f' {list(_DEFAULT_URIS.keys())}'
    )
  data_uri = _DEFAULT_URIS[app_key]
  adb_command = [
      'shell',
      'am',
      'start',
      '-a',
      'android.intent.action.VIEW',
      '-d',
      data_uri,
  ]
  response = issue_generic_request(adb_command, env, timeout_sec)
  return response


def launch_app(
    app_name: str,
    env: env_interface.AndroidEnvInterface,
) -> Optional[str]:
  """Uses regex and ADB activity to try to launch an app.

  Args:
    app_name: The name of the app, as represented as a key in
      _PATTERN_TO_ACTIVITY.
    env: The environment.

  Returns:
    The name of the app that is launched.
  """

  if app_name in _DEFAULT_URIS:
    _launch_default_app(app_name, env)
    return app_name

  activity = get_adb_activity(app_name)
  if activity is None:
    #  If the app name is not in the mapping, assume it is a package name.
    response = issue_generic_request(
        ['shell', 'monkey', '-p', app_name, '1'], env, timeout_sec=5
    )
    logging.info('Launching app by package name, response: %r', response)
    return app_name
  start_activity(activity, extra_args=[], env=env, timeout_sec=5)
  return app_name


def extract_package_name(activity: str) -> str:
  """Extract the package name from the activity string."""
  return activity.split('/')[0]


def close_recents(env: env_interface.AndroidEnvInterface):
  """Closes all recent apps."""
  response = issue_generic_request('shell dumpsys activity recents', env)
  if response.status != adb_pb2.AdbResponse.Status.OK:
    return
  recents_ids = re.findall(r'id=(\d+)', response.generic.output.decode())
  for recents_id in recents_ids:
    issue_generic_request(['shell', 'am', 'stack', 'remove', recents_id], env)


def close_app(
    app_name: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: Optional[float] = _DEFAULT_TIMEOUT_SECS,
) -> Optional[str]:
  """Uses regex and ADB package name to try to directly close an app.

  Args:
    app_name: The name of the app, as represented as a key in
      _PATTERN_TO_ACTIVITY.
    env: The environment.
    timeout_sec: The timeout.

  Returns:
    The app name that is closed.
  """
  activity = get_adb_activity(app_name)
  if activity is None:
    logging.error('Failed to close app: %r', app_name)
    return None
  package_name = extract_package_name(activity)
  issue_generic_request(
      ['shell', 'am', 'force-stop', package_name], env, timeout_sec
  )
  return app_name


def generate_swipe_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = None,
) -> list[str]:
  """Sends a swipe action to the simulator.

  Args:
    start_x: The x-coordinate of the start of the swipe.
    start_y: The y-coordinate of the start of the swipe.
    end_x: The x-coordinate of the end of the swipe.
    end_y: The y-coordinate of the end of the swipe.
    duration_ms: If given, the duration of time in milliseconds to take to
      complete the swipe. This value can differentiate a swipe from a fling.

  Returns:
    List of adb arguments.
  """
  duration_str = str(duration_ms) if duration_ms else ''
  return [
      'shell',
      'input',
      'swipe',
      str(start_x),
      str(start_y),
      str(end_x),
      str(end_y),
      duration_str,
  ]


def generate_drag_and_drop_command(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: Optional[int] = None,
) -> list[str]:
  """Sends a drag and drop action to the simulator.

  Args:
    start_x: The x-coordinate of the start of the drag and drop.
    start_y: The y-coordinate of the start of the drag and drop.
    end_x: The x-coordinate of the end of the drag and drop.
    end_y: The y-coordinate of the end of the drag and drop.
    duration_ms: If given, the duration of time in milliseconds to take to
      complete the drag and drop.

  Returns:
    List of adb arguments.
  """
  duration_str = str(duration_ms) if duration_ms else ''
  return [
      'shell',
      'input',
      'draganddrop',
      str(start_x),
      str(start_y),
      str(end_x),
      str(end_y),
      duration_str,
  ]


def send_android_intent(
    command: str,
    action: str,
    env: env_interface.AndroidEnvInterface,
    data_uri: str | None = None,
    mime_type: str | None = None,
    extras: dict[str, Any] | None = None,
    timeout_sec: int = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Sends an intent to Android device using adb.

  This is a low-level command for sending an intent with additional parameters.
  When these additional parameters are not necessary, consider instead using
  `adb_utils.start_activity()` or `env.execute_adb_call()` with
  `AdbRequest.StartActivity` or `AdbRequest.SendBroadcast`.

  Args:
    command: Either "start" for start activity intents or "broadcast" for
      broadcast intents.
    action: The broadcast action (e.g. "android.intent.action.VIEW").
    env: The environment to which the broadcast is sent.
    data_uri: Optional intent data URI (e.g. "content://contacts/people/1").
    mime_type: Optional mime type (e.g. "image/png").
    extras: Dictionary containing keys and values to be sent as extras.
    timeout_sec: The maximum time in seconds to wait for the broadcast to
      complete.

  Returns:
    AdbResponse object.
  """
  if command not in ['start', 'broadcast']:
    raise ValueError('Intent command must be either "start" or "broadcast"')

  adb_command = ['shell', 'am', command, '-a', action]

  if data_uri:
    adb_command.extend(['-d', f'"{data_uri}"'])

  if mime_type:
    adb_command.extend(['-t', f'"{mime_type}"'])

  if extras:
    for key, value in extras.items():
      if isinstance(value, tuple):
        type_override, value = value
        if type_override == 'str':
          adb_command.extend(['--es', key, f'"{value}"'])
        elif type_override == 'bool':
          adb_command.extend(['--ez', key, f'"{value}"'])
        elif type_override == 'int':
          adb_command.extend(['--ei', key, f'"{value}"'])
        elif type_override == 'long':  # long type only available via override.
          adb_command.extend(['--el', key, f'"{value}"'])
        elif type_override == 'float':
          adb_command.extend(['--ef', key, f'"{value}"'])
        elif type_override == 'string array':
          array_str = ','.join(value)
          adb_command.extend(['--esa', key, f'"{array_str}"'])
      elif isinstance(value, str):
        adb_command.extend(['--es', key, f'"{value}"'])
      elif isinstance(value, bool):
        adb_command.extend(['--ez', key, f'"{value}"'])
      elif isinstance(value, int):
        adb_command.extend(['--ei', key, f'"{value}"'])
      # long type only available via override above.
      elif isinstance(value, float):
        adb_command.extend(['--ef', key, f'"{value}"'])
      elif isinstance(value, list):
        array_str = ','.join(value)
        adb_command.extend(['--esa', key, f'"{array_str}"'])
      else:
        raise ValueError(f'Unrecognized extra type for {key}')

  return issue_generic_request(adb_command, env, timeout_sec)


def get_api_level(env: env_interface.AndroidEnvInterface) -> int:
  """Gets the API level of the device.

  Args:
    env: The environment.

  Returns:
    The API level.

  Raises:
    RuntimeError: If adb command does not successfully execute.
  """
  version = issue_generic_request(
      ['shell', 'getprop ro.build.version.sdk'], env
  )
  if version.status != adb_pb2.AdbResponse.Status.OK:
    raise RuntimeError('Failed to get API level.')
  return int(version.generic.output)


def _toggle_svc(
    service: str,
    on_or_off: Literal['on', 'off'],
    env: env_interface.AndroidEnvInterface,
) -> adb_pb2.AdbResponse:
  """Toggles a system service on or off using svc.

  Args:
    service: The name of the service to toggle.
    on_or_off: The state to set ('on' or 'off').
    env: The Android environment.

  Returns:
    adb status.

  Raises:
    ValueError: If invalid on_or_off is provided.
  """
  if on_or_off not in ('on', 'off'):
    raise ValueError('Must be one of on or off.')

  cmd = 'enable' if on_or_off == 'on' else 'disable'
  return issue_generic_request(['shell', 'svc', service, cmd], env)


def toggle_wifi(
    env: env_interface.AndroidEnvInterface, on_or_off: Literal['on', 'off']
) -> adb_pb2.AdbResponse:
  """Toggles wifi on or off.

  Args:
    env: The Android environment.
    on_or_off: Whether to turn it on or off.

  Returns:
    adb status.
  """
  return _toggle_svc('wifi', on_or_off, env)


def toggle_bluetooth(
    env: env_interface.AndroidEnvInterface, on_or_off: Literal['on', 'off']
) -> adb_pb2.AdbResponse:
  """Toggles Bluetooth on or off.

  Args:
    env: The Android environment.
    on_or_off: Whether to turn it on or off.

  Returns:
    adb status.
  """
  return _toggle_svc('bluetooth', on_or_off, env)


def set_brightness(
    max_or_min: str, env: env_interface.AndroidEnvInterface
) -> adb_pb2.AdbResponse:
  """Sets screen brightness to maximum or minimum.

  Args:
    max_or_min: Whether to set it to maximum or minimum.
    env: The environment.

  Returns:
    The adb status.

  Raises:
    ValueError: If invalid max_or_min is provided.
  """
  if max_or_min not in ('max', 'min'):
    raise ValueError('Must be one of max or min.')

  brightness_level = '255' if max_or_min == 'max' else '1'

  return issue_generic_request(
      [
          'shell',
          'settings',
          'put',
          'system',
          'screen_brightness',
          brightness_level,
      ],
      env,
  )


def clear_app_data(
    package_name: str, env: env_interface.AndroidEnvInterface
) -> adb_pb2.AdbResponse:
  """Clears all data for a given package.

  Args:
    package_name: The package name of the app whose data is to be cleared.
    env: The environment.

  Returns:
    adb status.
  """
  try:
    return issue_generic_request(['shell', 'pm', 'clear', package_name], env)
  except errors.AdbControllerError as exc:
    raise errors.AdbControllerError(
        f'Failed to clear app data for package {package_name}. Is the app'
        ' installed?'
    ) from exc


def toggle_airplane_mode(
    on_or_off: Literal['on', 'off'], env: env_interface.AndroidEnvInterface
) -> adb_pb2.AdbResponse:
  """Toggles airplane mode on or off.

  Args:
    on_or_off: Whether to turn it on or off.
    env: The Android environment.

  Returns:
    adb status.

  Raises:
    ValueError: If invalid on_or_off is provided.
  """
  if on_or_off not in ('on', 'off'):
    raise ValueError('Must be one of on or off.')
  state = '1' if on_or_off == 'on' else '0'
  return issue_generic_request(
      ['shell', 'settings', 'put', 'global', 'airplane_mode_on', state], env
  )


def install_apk(
    apk_location: str, env: env_interface.AndroidEnvInterface
) -> None:
  """Installs Android World APK.

  Args:
    apk_location: Location of apk.
    env: The environment.

  Raises:
    ValueError: If apk location does not exist.
  """
  if not os.path.exists(apk_location):
    raise ValueError('APK does not exist.')
  issue_generic_request(['install', apk_location], env, timeout_sec=30.0)


def check_airplane_mode(env: env_interface.AndroidEnvInterface) -> bool:
  """Checks if airplane mode is enabled.

  Args:
    env: The Android environment.

  Returns:
    True if airplane mode is enabled, False otherwise.

  Raises:
    RuntimeError: If cannot execute airplane mode check.
  """
  response = issue_generic_request(
      ['shell', 'settings', 'get', 'global', 'airplane_mode_on'], env
  )

  if response.status != adb_pb2.AdbResponse.Status.OK:
    raise RuntimeError(
        f'ADB command failed with status {response.status}:'
        f' {response.generic.output.decode()}.'
    )

  return response.generic.output.decode().replace('\r', '').strip('\n') == '1'


def extract_broadcast_data(raw_output: str) -> Optional[str]:
  """Extracts the data from an adb broadcast command output.

  Args:
    raw_output: The adb command output.

  Returns:
    Extracted data as a string, or None if the result is 0.
  """
  if 'Broadcast completed: result=-1, data=' in raw_output:
    return raw_output.split('data=')[1].strip('"\r\n')
  elif 'Broadcast completed: result=0' in raw_output:
    return None
  else:
    raise ValueError(f'Unexpected broadcast output: {raw_output}')


def _extract_clipper_output(raw_output: str) -> str:
  """Parses the clipper output from the adb command.

  Args:
    raw_output: The adb command output.

  Returns:
    The clipboard content as a string.

  Raises:
    RuntimeError: If the adb command does not successfully execute or if the
      app is not in the foreground.
  """
  parsed_data = extract_broadcast_data(raw_output)
  if parsed_data is not None:
    return parsed_data
  else:
    raise RuntimeError(
        'Clipper app must be in the foreground to access clipboard. '
        'Additionally, app privileges must be granted manually by opening the '
        'clipper app and granting them.'
    )


def get_clipboard_contents(env: env_interface.AndroidEnvInterface) -> str:
  """Gets the clipboard content from the Android device.

  Args:
    env: The environment.

  Returns:
    The clipboard content as a string.

  Raises:
    RuntimeError: If the adb command does not successfully execute or if the
      app is not in the foreground.
  """
  if launch_app('clipper', env) is None:
    raise RuntimeError(
        'Clipper app must be in the foreground to access clipboard. You may'
        ' need to install clipper app.'
    )

  time.sleep(0.5)
  res = issue_generic_request(
      ['shell', 'am', 'broadcast', '-a', 'clipper.get'], env
  )

  if res.status != adb_pb2.AdbResponse.Status.OK:
    raise RuntimeError('Failed to get clipboard content.')

  output_str = res.generic.output.decode('utf-8')
  result = _extract_clipper_output(output_str)

  press_back_button(env)
  return result


def change_orientation(
    orientation: str, env: env_interface.AndroidEnvInterface
) -> None:
  """Changes the screen orientation.

  Args:
    orientation: str, The new orientation. Can be portrait, landscape,
      reverse_portrait, or reverse_landscape.
    env: The environment.

  Raises:
    ValueError if invalid orientation is provided.
  """
  if orientation not in _ORIENTATIONS:
    raise ValueError(
        f'Unknown orientation provided: {orientation} not in'
        f' {_ORIENTATIONS.keys()}'
    )
  command = [
      'shell',
      'settings',
      'put',
      'system',
  ]
  # Turn off accelerometer.
  issue_generic_request(command + ['accelerometer_rotation', '0'], env)
  issue_generic_request(
      command + ['user_rotation', _ORIENTATIONS[orientation]], env
  )


def set_clipboard_contents(
    content: str, env: env_interface.AndroidEnvInterface
) -> None:
  """Sets the clipboard content on the Android device.

  NOTE: If using an Emulator, the contents of your clipboard on your local
  machine may transfer to the emulator when focused on the emulator. Thus the
  result of this function can be overwritten just by switching windows.

  Args:
    content: Content to put into clipboard.
    env: The environment.

  Raises:
    RuntimeError: If the adb command does not successfully execute or if the
    app is not in the foreground.
  """
  if launch_app('clipper', env) is None:
    raise RuntimeError(
        'Clipper app must be in the foreground to access clipboard. You may'
        ' need to install clipper app.'
    )

  time.sleep(0.5)
  content = _adb_text_format(content)
  output_str = issue_generic_request(
      ['shell', 'am', 'broadcast', '-a', 'clipper.set', '-e', 'text', content],
      env,
  ).generic.output.decode('utf-8')
  _extract_clipper_output(output_str)
  press_back_button(env)


def grant_permissions(
    activity_name: str,
    permission: str,
    env: env_interface.AndroidEnvInterface,
) -> None:
  """Grants permissions on an activity.

  This is useful because it prevents pop-ups prompting user/agent for
  permission.

  See https://developer.android.com/reference/android/Manifest.permission for
  available permissions to grant.

  Args:
    activity_name: The name of the activity.
    permission: The permission to grant.
    env: The AndroidEnv instance.
  """
  issue_generic_request(
      ['shell', 'pm', 'grant', activity_name, permission],
      env,
  )


def execute_sql_command(
    db_path: str,
    sql_command: str,
    env: env_interface.AndroidEnvInterface,
) -> adb_pb2.AdbResponse:
  """Execute an arbitrary SQL command on a SQLite database file via ADB.

  Args:
    db_path: The path to the SQLite database on the Android device.
    sql_command: The SQL command to execute.
    env: The environment.

  Returns:
    The adb response received after issuing the request.
  """
  set_root_if_needed(env)
  adb_command = ['shell', f'sqlite3 {db_path} "{sql_command}"']
  adb_response = issue_generic_request(adb_command, env)
  return adb_response


def get_call_state(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> str:
  """Query the call state and the dialed number of the phone through ADB.

  Args:
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A tuple containing the call state as a string and the dialed number as a
    string.
  """
  adb_args = ['shell', 'dumpsys', 'telephony.registry']
  response = issue_generic_request(adb_args, env, timeout_sec)

  output = response.generic.output.decode('utf-8')
  state_match = re.search(r'mCallState=(\d)', output)

  state = 'UNKNOWN'

  if state_match:
    state_code = state_match.group(1)
    if state_code == '0':
      state = 'IDLE'
    elif state_code == '1':
      state = 'RINGING'
    elif state_code == '2':
      state = 'OFFHOOK'

  return state


def call_emulator(
    env: env_interface.AndroidEnvInterface,
    phone_number: str,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Simulate an incoming call in an emulator using ADB.

  Args:
    env: The Android environment interface.
    phone_number: The incoming phone number.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    adb_pb2.AdbResponse: A response object containing the ADB operation result.
  """
  escaped_phone_number = re.sub(r'[^0-9+]', '', phone_number)
  adb_args = ['emu', 'gsm', 'call', f'{escaped_phone_number}']
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def end_call_if_active(
    env: 'env_interface.AndroidEnvInterface',
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> None:
  """Ends phone call if on an active call."""
  current_state = get_call_state(env, timeout_sec)

  # This check is crucial. Otherwise pressing endcall key results in black
  # screen, potentially because it's simulating turning display off?
  if current_state in ('OFFHOOK', 'RINGING'):
    adb_args = ['shell', 'input', 'keyevent', 'KEYCODE_ENDCALL']
    issue_generic_request(adb_args, env, timeout_sec)


def clear_android_emulator_call_log(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> None:
  """Clears the call log of a specific Android emulator using the Android environment interface.

  Args:
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.
  """
  adb_args = ['shell', 'content', 'delete', '--uri', 'content://call_log/calls']
  issue_generic_request(adb_args, env, timeout_sec)


def call_phone_number(
    env: env_interface.AndroidEnvInterface,
    phone_number: str,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Initiate a phone call using ADB.

  Args:
    env: The Android environment interface.
    phone_number: The phone number to dial.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  escaped_phone_number = re.sub(r'[^0-9]', '', phone_number)
  adb_args = [
      'shell',
      'am',
      'start',
      '-a',
      'android.intent.action.CALL',
      '-d',
      f'tel:{escaped_phone_number}',
  ]
  return issue_generic_request(adb_args, env, timeout_sec)


def text_emulator(
    env: env_interface.AndroidEnvInterface,
    phone_number: str,
    message: str,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Simulate an incoming text message in an emulator using ADB.

  Args:
    env: The Android environment interface.
    phone_number: The sender's phone number.
    message: The text message content.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  escaped_phone_number = re.sub(r'[^0-9+]', '', phone_number)
  adb_args = [
      'emu',
      'sms',
      'send',
      f'{escaped_phone_number}',
      f'{message}',
  ]
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def set_default_app(
    setting_key: str,
    package_name: str,
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Set the default application for a given type using ADB.

  Args:
    setting_key: The setting key for the default application type (e.g.,
      'sms_default_application').
    package_name: The package name of the application to be set as default.
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  adb_args = ['shell', 'settings', 'put', 'secure', setting_key, package_name]
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def disable_headsup_notifications(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Disables the heads up notifications.

  Args:
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  adb_args = [
      'shell',
      'settings',
      'put',
      'global',
      'heads_up_notifications_enabled',
      '0',
  ]
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def enable_headsup_notifications(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Enables the heads up notifications.

  Args:
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  adb_args = [
      'shell',
      'settings',
      'put',
      'global',
      'heads_up_notifications_enabled',
      '1',
  ]
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def put_settings(
    namespace: adb_pb2.AdbRequest.SettingsRequest.Namespace,
    key: str,
    value: str,
    env: env_interface.AndroidEnvInterface,
) -> adb_pb2.AdbResponse:
  """Change a setting in the Android system via ADB.

  Args:
    namespace: The namespace in which the setting resides (SYSTEM, SECURE,
      GLOBAL).
    key: The key of the setting to change.
    value: The new value for the setting.
    env: The Android environment interface.

  Returns:
    The adb response received after issuing the request.
  """
  if not key:
    raise ValueError('Key must be provided.')
  if not value:
    raise ValueError('Value must be provided.')
  settings_request = adb_pb2.AdbRequest.SettingsRequest(
      name_space=namespace,
      put=adb_pb2.AdbRequest.SettingsRequest.Put(key=key, value=value),
  )
  adb_request = adb_pb2.AdbRequest(settings=settings_request)
  return env.execute_adb_call(adb_request)


def delete_contacts(
    env: env_interface.AndroidEnvInterface,
    timeout_sec: float = _DEFAULT_TIMEOUT_SECS,
) -> adb_pb2.AdbResponse:
  """Deletes all contacts.

  Args:
    env: The Android environment interface.
    timeout_sec: A timeout for the ADB operation.

  Returns:
    A response object containing the ADB operation result.
  """
  adb_args = [
      'shell',
      'pm',
      'clear',
      'com.android.providers.contacts',
  ]
  response = issue_generic_request(adb_args, env, timeout_sec)
  return response


def _parse_screen_size_response(response: str) -> tuple[int, int]:
  """Parse the adb response to extract screen size.

  Args:
    response: The adb response string.

  Returns:
    The screen width and height in pixels.
  """
  match = re.search(r'Physical size: (\d+)x(\d+)', response)
  if match:
    width, height = map(int, match.groups())
    return width, height
  else:
    raise ValueError(
        f'Screen size information not found in adb response: "{response}"'
    )


def get_screen_size(env: env_interface.AndroidEnvInterface) -> tuple[int, int]:
  """Get the screen size in pixels of an Android device via ADB.

  Args:
    env: The environment.

  Returns:
    The screen width and height in pixels.
  """
  adb_command = ['shell', 'wm size']
  adb_response = issue_generic_request(adb_command, env)
  return _parse_screen_size_response(
      adb_response.generic.output.decode('utf-8')
  )


def get_logical_screen_size(
    env: env_interface.AndroidEnvInterface,
) -> tuple[int, int]:
  """Returns the logical screen size.

  The logical screen size is the screen size that applications use to render
  their interfaces which might be different than the physical screen size when
  orientation/resolution changes. The coordinates we get from A11y tree are
  based on the logical screen size.

  Args:
    env: The AndroidEnv interface.

  Returns:
    The logical screen size in (width, height).
  """
  response = issue_generic_request(
      'shell dumpsys input | grep logicalFrame', env
  )
  if response.status:
    raw_output = response.generic.output.decode('utf-8')
    pattern = r'logicalFrame=\[0, 0, (\d+), (\d+)\]'
    matches = re.findall(pattern, raw_output)
    for m in matches:
      if int(m[0]) == 0 and int(m[1]) == 0:
        continue
      width, height = (int(m[0]), int(m[1]))
      return (width, height)
  raise ValueError('Failed to get logical screen size.')


def get_physical_frame_boundary(
    env: env_interface.AndroidEnvInterface,
) -> tuple[int, int, int, int]:
  """Returns the physical frame boundary.

  Args:
    env: The AndroidEnv interface.

  Returns:
    First two integers are the coordinates for top left corner, last two are for
    lower right corner. All coordinates are given in portrait orientation.
  """
  response = issue_generic_request(
      'shell dumpsys input | grep physicalFrame', env
  )
  if response.status:
    raw_output = response.generic.output.decode('utf-8')
    pattern = r'physicalFrame=\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, raw_output)
    for m in matches:
      if (
          int(m[0]) == 0
          and int(m[1]) == 0
          and int(m[2]) == 0
          and int(m[3]) == 0
      ):
        continue
      orientation = get_orientation(env)
      if orientation == 0 or orientation == 2:
        return (int(m[0]), int(m[1]), int(m[2]), int(m[3]))
      return (int(m[1]), int(m[0]), int(m[3]), int(m[2]))
  raise ValueError('Failed to get physical frame boundary.')


def get_orientation(
    env: env_interface.AndroidEnvInterface,
) -> int:
  """Returns the current screen orientation.

  The returned value follows the normal convention, 0 for portrait, 1 for
  landscape, 2 for reverse portrait, 3 for reverse landscape.

  Args:
    env: The AndroidEnv interface.

  Returns:
    The screen orientation.
  """
  response = issue_generic_request(
      'shell dumpsys window | grep mCurrentRotation', env
  )
  if response.status:
    raw_output = response.generic.output.decode('utf-8')
    pattern = r'mCurrentRotation=ROTATION_(\d+)'
    matches = re.findall(pattern, raw_output)
    for m in matches:
      return int(m) // 90
  raise ValueError('Failed to get orientation.')


def set_screen_size(
    width: int,
    height: int,
    env: env_interface.AndroidEnvInterface,
) -> adb_pb2.AdbResponse:
  """Sets the (logical) screen size (resolution) of the Android device via ADB.

  Args:
    width: The desired screen width.
    height: The desired screen height.
    env: The AndroidEnv interface.

  Returns:
    The adb response received after issuing the request.
  """
  # Command will fail if width equals height.
  if width <= 0 or height <= 0 or width == height:
    raise ValueError(
        'Screen size not valid (need to be positive, width can not equal'
        ' height).'
    )
  # Construct the ADB command for setting screen size
  adb_command = ['shell', f'wm size {width}x{height}']

  # Issue the command and return the response
  return issue_generic_request(adb_command, env)


def retry(n: int) -> Callable[[Any], Any]:
  """Decorator to retry ADB commands."""

  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args: Any, **kwargs: Any) -> T:
      attempts = 0
      while attempts < n:
        try:
          return func(*args, **kwargs)
        except errors.AdbControllerError:
          attempts += 1
          if attempts >= n:
            raise
          print(f'Could not execute {func}. Retrying...')
          time.sleep(2)
        except Exception as exc:
          raise exc

    return wrapper

  return decorator


def set_root_if_needed(
    env: env_interface.AndroidEnvInterface, timeout_sec: Optional[float] = None
) -> adb_pb2.AdbResponse:
  """Checks if ADB is running as root, and if not, attempts to set root.

  Args:
      env: The environment.
      timeout_sec: A timeout to use for this operation.

  Returns:
      bool: True if root is set (or was already set), False otherwise.
  """
  response = issue_generic_request(['shell', 'whoami'], env, timeout_sec)

  if response.generic.output.decode('utf-8').strip() == 'root':
    return response

  return issue_generic_request(['root'], env, timeout_sec)


def uiautomator_dump(env, timeout_sec: Optional[float] = 30) -> str:
  """Issues a uiautomator dump request and returns the UI hierarchy."""
  dump_args = 'shell uiautomator dump /sdcard/window_dump.xml'
  issue_generic_request(dump_args, env, timeout_sec=timeout_sec)

  read_args = 'shell cat /sdcard/window_dump.xml'
  response = issue_generic_request(read_args, env, timeout_sec=timeout_sec)

  return response.generic.output.decode('utf-8')
