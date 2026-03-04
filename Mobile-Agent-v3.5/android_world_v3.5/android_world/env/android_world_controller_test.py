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

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from android_env import env_interface
from android_env.wrappers import a11y_grpc_wrapper
from android_world.env import adb_utils
from android_world.env import android_world_controller
from android_world.env import representation_utils
from android_world.utils import fake_adb_responses
from android_world.utils import file_test_utils
from android_world.utils import file_utils
import dm_env


def create_file_with_contents(contents: str) -> str:
  temp_dir = tempfile.mkdtemp()
  file_path = file_utils.convert_to_posix_path(temp_dir, 'file.txt')
  with open(file_path, 'w') as f:
    f.write(contents)
  return file_path


class AndroidWorldControllerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_issue_generic_request = self.enter_context(
        mock.patch.object(adb_utils, 'issue_generic_request')
    )
    mock_issue_generic_request.return_value = (
        fake_adb_responses.create_successful_generic_response(
            'Physical size: 100x200'
        )
    )
    self.mock_a11y_wrapper = self.enter_context(
        mock.patch.object(
            a11y_grpc_wrapper,
            'A11yGrpcWrapper',
            spec=a11y_grpc_wrapper.A11yGrpcWrapper,
        )
    )

    self.table_name = 'events'

    self.mock_copy_db = self.enter_context(
        mock.patch.object(
            file_utils,
            'tmp_directory_from_device',
            side_effect=file_test_utils.mock_tmp_directory_from_device,
        )
    )
    self.mock_copy_data_to_device = self.enter_context(
        mock.patch.object(
            file_utils,
            'copy_data_to_device',
            side_effect=file_test_utils.mock_copy_data_to_device,
        )
    )

    self.mock_remove_files = self.enter_context(
        mock.patch.object(
            file_utils,
            'clear_directory',
            side_effect=file_test_utils.mock_remove_files,
        )
    )

  def test_initialization(self):
    mock_env = mock.Mock(spec=env_interface.AndroidEnvInterface)

    env = android_world_controller.AndroidWorldController(mock_env)

    self.mock_a11y_wrapper.assert_called_with(
        mock_env,
        install_a11y_forwarding=True,
        start_a11y_service=True,
        enable_a11y_tree_info=True,
        latest_a11y_info_only=True,
    )
    env._env.reset.assert_called_once()

  def test_screen_size(self):
    mock_base_env = mock.Mock(spec=env_interface.AndroidEnvInterface)

    env = android_world_controller.AndroidWorldController(mock_base_env)

    self.assertEqual(env.device_screen_size, (100, 200))

  @mock.patch.object(adb_utils, 'get_logical_screen_size')
  @mock.patch.object(android_world_controller, 'get_a11y_tree')
  @mock.patch.object(representation_utils, 'forest_to_ui_elements')
  def test_process_timestep(
      self, mock_forest_to_ui, mock_get_a11y_tree, mock_get_logical_screen_size
  ):
    mock_base_env = mock.Mock(spec=env_interface.AndroidEnvInterface)
    env = android_world_controller.AndroidWorldController(mock_base_env)
    mock_forest = mock.Mock()
    mock_ui_elements = mock.Mock()
    mock_get_logical_screen_size.return_value = (100, 200)
    mock_get_a11y_tree.return_value = mock_forest
    mock_forest_to_ui.return_value = mock_ui_elements
    timestep = dm_env.TimeStep(
        observation={}, reward=None, discount=None, step_type=None
    )

    processed_timestep = env._process_timestep(timestep)

    self.assertEqual(processed_timestep.observation['forest'], mock_forest)
    self.assertEqual(
        processed_timestep.observation['ui_elements'], mock_ui_elements
    )
    mock_forest_to_ui.assert_called_with(
        mock_forest,
        exclude_invisible_elements=True,
    )

  @mock.patch.object(adb_utils, 'check_airplane_mode')
  @mock.patch.object(android_world_controller, 'get_controller')
  @mock.patch.object(android_world_controller, '_has_wrapper')
  @mock.patch.object(
      android_world_controller.AndroidWorldController, 'refresh_env'
  )
  def test_refresh_env(
      self,
      mock_refresh_env,
      mock_has_wrapper,
      mock_get_controller,
      mock_check_airplane_mode,
  ):
    del mock_has_wrapper, mock_get_controller, mock_check_airplane_mode
    mock_base_env = mock.Mock(spec=env_interface.AndroidEnvInterface)
    env = android_world_controller.AndroidWorldController(mock_base_env)
    unused_mock_check_airplane_mode = False
    env._env.accumulate_new_extras.side_effect = [
        {},
        {},
        {},
        {},
        {},
        {'accessibility_tree': ['success']},
    ]

    forest = env.get_a11y_forest()

    self.assertEqual(forest, 'success')
    mock_refresh_env.assert_called_once()

  def test_pull_file(self):
    file_contents = 'test file contents'
    remote_file_path = create_file_with_contents(file_contents)
    mock_base_env = mock.Mock(spec=env_interface.AndroidEnvInterface)
    env = android_world_controller.AndroidWorldController(mock_base_env)

    with env.pull_file(remote_file_path) as local_dir:
      local_path = os.path.split(remote_file_path)[1]
      local_file = open(
          file_utils.convert_to_posix_path(local_dir, local_path), 'r'
      )
      self.assertEqual(open(remote_file_path, 'r').read(), local_file.read())

    self.mock_copy_db.assert_called_once_with(
        os.path.dirname(remote_file_path), env._env, None
    )

  def test_push_file(self):
    old_file_contents = 'test file contents'
    new_file_contents = 'new file'
    remote_file_path = create_file_with_contents(old_file_contents)
    mock_base_env = mock.Mock(spec=env_interface.AndroidEnvInterface)
    env = android_world_controller.AndroidWorldController(mock_base_env)
    new_file = create_file_with_contents(new_file_contents)

    env.push_file(new_file, remote_file_path, None)

    self.assertEqual(open(remote_file_path, 'r').read(), new_file_contents)


if __name__ == '__main__':
  absltest.main()
