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
from android_env.proto import adb_pb2
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.single import markor
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from android_world.utils import test_utils


class TestMarkorEditNote(test_utils.AdbEvalTestBase):

  def test_is_successful_edit_header(self):
    self.mock_create_file.return_value = 'Original Content'
    edited_content = adb_pb2.AdbResponse()
    edited_content.generic.output = b'Header\nOriginal Content'
    self.mock_issue_generic_request.return_value = edited_content
    env = mock.create_autospec(interface.AsyncEnv)
    params = {
        'file_name': 'test_note.md',
        'edit_type': 'header',
        'header': 'Header',
    }

    task = markor.MarkorEditNote(params)

    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_create_file.assert_called()
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()
    self.mock_issue_generic_request.assert_called()

  def test_is_successful_edit_footer(self):
    self.mock_create_file.return_value = 'Original Content'

    edited_content = adb_pb2.AdbResponse()
    edited_content.generic.output = b'Original Content\nFooter'

    self.mock_issue_generic_request.return_value = edited_content

    env = mock.create_autospec(interface.AsyncEnv)
    params = {
        'file_name': 'test_note.md',
        'edit_type': 'footer',
        'footer': 'Footer',
    }

    task = markor.MarkorEditNote(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)

    # Assert that the mock functions were called
    self.mock_create_file.assert_called()
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()
    self.mock_issue_generic_request.assert_called()

  # Test for 'replace'
  def test_is_successful_edit_replace(self):
    self.mock_create_file.return_value = 'Original Content'

    mock_edited_content = adb_pb2.AdbResponse()
    mock_edited_content.generic.output = b'Replacement Text'

    self.mock_issue_generic_request.return_value = mock_edited_content

    env = mock.create_autospec(interface.AsyncEnv)
    params = {
        'file_name': 'test_note.md',
        'edit_type': 'replace',
        'replace_text': 'Replacement Text',
    }

    task = markor.MarkorEditNote(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)

    # Assert that the mock functions were called
    self.mock_create_file.assert_called()
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()
    self.mock_issue_generic_request.assert_called()

  # Test for failure case (nothing replaced)
  def test_is_not_successful(self):
    self.mock_create_file.return_value = 'Original Content'

    mock_edited_content = adb_pb2.AdbResponse()
    mock_edited_content.generic.output = b'Original Content'

    self.mock_issue_generic_request.return_value = mock_edited_content

    env = mock.create_autospec(interface.AsyncEnv)
    params = {
        'file_name': 'test_note.md',
        'edit_type': 'replace',
        'replace_text': 'Replacement Text',
    }

    task = markor.MarkorEditNote(params)
    self.assertEqual(test_utils.perform_task(task, env), 0)

    # Assert that the mock functions were called
    self.mock_create_file.assert_called()
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()
    self.mock_issue_generic_request.assert_called()


class TestMarkorCreateFolder(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    self.mock_check_file_or_folder_exists.return_value = True

    env = mock.create_autospec(interface.AsyncEnv)
    params = {'folder_name': 'my_folder'}

    task = markor.MarkorCreateFolder(params)
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()

  def test_initialize_task_wrong_name(self):
    self.mock_check_file_or_folder_exists.return_value = False

    env = mock.create_autospec(interface.AsyncEnv)
    params = {'folder_name': 'my_folder'}

    task = markor.MarkorCreateFolder(params)
    self.assertEqual(test_utils.perform_task(task, env), 0)
    self.mock_remove_files.assert_called()
    self.mock_create_random_files.assert_called()


class TestMarkorDeleteNewestNote(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    env = mock.create_autospec(interface.AsyncEnv)
    file_change_time = datetime.datetime.now()
    self.mock_get_file_list_with_metadata.side_effect = [
        [
            file_utils.FileWithMetadata(
                file_name='test.txt',
                full_path='/test.txt',
                file_size=1000,
                change_time=file_change_time,
            ),
            file_utils.FileWithMetadata(
                file_name='test2.txt',
                full_path='/test.txt',
                file_size=1000,
                change_time=file_change_time + datetime.timedelta(hours=1),
            ),
        ],
        [
            file_utils.FileWithMetadata(
                file_name='test.txt',
                full_path='/test.txt',
                file_size=1000,
                change_time=file_change_time,
            ),
        ],
    ]

    task = markor.MarkorDeleteNewestNote({})
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_create_file.assert_called()
    self.mock_advance_system_time.assert_called()
    self.mock_get_file_list_with_metadata.assert_called()


class TestMarkorDeleteAllNotes(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    env = mock.create_autospec(interface.AsyncEnv)
    self.mock_get_file_list_with_metadata.side_effect = [
        [
            file_utils.FileWithMetadata(
                file_name='test.txt',
                full_path='/test.txt',
                file_size=1000,
                change_time=datetime.datetime.now(),
            )
        ],
        [],
    ]

    task = markor.MarkorDeleteAllNotes({})
    self.assertEqual(test_utils.perform_task(task, env), 1)
    self.mock_create_random_files.assert_called_once()
    self.assertEqual(self.mock_get_file_list_with_metadata.call_count, 2)


class TestMarkorCreateNoteFromClipboard(test_utils.AdbEvalTestBase):

  # Given the task is a simple variant of CreateFile task, only test
  # initialization here.
  def test_initialized_correctly(self):
    async_env = mock.create_autospec(interface.AsyncEnv)
    test_file_content = 'test file content'
    self.mock_get_clipboard_contents.return_value = test_file_content

    task = markor.MarkorCreateNoteFromClipboard(
        {'file_name': 'test_file.md', 'file_content': test_file_content}
    )
    self.mock_check_file_or_folder_exists.return_value = False
    task.initialize_task(async_env)
    self.assertIsNotNone(task.create_file_task)
    self.assertDictEqual(
        task.create_file_task.params,
        {'file_name': 'test_file.md', 'text': task.params['file_content']},
    )


class TestMarkorMergeNotes(test_utils.AdbEvalTestBase):

  def test_initialized_correctly(self):
    async_env = mock.create_autospec(interface.AsyncEnv)

    task = markor.MarkorMergeNotes({
        'file1_name': 'file1',
        'file2_name': 'file2',
        'file3_name': 'file3',
        'new_file_name': 'new_file_name',
        'file1_content': 'file1 content.\n',
        'file2_content': 'file2 content.\n',
        'file3_content': 'file3 content.\n',
    })
    self.mock_check_file_or_folder_exists.return_value = False
    task.initialize_task(async_env)
    self.assertIsNotNone(task.create_file_task)
    self.assertEqual(self.mock_create_file.call_count, 3)
    self.assertDictEqual(
        task.create_file_task.params,
        {
            'file_name': 'new_file_name',
            'text': (
                '\n\n'.join([
                    task.params['file1_content'],
                    task.params['file2_content'],
                    task.params['file3_content'],
                ])
                + '\n'
            ),
        },
    )

  @mock.patch.object(user_data_generation, 'clear_device_storage')
  @mock.patch.object(file_utils, 'clear_directory')
  def test_is_successful(
      self, unused_mock_clear_directory, unused_mock_clear_device_storage
  ):
    env = mock.create_autospec(interface.AsyncEnv)

    task = markor.MarkorMergeNotes({
        'file1_name': 'file1',
        'file2_name': 'file2',
        'file3_name': 'file3',
        'new_file_name': 'new_file_name',
        'file1_content': 'file1 content.\n',
        'file2_content': 'file2 content.\n',
        'file3_content': 'file3 content.\n',
    })

    self.mock_check_file_or_folder_exists.return_value = True
    merged_content = adb_pb2.AdbResponse()
    merged_content.generic.output = (
        b'file1 content.\n\nfile2 content.\n\nfile3 content.\n'
    )
    self.mock_issue_generic_request.side_effect = [
        merged_content,
        merged_content,
    ]

    self.assertEqual(test_utils.perform_task(task, env), 1)


class TestMarkorChangeNoteContent(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    env = mock.create_autospec(interface.AsyncEnv)
    self.mock_check_file_or_folder_exists.side_effect = [True, False, True]
    new_content = adb_pb2.AdbResponse()
    new_content.generic.output = b'new content'
    self.mock_issue_generic_request.return_value = new_content

    task = markor.MarkorChangeNoteContent({
        'original_name': 'test_file',
        'new_name': 'new_file',
        'updated_content': 'new content',
    })

    self.assertEqual(test_utils.perform_task(task, env), 1)


class TestMarkorAddNoteHeader(test_utils.AdbEvalTestBase):

  def test_is_successful(self):
    env = mock.create_autospec(interface.AsyncEnv)
    self.mock_check_file_or_folder_exists.side_effect = [True, False, True]
    new_content = adb_pb2.AdbResponse()
    new_content.generic.output = b'header to add\n\noriginal content\n'
    self.mock_issue_generic_request.return_value = new_content

    task = markor.MarkorAddNoteHeader({
        'original_name': 'test_file',
        'new_name': 'new_file',
        'header': 'header to add',
        'original_content': 'original content',
    })

    self.assertEqual(test_utils.perform_task(task, env), 1)


@mock.patch.object(file_utils, 'copy_data_to_device')
class GalleryMarkorTest(test_utils.AdbEvalTestBase):

  def setUp(self):
    super().setUp()
    self.mock_env = mock.create_autospec(spec=interface.AsyncEnv)
    self.params = {
        'file_name': 'receipt.md',
        'text': 'Date, Item, Amount\n2023-07-04, Monitor Stand, $21.52',
        'img': mock.MagicMock(),
    }

  def test_initialize_task(self, mock_copy_data_to_device):
    mock_env = mock.create_autospec(spec=interface.AsyncEnv)
    task = markor.MarkorTranscribeReceipt(self.params)
    self.mock_check_file_or_folder_exists.side_effect = [False, True]

    task.initialize_task(mock_env)

    receipt_img_path = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), 'receipt.png'
    )
    task.img.save.assert_called_once_with(receipt_img_path)
    mock_copy_data_to_device.assert_called_once_with(
        receipt_img_path,
        device_constants.GALLERY_DATA,
        mock_env.controller,
    )


if __name__ == '__main__':
  absltest.main()
