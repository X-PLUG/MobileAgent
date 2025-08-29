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

import tempfile
from absl.testing import absltest
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
import cv2


def get_video_properties(file_path: str) -> tuple[int, float]:
  """Retrieve the total number of frames and FPS of a video file.

  Args:
    file_path: Path to the video file.

  Returns:
    A tuple containing the total number of frames and the FPS of the video.
  """
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    raise ValueError(f"Failed to open video file: {file_path}")

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  cap.release()

  return total_frames, fps


class TestCreateMpegWithMessages(absltest.TestCase):

  def test_video_properties(self):
    file_path = file_utils.convert_to_posix_path(
        tempfile.mkdtemp(), "test_video.mp4"
    )
    messages = ["Hello", "World"]
    width = 10
    height = 12
    fps = 30
    display_time = 5

    user_data_generation._create_mpeg_with_messages(
        file_path, messages, width, height, fps, display_time
    )

    total_frames, video_fps = get_video_properties(file_path)
    self.assertEqual(video_fps, fps)
    self.assertEqual(total_frames, 300)


if __name__ == "__main__":
  absltest.main()
