import time
import warnings

from PIL import Image
import requests
from io import BytesIO
import base64
import re
import traceback
import os
import random
from pathlib import Path


def base64decode(s: str):
    """
    Decode base64 `str` to original `bytes`.
    If the input is not a valid base64 string, return None.

    Args:
        s(str): A base64 `str` that can be used in text file.

    Returns:
        Optional[bytes]: The original decoded data with type `bytes`.
            If the input is not a valid base64 string, return None.
    """
    # return base64.b64decode(s)
    _base64_regex = re.compile(r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$')
    s = s.translate(base64._urlsafe_decode_translation)
    if not _base64_regex.fullmatch(s):
        return None
    try:
        return base64.urlsafe_b64decode(s)
    except base64.binascii.Error:
        return None

def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(video, max_num_frames, start_time=None, end_time=None):
    import decord
    decord.bridge.set_bridge("torch")
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    if video.lower().endswith('.webm'):
        # a workaround for webm, large/auto num_threads will cause error.
        num_threads = 2
    else:
        num_threads = 0

    vr = decord.VideoReader(video, num_threads=num_threads)
    fps = vr.get_avg_fps()
    # sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr))] # 所有帧
    if start_time is not None:
        assert end_time is not None
        start_idx = max(int(fps * start_time), 0)
        end_idx = min(int(fps * end_time), len(frame_idx))
        if start_idx>=end_idx:
            start_idx = end_idx-1
        frame_idx = frame_idx[start_idx:end_idx]
    
    # 最多每秒4帧 最少每秒1帧
    duration = len(frame_idx)/float(fps)
    # print('duration:',duration)
    target_max_num_frames = round(duration*4)
    target_min_num_frames = round(duration*1)
    if target_max_num_frames > max_num_frames:
        target_num_frames = max_num_frames
    else:
        # print(target_min_num_frames, target_max_num_frames)
        target_num_frames = random.randint(target_min_num_frames, target_max_num_frames)
    target_num_frames = min(len(frame_idx), target_num_frames)
    frame_idx = uniform_sample(frame_idx, target_num_frames)
    video = vr.get_batch(frame_idx).numpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    # print('video frames:', len(video))
    return video


class ImageIO(object):
    def __init__(self,):
        self.bucket = {}
        self.retry_num = 10
        
    
    def check_exists(self, image_url):
        if isinstance(image_url, Path):
            image_url = str(image_url.absolute())
        if os.path.isfile(image_url):
            return True
        return False
        
    def __call__(self, image_url, auto_retry=True, debug=False):
        if isinstance(image_url, Path):
            image_url = str(image_url.absolute())
        for i in range(self.retry_num):
            try:
                if image_url.startswith("http://") or image_url.startswith("https://"):
                    if debug:
                        print(image_url)
                    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                    if debug:
                        print(type(image))
                elif os.path.isfile(image_url):
                    image = Image.open(image_url).convert('RGB')
                else:
                    image_bytes = base64decode(image_url)
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                return image
            except Exception as e:
                if isinstance(e, (ConnectionResetError, requests.exceptions.ConnectionError)):
                    print(f'Image Reading Error: {e}')
                else:
                    traceback.print_exc()
                if auto_retry:
                    print(f"[ImageIO] Retrying {i} / {self.retry_num} {image_url}')")
                else:
                    return None
        return None
    
    def _load_video(self, videos, max_num_video_frame=16):
        # num_frames = self.num_video_frame
        if isinstance(videos, (str, dict)):
            videos = [videos]
        
        video_tensors = []
        num_frames = []
        for video_url in videos:
            encode_video_kwargs = {}
            if isinstance(video_url, dict):
                bound = video_url.get('bound', None)
                video_url = video_url['video']
                encode_video_kwargs['start_time'] = bound[0]
                encode_video_kwargs['end_time'] = bound[1]
            
        
            if Path(video_url).is_dir():
                video = [_ for _ in Path(video_url).iterdir()]
                video = sorted(video, key=lambda m: m.name)
                video = [self(str(_)) for _ in video]
                if len(video)>max_num_video_frame:
                    frame_idx = list(range(len(video)))
                    frame_idx = uniform_sample(frame_idx, max_num_video_frame)
                    video = [video[i] for i in frame_idx]
            else:
                video = encode_video(video_url, max_num_frames=max_num_video_frame, **encode_video_kwargs)
            # video, timestamp = read_frames_decord(video_url, num_frames=num_frames, sample='rand' if self.split == 'train' else 'middle')
            video_tensors.append(video)
            num_frames.append(len(video))
                # timestamps.append(timestamp)
            # else:
            #     raise NotImplementedError
        # to_pil = transforms.ToPILImage()  # 转换器定义
        video_tensors = [vt[ti] for vt in video_tensors for ti in range(len(vt))]
        return video_tensors, num_frames

class ImageIO2(object):
    def __init__(self,):
        self.retry_num = 10

    
    def check_exists(self, image_url):
        if isinstance(image_url, Path):
            image_url = str(image_url.absolute())

        if os.path.isfile(image_url):
            return True
        return False
        
    def __call__(self, image_url, auto_retry=True, debug=False):
        if isinstance(image_url, Path):
            image_url = str(image_url.absolute())
        for i in range(self.retry_num):
            try:
                if image_url.startswith("http://") or image_url.startswith("https://"):
                    if debug:
                        print(image_url)
                    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                    if debug:
                        print(type(image))
                elif os.path.isfile(image_url):
                    image = Image.open(image_url).convert('RGB')
                else:
                    image_bytes = base64decode(image_url)
                    Image.open(BytesIO(image_bytes)).convert('RGB')
                return image
            except Exception as e:
                if isinstance(e, (ConnectionResetError, requests.exceptions.ConnectionError)):
                    print(f'Image Reading Error: {e}')
                else:
                    traceback.print_exc()
                if auto_retry:
                    print(f"[ImageIO] Retrying {i} / {self.retry_num} {image_url}')")
                else:
                    return None
                time.sleep(0.5)
        return None
    
    def _load_video(self, videos, max_num_video_frame=16):
        # num_frames = self.num_video_frame
        if isinstance(videos, (str, dict)):
            videos = [videos]
        
        video_tensors = []
        num_frames = []
        for video_url in videos:
            encode_video_kwargs = {}
            if isinstance(video_url, dict):
                bound = video_url.get('bound', None)
                video_url = video_url['video']
                encode_video_kwargs['start_time'] = bound[0]
                encode_video_kwargs['end_time'] = bound[1]
            
            if Path(video_url).is_dir():
                video = [_ for _ in Path(video_url).iterdir()]
                video = sorted(video, key=lambda m: m.name)
                video = [self(str(_)) for _ in video]
                if len(video)>max_num_video_frame:
                    frame_idx = list(range(len(video)))
                    frame_idx = uniform_sample(frame_idx, max_num_video_frame)
                    video = [video[i] for i in frame_idx]
            else:
                video = encode_video(video_url, max_num_frames=max_num_video_frame, **encode_video_kwargs)
            # video, timestamp = read_frames_decord(video_url, num_frames=num_frames, sample='rand' if self.split == 'train' else 'middle')
            video_tensors.append(video)
            num_frames.append(len(video))
            # timestamps.append(timestamp)
            # else:
            #     raise NotImplementedError
        # to_pil = transforms.ToPILImage()  # 转换器定义
        video_tensors = [vt[ti] for vt in video_tensors for ti in range(len(vt))]
        return video_tensors, num_frames