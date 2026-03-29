from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch

import json

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    
    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps

def insert_subtitles(subtitles):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            subtitle_text = subtitle["text"]
        else:
            subtitle_text = subtitle["line"]

        interleaved_list.append(subtitle_text)

    return interleaved_list
        
def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        while cur_i < len(frame_timestamps) and frame_timestamps[cur_i] < subtitle_timestamp:
            interleaved_list.append(frames[cur_i])
            cur_i += 1

        interleaved_list.append(subtitle_text)

    while cur_i < len(frame_timestamps):
        interleaved_list.append(frames[cur_i])
        cur_i += 1

    return interleaved_list

class LongVideoBenchDataset(Dataset):
    def __init__(self, ann_file, video_root, transform=None, max_num_frames=16):
        self.ann_file = ann_file
        self.video_root = video_root
        self.transform = transform
        self.max_num_frames = max_num_frames

        with open(ann_file, 'r') as f:
            self.anns = json.load(f)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        video_path = os.path.join(self.video_root, ann['video'])
        duration = ann['duration']
        frames, frame_timestamps = load_video(video_path, duration, self.max_num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        subtitles = ann.get('subtitles', [])
        interleaved = insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 0, duration)

        return interleaved
