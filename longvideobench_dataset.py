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
    def __init__(self, data_path, split, max_num_frames=16, max_length=2048, tokenizer=None, 
                 video_dir=None, subtitles_dir=None, use_subtitles=True, use_audio=False):
        self.data_path = data_path
        self.split = split
        self.max_num_frames = max_num_frames
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.video_dir = video_dir
        self.subtitles_dir = subtitles_dir
        self.use_subtitles = use_subtitles
        self.use_audio = use_audio
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        self.video_ids = list(self.data.keys())
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_info = self.data[video_id]
        
        video_file = os.path.join(self.video_dir, f"{video_id}.mp4")
        duration = video_info["duration"]
        
        frames, frame_timestamps = load_video(video_file, duration, self.max_num_frames)
        
        if self.use_subtitles and self.subtitles_dir:
            subtitle_file = os.path.join(self.subtitles_dir, f"{video_id}.json")
            if os.path.exists(subtitle_file):
                with open(subtitle_file, 'r') as f:
                    subtitles = json.load(f)
                
                # Insert subtitles into frames
                interleaved_content = insert_subtitles_into_frames(
                    frames, frame_timestamps, subtitles, 0, duration
                )
            else:
                interleaved_content = frames
        else:
            interleaved_content = frames
            
        # Get question and answer
        question = video_info["question"]
        answer = video_info["answer"]
        
        # Prepare input
        if self.tokenizer:
            # Tokenize the input
            input_text = f"Question: {question}\\nAnswer:"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)
            
            # Tokenize the answer for labels
            labels = self.tokenizer(answer, return_tensors="pt", max_length=self.max_length, truncation=True)
            
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels["input_ids"],
                "frames": frames,
                "video_id": video_id
            }
        else:
            return {
                "question": question,
                "answer": answer,
                "frames": frames,
                "video_id": video_id,
                "interleaved_content": interleaved_content
            }