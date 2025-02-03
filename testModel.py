import os
import cv2
import sys
import yt_dlp
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model

# Configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp."""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False

def predict_on_video(video_path, output_path, model):
    """Process video and predict actions."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return False
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, frame_size)
        
        frames_queue = deque(maxlen=SEQUENCE_LENGTH)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == SEQUENCE_LENGTH:
                frames_array = np.array(frames_queue)
                frames_array = np.expand_dims(frames_array, axis=0)
                predicted_labels = model.predict(frames_array)[0]
                predicted_class = CLASSES_LIST[np.argmax(predicted_labels)]
                
                cv2.putText(frame, predicted_class, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

if __name__ == "__main__":
    try:
        # Setup
        test_videos_directory = 'test_videos'
        os.makedirs(test_videos_directory, exist_ok=True)
        
        # Load model
        model_path = 'convlstm_model_2025_02_02__18_39_25.h5'
        model = load_model(model_path)
        
        # Process video
        video_url = 'https://www.youtube.com/watch?v=8u0qjmHIOcE'
        video_path = os.path.join(test_videos_directory, 'input_video.mp4')
        output_path = os.path.join(test_videos_directory, 'output_video.mp4')
        
        print("Downloading video...")
        if download_youtube_video(video_url, video_path):
            print("Processing video...")
            if predict_on_video(video_path, output_path, model):
                print(f"Video processed successfully. Output saved to: {output_path}")
            else:
                print("Error processing video")
        else:
            print("Error downloading video")
            
    except Exception as e:
        print(f"Error in main execution: {e}")