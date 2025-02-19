import os
import cv2
import yt_dlp
import numpy as np
from collections import deque

def process_video_thread(gui):
    try:
        output_dir = gui.output_entry.get()
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = gui.video_path if gui.video_path else os.path.join(output_dir, 'input_video.mp4')
        output_path = os.path.join(output_dir, 'output_video.mp4')
        
        if not gui.video_path:
            # Download video
            gui.status_queue.put("Downloading video...")
            if not download_youtube_video(gui.url_entry.get(), video_path):
                gui.status_queue.put("Error downloading video")
                return
        
        # Process video
        gui.status_queue.put("Processing video...")
        if predict_on_video(gui, video_path, output_path):
            gui.status_queue.put(f"Video processed successfully. Output saved to: {output_path}")
            play_video(gui, output_path)
        else:
            gui.status_queue.put("Error processing video")
    
    except Exception as e:
        gui.status_queue.put(f"Error: {str(e)}")
    finally:
        gui.app.after(0, gui.cleanup)

def download_youtube_video(url, output_path):
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
        return False

def predict_on_video(gui, video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            gui.status_queue.put("Error: Could not open video.")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, frame_size)
        
        frames_queue = deque(maxlen=gui.SEQUENCE_LENGTH)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized_frame = cv2.resize(frame, (gui.IMAGE_HEIGHT, gui.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == gui.SEQUENCE_LENGTH:
                frames_array = np.array(frames_queue)
                frames_array = np.expand_dims(frames_array, axis=0)
                predicted_labels = gui.model.predict(frames_array)[0]
                predicted_class = gui.CLASSES_LIST[np.argmax(predicted_labels)]
                
                cv2.putText(frame, predicted_class, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                progress = frame_count / total_frames
                gui.status_queue.put(f"Processed {frame_count}/{total_frames} frames ({progress:.1%})")
        
        cap.release()
        out.release()
        return True
        
    except Exception as e:
        gui.status_queue.put(f"Error processing video: {e}")
        return False