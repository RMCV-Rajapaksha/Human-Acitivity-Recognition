import os
import sys
import cv2
import yt_dlp
import numpy as np
import tensorflow as tf
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
from collections import deque
from tensorflow.keras.models import load_model
from datetime import datetime

# Configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

class VideoActivityRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Activity Recognition")
        self.root.geometry("1100x720")
        self.root.minsize(1000, 700)
        
        # Set color theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"
        
        # Initialize variables
        self.video_path = ""
        self.output_path = ""
        self.model = None
        self.model_path = ""
        self.processing = False
        self.preview_frame = None
        self.cap = None
        self.current_frame = None
        self.youtube_url = tk.StringVar()
        self.status_message = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)
        self.is_preview_playing = False
        self.frame_index = 0
        self.total_frames = 0
        self.display_frame = None
        
        # Create UI
        self.create_ui()
        
        # Set up directory
        self.test_videos_directory = 'test_videos_new'
        os.makedirs(self.test_videos_directory, exist_ok=True)
        
        # Schedule setup tasks
        self.root.after(100, self.load_default_model)
        
    def create_ui(self):
        # Main frame using grid layout
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure the grid layout
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create left panel for controls
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure left panel grid
        left_panel.grid_columnconfigure(0, weight=1)
        for i in range(12):
            left_panel.grid_rowconfigure(i, weight=0)
        left_panel.grid_rowconfigure(12, weight=1)  # For log text, allow expansion
        
        # App Logo/Title
        title_label = ctk.CTkLabel(left_panel, text="Video Activity\nRecognition", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, pady=(20, 30), sticky="ew")
        
        # Model selection section
        model_label = ctk.CTkLabel(left_panel, text="Model", font=ctk.CTkFont(size=16, weight="bold"))
        model_label.grid(row=1, column=0, pady=(5, 0), sticky="w", padx=10)
        
        self.model_status = ctk.CTkLabel(left_panel, text="Not loaded", 
                                        font=ctk.CTkFont(size=12))
        self.model_status.grid(row=2, column=0, pady=(0, 5), sticky="w", padx=10)
        
        load_model_btn = ctk.CTkButton(left_panel, text="Load Model", command=self.browse_model)
        load_model_btn.grid(row=3, column=0, pady=5, padx=20, sticky="ew")
        
        # Horizontal separator
        separator1 = ttk.Separator(left_panel, orient='horizontal')
        separator1.grid(row=4, column=0, sticky='ew', pady=15, padx=10)
        
        # Input video section
        input_label = ctk.CTkLabel(left_panel, text="Input Video", font=ctk.CTkFont(size=16, weight="bold"))
        input_label.grid(row=5, column=0, pady=(5, 0), sticky="w", padx=10)
        
        browse_btn = ctk.CTkButton(left_panel, text="Browse Local Video", command=self.browse_video)
        browse_btn.grid(row=6, column=0, pady=5, padx=20, sticky="ew")
        
        # YouTube URL input
        youtube_label = ctk.CTkLabel(left_panel, text="Or YouTube URL:")
        youtube_label.grid(row=7, column=0, pady=(10, 0), sticky="w", padx=10)
        
        youtube_entry = ctk.CTkEntry(left_panel, textvariable=self.youtube_url, placeholder_text="Enter YouTube URL")
        youtube_entry.grid(row=8, column=0, pady=5, padx=20, sticky="ew")
        
        download_btn = ctk.CTkButton(left_panel, text="Download & Use", command=self.download_youtube)
        download_btn.grid(row=9, column=0, pady=(0, 10), padx=20, sticky="ew")
        
        # Process button
        separator2 = ttk.Separator(left_panel, orient='horizontal')
        separator2.grid(row=10, column=0, sticky='ew', pady=15, padx=10)
        
        process_btn = ctk.CTkButton(left_panel, text="Process Video", 
                                   command=self.process_video, 
                                   fg_color="#28a745", hover_color="#218838")
        process_btn.grid(row=11, column=0, pady=10, padx=20, sticky="ew")
        
        # Log and status area
        log_frame = ctk.CTkFrame(left_panel)
        log_frame.grid(row=12, column=0, sticky="nsew", padx=10, pady=10)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        self.log_text = ctk.CTkTextbox(log_frame)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text.configure(state="disabled")
        
        # Create right panel for video display
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Configure right panel grid
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=0)
        right_panel.grid_rowconfigure(2, weight=0)
        
        # Video preview area
        self.preview_frame = ctk.CTkFrame(right_panel, fg_color="#1A1A1A")
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Set initial message in preview area
        self.display_frame = ctk.CTkLabel(self.preview_frame, text="No video loaded", 
                                         font=ctk.CTkFont(size=16))
        self.display_frame.pack(expand=True, fill="both")
        
        # Video controls
        controls_frame = ctk.CTkFrame(right_panel)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        self.play_btn = ctk.CTkButton(controls_frame, text="▶ Play", command=self.toggle_play,
                                     width=80, state="disabled")
        self.play_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_btn = ctk.CTkButton(controls_frame, text="■ Stop", command=self.stop_preview,
                                     width=80, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        save_btn = ctk.CTkButton(controls_frame, text="💾 Save Output", command=self.save_output,
                                width=120, state="normal")
        save_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Progress and status bar
        status_frame = ctk.CTkFrame(right_panel)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        status_frame.grid_columnconfigure(0, weight=3)
        status_frame.grid_columnconfigure(1, weight=1)
        
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.progress_bar.set(0)
        
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_message)
        status_label.grid(row=0, column=1, padx=10, pady=10)
        
    def load_default_model(self):
        """Try to load the default model specified in the original script."""
        default_model_path = 'convlstm_model_2025_02_02__18_39_25.h5'
        if os.path.exists(default_model_path):
            self.model_path = default_model_path
            self.load_model(default_model_path)
        else:
            self.log("Default model not found. Please load a model manually.")
    
    def browse_model(self):
        """Open file dialog to select a model file."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.load_model(file_path)
    
    def load_model(self, model_path):
        """Load the selected model."""
        try:
            self.log(f"Loading model from: {model_path}")
            self.model = load_model(model_path)
            self.log("Model loaded successfully!")
            self.model_status.configure(text=f"Loaded: {os.path.basename(model_path)}")
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.model_status.configure(text="Not loaded")
    
    def browse_video(self):
        """Open file dialog to select a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.output_path = os.path.join(
                self.test_videos_directory, 
                f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            self.log(f"Selected video: {file_path}")
            self.update_preview(file_path)
            self.enable_controls()
    
    def download_youtube(self):
        """Download video from YouTube URL."""
        url = self.youtube_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
            
        self.status_message.set("Downloading...")
        self.progress_bar.set(0.1)
        
        # Run download in a separate thread
        download_thread = threading.Thread(
            target=self._download_youtube_thread,
            args=(url,)
        )
        download_thread.daemon = True
        download_thread.start()
    
    def _download_youtube_thread(self, url):
        """Background thread for YouTube download."""
        try:
            output_filename = os.path.join(
                self.test_videos_directory, 
                f"youtube_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            
            self.log(f"Downloading from YouTube: {url}")
            
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_filename,
                'quiet': True,
                'progress_hooks': [self._download_progress_hook]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            self.video_path = output_filename
            self.output_path = os.path.join(
                self.test_videos_directory, 
                f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            
            self.log(f"Download complete: {output_filename}")
            
            # Update UI in main thread
            self.root.after(0, lambda: self.update_preview(output_filename))
            self.root.after(0, self.enable_controls)
            self.root.after(0, lambda: self.status_message.set("Download complete"))
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            
        except Exception as e:
            self.log(f"Error downloading video: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Download Error", 
                                                        f"Failed to download video: {str(e)}"))
            self.root.after(0, lambda: self.status_message.set("Download failed"))
    
    def _download_progress_hook(self, d):
        """Progress hook for youtube-dl."""
        if d['status'] == 'downloading':
            p = d.get('_percent_str', '0%')
            p = p.replace('%', '')
            try:
                progress = float(p) / 100
                self.root.after(0, lambda: self.progress_bar.set(progress))
                if int(progress * 100) % 10 == 0:  # Log every 10%
                    self.log(f"Download progress: {p}%")
            except:
                pass
    
    def update_preview(self, video_path):
        """Load video and display first frame as preview."""
        try:
            # Close previous capture if open
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception("Could not open video file")
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_index = 0
            
            # Read first frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_video_frame(frame)
                self.log(f"Video loaded: {os.path.basename(video_path)}")
                
                # Enable playback controls
                self.play_btn.configure(state="normal")
                self.stop_btn.configure(state="normal")
            else:
                raise Exception("Could not read video frame")
                
        except Exception as e:
            self.log(f"Error loading video preview: {str(e)}")
            messagebox.showerror("Preview Error", f"Failed to load video preview: {str(e)}")
    
    def display_video_frame(self, frame):
        """Convert OpenCV frame to tkinter compatible image and display it."""
        if self.display_frame is None:
            return
            
        # Convert frame to RGB (from BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate new dimensions to fit in the preview frame while maintaining aspect ratio
        preview_width = self.preview_frame.winfo_width()
        preview_height = self.preview_frame.winfo_height()
        
        if preview_width <= 1 or preview_height <= 1:
            # If the preview frame size is not yet determined, use default values
            preview_width = 640
            preview_height = 480
        
        h, w = rgb_frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(preview_width/w, preview_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized_frame = cv2.resize(rgb_frame, (new_w, new_h))
        
        # Convert to PhotoImage
        img = Image.fromarray(resized_frame)
        photo_img = ImageTk.PhotoImage(image=img)
        
        # If first time, replace the label text with an empty label
        if isinstance(self.display_frame, ctk.CTkLabel) and self.display_frame.cget("text") != "":
            self.display_frame.destroy()
            self.display_frame = ctk.CTkLabel(self.preview_frame, text="")
            self.display_frame.pack(expand=True, fill="both")
        
        # Update the image
        self.display_frame.configure(image=photo_img)
        self.display_frame.image = photo_img  # Keep a reference to prevent garbage collection
    
    def toggle_play(self):
        """Toggle video preview playback."""
        if self.is_preview_playing:
            self.is_preview_playing = False
            self.play_btn.configure(text="▶ Play")
        else:
            self.is_preview_playing = True
            self.play_btn.configure(text="⏸ Pause")
            self.play_preview()
    
    def play_preview(self):
        """Play the video preview."""
        if not self.is_preview_playing or self.cap is None:
            return
            
        # Read the next frame
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            self.display_video_frame(frame)
            self.frame_index += 1
            
            # Update progress
            progress = self.frame_index / self.total_frames if self.total_frames > 0 else 0
            self.progress_bar.set(progress)
            
            # Loop video at end
            if self.frame_index >= self.total_frames - 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                
            # Schedule next frame (using 30 fps playback)
            self.root.after(33, self.play_preview)
        else:
            # End of video
            self.is_preview_playing = False
            self.play_btn.configure(text="▶ Play")
    
    def stop_preview(self):
        """Stop video playback and reset to first frame."""
        self.is_preview_playing = False
        self.play_btn.configure(text="▶ Play")
        
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            
            # Read first frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_video_frame(frame)
                self.progress_bar.set(0)
    
    def process_video(self):
        """Process the video with activity recognition."""
        if not self.video_path:
            messagebox.showerror("Error", "No video selected")
            return
            
        if self.model is None:
            messagebox.showerror("Error", "No model loaded")
            return
            
        if self.processing:
            messagebox.showinfo("Info", "Processing already in progress")
            return
            
        # Change UI state
        self.processing = True
        self.status_message.set("Processing...")
        self.progress_bar.set(0)
        
        # Run processing in a separate thread
        process_thread = threading.Thread(
            target=self._process_video_thread
        )
        process_thread.daemon = True
        process_thread.start()
    
    def _process_video_thread(self):
        """Background thread for video processing."""
        try:
            self.log(f"Processing video: {self.video_path}")
            self.log(f"Output will be saved to: {self.output_path}")
            
            # Process video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
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
                    predicted_labels = self.model.predict(frames_array, verbose=0)
                    predicted_class = CLASSES_LIST[np.argmax(predicted_labels[0])]
                    confidence = np.max(predicted_labels[0]) * 100
                    
                    # Display prediction with confidence
                    text = f"{predicted_class} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(frame)
                
                # Update progress in main thread
                progress = frame_count / total_frames if total_frames > 0 else 0
                self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                
                frame_count += 1
                if frame_count % 30 == 0:
                    self.log(f"Processed {frame_count} frames")
                    
                    # Update preview occasionally
                    preview_frame = frame.copy()
                    self.root.after(0, lambda f=preview_frame: self.display_video_frame(f))
            
            cap.release()
            out.release()
            
            self.log(f"Processing complete! Output saved to: {self.output_path}")
            
            # Update UI in main thread
            self.root.after(0, lambda: self.status_message.set("Processing complete"))
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.update_preview(self.output_path))
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                        "Video processing complete!"))
                                                        
        except Exception as e:
            self.log(f"Error processing video: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Processing Error", 
                                                        f"Failed to process video: {str(e)}"))
            self.root.after(0, lambda: self.status_message.set("Processing failed"))
            
        finally:
            self.processing = False
    
    def save_output(self):
        """Save the processed output video to a user-selected location."""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showerror("Error", "No processed output available")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
            initialfile=os.path.basename(self.output_path)
        )
        
        if save_path:
            try:
                # Copy the file
                import shutil
                shutil.copy2(self.output_path, save_path)
                self.log(f"Output saved to: {save_path}")
                messagebox.showinfo("Success", f"Output saved to:\n{save_path}")
            except Exception as e:
                self.log(f"Error saving output: {str(e)}")
                messagebox.showerror("Save Error", f"Failed to save output: {str(e)}")
    
    def enable_controls(self):
        """Enable UI controls after loading a video."""
        self.play_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
    
    def log(self, message):
        """Add a message to the log text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Access text widget safely from any thread
        def update_log():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", log_message)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            
        if threading.current_thread() is threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)


if __name__ == "__main__":
    # Use CTk as root window class
    root = ctk.CTk()
    app = VideoActivityRecognizerApp(root)
    root.mainloop()