import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from threading import Thread
import queue
from video_processor import process_video_thread
from model_loader import load_model
from utils import log_message, check_queue, play_video

class VideoProcessorGUI:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Video Action Recognition")
        self.app.geometry("1000x800")
        
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configuration
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = 64, 64
        self.SEQUENCE_LENGTH = 20
        self.CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
        
        # Create queue for thread communication
        self.status_queue = queue.Queue()
        
        self.setup_gui()
        self.load_model()
        
        # Start queue checker
        self.app.after(100, self.check_queue)
    
    def setup_gui(self):
        # Create main container
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Video Action Recognition",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)
        
        # Input Frame
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill="x", padx=20, pady=10)
        
        # Source Selection
        self.source_var = ctk.StringVar(value="youtube")
        self.youtube_radio = ctk.CTkRadioButton(
            self.input_frame, text="YouTube", variable=self.source_var, value="youtube"
        )
        self.youtube_radio.pack(anchor="w", pady=(10, 5))
        self.local_radio = ctk.CTkRadioButton(
            self.input_frame, text="Local Storage", variable=self.source_var, value="local"
        )
        self.local_radio.pack(anchor="w", pady=(10, 5))
        
        # URL Input
        self.url_label = ctk.CTkLabel(self.input_frame, text="YouTube URL:")
        self.url_label.pack(anchor="w", pady=(10, 5))
        
        self.url_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Enter YouTube URL",
            width=400
        )
        self.url_entry.pack(fill="x", pady=(0, 10))
        
        # Local File Input
        self.local_file_label = ctk.CTkLabel(self.input_frame, text="Local File:")
        self.local_file_label.pack(anchor="w", pady=(10, 5))
        
        self.local_file_frame = ctk.CTkFrame(self.input_frame)
        self.local_file_frame.pack(fill="x", pady=(0, 10))
        
        self.local_file_entry = ctk.CTkEntry(
            self.local_file_frame,
            placeholder_text="Select local video file"
        )
        self.local_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.local_file_button = ctk.CTkButton(
            self.local_file_frame,
            text="Browse",
            width=100,
            command=self.browse_local_file
        )
        self.local_file_button.pack(side="right")
        
        # Path Selection Frame
        self.paths_frame = ctk.CTkFrame(self.main_frame)
        self.paths_frame.pack(fill="x", padx=20, pady=10)
        
        # Output Directory
        self.output_label = ctk.CTkLabel(self.paths_frame, text="Output Directory:")
        self.output_label.pack(anchor="w", pady=(10, 5))
        
        self.output_frame = ctk.CTkFrame(self.paths_frame)
        self.output_frame.pack(fill="x", pady=(0, 10))
        
        self.output_entry = ctk.CTkEntry(
            self.output_frame,
            placeholder_text="Select output directory"
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.output_entry.insert(0, "test_videos")
        
        self.output_button = ctk.CTkButton(
            self.output_frame,
            text="Browse",
            width=100,
            command=self.browse_output
        )
        self.output_button.pack(side="right")
        
        # Model Path
        self.model_label = ctk.CTkLabel(self.paths_frame, text="Model Path:")
        self.model_label.pack(anchor="w", pady=(10, 5))
        
        self.model_frame = ctk.CTkFrame(self.paths_frame)
        self.model_frame.pack(fill="x", pady=(0, 10))
        
        self.model_entry = ctk.CTkEntry(
            self.model_frame,
            placeholder_text="Select model file"
        )
        self.model_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.model_entry.insert(0, "convlstm_model_2025_02_02__18_39_25.h5")
        
        self.model_button = ctk.CTkButton(
            self.model_frame,
            text="Browse",
            width=100,
            command=self.browse_model
        )
        self.model_button.pack(side="right")
        
        # Status Frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=20, pady=10)
        
        # Status Label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Ready",
            font=ctk.CTkFont(weight="bold")
        )
        self.status_label.pack(pady=10)
        
        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(fill="x", pady=10)
        self.progress_bar.set(0)
        
        # Process Button
        self.process_button = ctk.CTkButton(
            self.main_frame,
            text="Process Video",
            command=self.start_processing,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.process_button.pack(pady=20)
        
        # Log Area
        self.log_label = ctk.CTkLabel(self.main_frame, text="Processing Log:")
        self.log_label.pack(anchor="w", padx=20, pady=(20, 5))
        
        self.log_text = ctk.CTkTextbox(
            self.main_frame,
            height=200,
            font=ctk.CTkFont(family="Courier", size=12)
        )
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Video Playback Frame
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.video_label = ctk.CTkLabel(self.video_frame)
        self.video_label.pack()
    
    def load_model(self):
        try:
            model_path = self.model_entry.get()
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.log_message("Model loaded successfully")
            else:
                self.log_message("Error: Model file not found")
        except Exception as e:
            self.log_message(f"Error loading model: {e}")
    
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, directory)
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, file_path)
            self.load_model()
    
    def browse_local_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.local_file_entry.delete(0, "end")
            self.local_file_entry.insert(0, file_path)
    
    def log_message(self, message):
        log_message(self.log_text, message)
    
    def check_queue(self):
        check_queue(self.status_queue, self.status_label, self.log_text)
        self.app.after(100, self.check_queue)
    
    def start_processing(self):
        source = self.source_var.get()
        if source == "youtube":
            url = self.url_entry.get()
            if not url:
                messagebox.showerror("Error", "Please enter a YouTube URL")
                return
            self.video_path = None
        else:
            video_path = self.local_file_entry.get()
            if not video_path:
                messagebox.showerror("Error", "Please select a local video file")
                return
            self.video_path = video_path
        
        self.process_button.configure(state="disabled")
        self.progress_bar.start()
        
        # Start processing in a separate thread
        thread = Thread(target=process_video_thread, args=(self,), daemon=True)
        thread.start()
    
    def run(self):
        self.app.mainloop()

def main():
    app = VideoProcessorGUI()
    app.run()

if __name__ == "__main__":
    main()