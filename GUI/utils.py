import cv2
from PIL import Image, ImageTk

def log_message(log_text, message):
    log_text.insert("end", message + "\n")
    log_text.see("end")

def check_queue(status_queue, status_label, log_text):
    try:
        while True:
            message = status_queue.get_nowait()
            status_label.configure(text=f"Status: {message}")
            log_message(log_text, message)
    except queue.Empty:
        pass

def play_video(gui, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        gui.status_queue.put("Error: Could not open video for playback.")
        return
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            gui.video_label.config(image=img)
            gui.video_label.image = img
            gui.app.after(33, update_frame)  # 30 FPS
        else:
            cap.release()
    
    update_frame()