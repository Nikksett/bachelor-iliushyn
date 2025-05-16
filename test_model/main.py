import os
import platform
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Thread
from screeninfo import get_monitors
from pathlib import Path
from PIL import Image, ImageTk
import ctypes



class YOLOv8App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Аналізатор")
        self.root.geometry("500x200")
        self.root.configure(bg="#FFFFFF")
        try:
            ico = Path(__file__).parent / "icon" / "AI.ico"
            self.root.iconbitmap(str(ico))
        except Exception:
            pass
        try:
            img = Image.open(Path(__file__).parent / "icon" / "image.png").resize((25, 25))
            self.image_icon = ImageTk.PhotoImage(img)
        except Exception:
            self.image_icon = None
        try:
            vid = Image.open(Path(__file__).parent / "icon" / "movie.png").resize((25, 25))
            self.video_icon = ImageTk.PhotoImage(vid)
        except Exception:
            self.video_icon = None

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 12, "bold"), background="#F5F5F5", foreground="#333333", borderwidth=1, padding=5)
        style.map("TButton",
                  background=[("active", "#DDDDDD"), ("pressed", "#CCCCCC")],
                  foreground=[("active", "#000000"), ("pressed", "#333333")])

        self.label = tk.Label(self.root, text="Оберіть файл для аналізу (зображення або відео)", font=("Arial", 14), bg="#FFFFFF", fg="#333333")
        self.label.pack(pady=10)

        button_frame = tk.Frame(self.root, bg="#FFFFFF")
        button_frame.pack()
        img_kwargs = {"image": self.image_icon, "compound": "left"} if self.image_icon else {}
        vid_kwargs = {"image": self.video_icon, "compound": "left"} if self.video_icon else {}

        self.select_image_button = ttk.Button(button_frame, text=" Обрати зображення", command=self.select_image, **img_kwargs)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=5)
        self.select_video_button = ttk.Button(button_frame, text=" Обрати відео", command=self.select_video, **vid_kwargs)
        self.select_video_button.grid(row=0, column=1, padx=10, pady=5)

        model_path = (Path(__file__).parent / ".." / "MILITARY_DETECTION_YOLO" / "miltech_yolov8n" / "weights" / "best.pt").resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        self.model = YOLO(str(model_path))

        self.color_map = {
            "circle": (255, 30, 133),
            "cross": (0, 252, 197),
            "squares": (83, 0, 255),
            "triangles": (208, 254, 1),
        }

        monitor = get_monitors()[0]
        self.screen_width, self.screen_height = monitor.width, monitor.height

    def letterbox(self, image: np.ndarray, new_shape: tuple = (600, 600)) -> tuple:
        orig_h, orig_w = image.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / orig_h, new_w / orig_w)
        new_unpad = (int(orig_w * r), int(orig_h * r))
        resized = cv2.resize(image, new_unpad)
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, r, (left, top)

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(__file__),
            filetypes=[
                ("Зображення", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")),
                ("Усі файли", "*.*")
            ]
        )
        if file_path:
            self.file_path = file_path
            Thread(target=self.run_image_detection).start()

    def select_video(self) -> None:
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(__file__),
            filetypes=[
                ("Відеофайли", ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.flv")),
                ("Усі файли", "*.*")
            ]
        )
        if file_path:
            self.video_path = file_path
            Thread(target=self.run_video_detection).start()

    def run_image_detection(self) -> None:
        frame = cv2.imread(self.file_path)
        if frame is None:
            return
        padded, r, pad = self.letterbox(frame)
        results = self.model(padded)
        self.scale_results(results, r, pad)
        self.draw_boxes_with_labels(frame, results)
        self.display_image(frame)

    def run_video_detection(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            padded, r, pad = self.letterbox(frame)
            results = self.model(padded)
            self.scale_results(results, r, pad)
            self.draw_boxes_with_labels(frame, results)
            self.display_video_frame(frame, cap)
        cap.release()
        cv2.destroyAllWindows()

    def scale_results(self, results, r: float, pad: tuple) -> None:
        pad_x, pad_y = pad
        scaled = []
        for *box, conf, cls in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_x) / r
            y1 = (y1 - pad_y) / r
            x2 = (x2 - pad_x) / r
            y2 = (y2 - pad_y) / r
            scaled.append([x1, y1, x2, y2, conf, cls])
        results[0].boxes.data = torch.tensor(scaled, device=results[0].boxes.data.device)

    def draw_boxes_with_labels(self, image: np.ndarray, results) -> None:
        font_scale = min(image.shape[:2]) / 1000
        font_thickness = max(1, int(font_scale * 2))
        text_positions = []
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = map(float, result)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = results[0].names[int(cls)].lower()
            color = self.color_map.get(label, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {int(conf * 100)}%"
            w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            xt, yt = x1, y1 - 2
            if yt - h < 0:
                yt = y2 + h + 2
            for tx, ty, tw, th in text_positions:
                if abs(xt - tx) < (w + tw) // 2 and abs(yt - ty) < (h + th) // 2:
                    yt -= (h + 5)
            cv2.rectangle(image, (xt, yt - h - 2), (xt + w + 5, yt), color, -1)
            cv2.putText(image, text, (xt, yt), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            text_positions.append((xt, yt, w, h))

    def display_image(self, image: np.ndarray) -> None:
        if platform.system() == "Windows":
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
        sw, sh = self.screen_width * 0.8, self.screen_height * 0.8
        scale = min(sw / image.shape[1], sh / image.shape[0])
        disp = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imshow("AI", disp)
        cv2.waitKey(0)
        if platform.system() == "Windows":
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)
        cv2.destroyAllWindows()

    def display_video_frame(self, frame: np.ndarray, cap) -> None:
        if platform.system() == "Windows":
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 6)
        sw, sh = self.screen_width * 0.8, self.screen_height * 0.8
        scale = min(sw / frame.shape[1], sh / frame.shape[0])
        disp = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        cv2.imshow("AI", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            if platform.system() == "Windows":
                ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.mainloop()
