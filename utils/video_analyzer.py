import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
from PIL import Image

# Load pre-trained Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    OCR_FRAME_INTERVAL = 10

    brightness_list = []
    focus_list = []
    visible_texts = []
    face_frames = 0
    scene_changes = 0
    last_hist = None
    best_thumbnail_frame = None
    best_frame_score = -np.inf

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % OCR_FRAME_INTERVAL == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_image)
            if text.strip():
                visible_texts.append(text.strip())

        # Brightness
        brightness = gray.mean()
        brightness_list.append(brightness)

        # Focus (sharpness)
        focus = variance_of_laplacian(gray)
        focus_list.append(focus)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            face_frames += 1

        # Scene change detection via histogram comparison
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if last_hist is not None:
            correlation = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < 0.8:  # Scene changed
                scene_changes += 1
        last_hist = hist

        # Best thumbnail frame (bright + focused + face)
        score = brightness + focus + (100 if len(faces) > 0 else 0)
        if score > best_frame_score:
            best_frame_score = score
            best_thumbnail_frame = frame.copy()

    avg_brightness = np.mean(brightness_list)
    avg_focus = np.mean(focus_list)
    face_percentage = (face_frames / frame_count) * 100 if frame_count > 0 else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_filename = f"best_frame_{timestamp}.jpg"
    frame_output_path = os.path.join("static", frame_filename)

    if best_thumbnail_frame is not None:
        cv2.imwrite(frame_output_path, best_thumbnail_frame)
    
    combined_text = "\n".join(visible_texts)

    cap.release()
    
    return {
        "total_frames": frame_count,
        "average_brightness": round(avg_brightness, 1),
        "average_focus": round(avg_focus, 1),
        "face_frame_percentage": round(face_percentage, 1),
        "scene_changes": scene_changes,
        "best_frame_filename": frame_filename,
        "visible_text": combined_text,
    }

