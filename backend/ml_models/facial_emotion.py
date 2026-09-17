import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

IMG_SIZE = 48
FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TARGET_EMOTIONS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprised": "surprise",
    "scared": "fear",
}


def find_face(image_bgr):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return gray[y:y + h, x:x + w]


def preprocess_face(face_gray):
    resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_emotion(model, face_input):
    raw_probs = model.predict(face_input, verbose=0)[0]
    return {label: float(prob) for label, prob in zip(FER2013_CLASSES, raw_probs)}


def narrow_to_target_emotions(fer_probs):
    selected = {
        target_name: fer_probs[fer_name]
        for target_name, fer_name in TARGET_EMOTIONS.items()
    }
    total = sum(selected.values())
    if total == 0:
        return {k: 1.0 / len(selected) for k in selected}

    return {k: v / total for k, v in selected.items()}


class FacialEmotionDetector:
    def __init__(self):
        self.model_path = self._resolve_model_path()
        self.model = None

    def _resolve_model_path(self):
        project_root = Path(__file__).resolve().parents[2]
        candidates = [
            project_root / 'emotion_console' / 'emotion_console' / 'model' / 'emotion_model.h5',
            project_root / 'models' / 'trained' / 'facial_emotion.h5',
            project_root / 'backend' / 'ml_models' / 'trained' / 'facial_emotion.h5',
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return str(candidates[0])

    def _load_model(self):
        if self.model is not None:
            return self.model

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found at {self.model_path}')

        from tensorflow.keras.models import load_model

        self.model = load_model(self.model_path)
        return self.model

    def detect_from_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'error': 'Could not read the image file.',
                    'face_detected': False,
                }

            face = find_face(image_bgr)
            if face is None:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'error': 'No face detected in image.',
                    'face_detected': False,
                }

            model = self._load_model()
            face_input = preprocess_face(face)
            fer_probs = predict_emotion(model, face_input)
            all_emotions = narrow_to_target_emotions(fer_probs)
            top_emotion = max(all_emotions, key=all_emotions.get)
            confidence = float(all_emotions[top_emotion])

            return {
                'emotion': top_emotion,
                'confidence': confidence,
                'all_emotions': all_emotions,
                'face_detected': True,
            }
        except Exception as exc:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(exc),
                'face_detected': False,
            }


facial_detector = FacialEmotionDetector()
