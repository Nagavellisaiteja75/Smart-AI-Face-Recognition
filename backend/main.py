import csv
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_FOLDER = PROJECT_ROOT / "frontend"
FACES_FOLDER = PROJECT_ROOT / "data" / "faces"
ATTENDANCE_FILE = PROJECT_ROOT / "data" / "attendance.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="SmartFace", version="2.0.0")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def normalize_name(name: str) -> tuple[str, str]:
    display_name = " ".join(name.split())
    folder_name = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")

    if len(display_name) < 2 or not folder_name:
        raise HTTPException(status_code=422, detail="Enter a name with at least two characters.")

    return display_name, folder_name


def get_faces(gray_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return list(faces)


def get_largest_face(gray_image: np.ndarray) -> np.ndarray | None:
    faces = get_faces(gray_image)
    if not faces:
        return None

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    face = gray_image[y : y + height, x : x + width]
    return cv2.resize(face, (200, 200))


def get_registered_people() -> list[dict[str, int | str]]:
    FACES_FOLDER.mkdir(parents=True, exist_ok=True)
    people = []

    for folder in sorted(FACES_FOLDER.iterdir()):
        if not folder.is_dir():
            continue

        samples = sum(
            image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            for image_path in folder.iterdir()
        )
        if samples:
            people.append(
                {
                    "name": folder.name.replace("-", " ").title(),
                    "samples": samples,
                }
            )

    return people


def train_recognizer():
    if not hasattr(cv2, "face"):
        raise HTTPException(
            status_code=500,
            detail="OpenCV face recognition is unavailable. Reinstall the project dependencies.",
        )

    training_faces = []
    training_labels = []
    label_to_name = {}

    for label, person in enumerate(get_registered_people()):
        person_folder = FACES_FOLDER / str(person["name"]).lower().replace(" ", "-")
        usable_samples = 0

        for image_path in person_folder.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            face = get_largest_face(image)
            if face is not None:
                training_faces.append(face)
                training_labels.append(label)
                usable_samples += 1

        if usable_samples:
            label_to_name[label] = str(person["name"])

    if not training_faces:
        raise HTTPException(
            status_code=409,
            detail="No usable face samples found. Add a clear, front-facing photo for a person.",
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(training_labels))
    return recognizer, label_to_name


def save_attendance(name: str, confidence: float) -> str:
    ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = ATTENDANCE_FILE.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with ATTENDANCE_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Date and Time", "Confidence"])
        writer.writerow([name, timestamp, f"{confidence:.1f}%"])

    return timestamp


def get_attendance(limit: int = 8) -> list[dict[str, str]]:
    if not ATTENDANCE_FILE.exists():
        return []

    with ATTENDANCE_FILE.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    return list(reversed(records[-limit:]))


def decode_image(image_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    return frame


@app.get("/api/health")
def health():
    people = get_registered_people()
    return {
        "status": "ready",
        "registered_people": len(people),
        "people": people,
    }


@app.get("/api/attendance")
def attendance():
    return {"records": get_attendance()}


@app.post("/api/people", status_code=201)
async def register_person(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
):
    display_name, folder_name = normalize_name(name)
    if not images:
        raise HTTPException(status_code=422, detail="Choose at least one photo.")

    person_folder = FACES_FOLDER / folder_name
    person_folder.mkdir(parents=True, exist_ok=True)
    saved_images = 0

    for image in images[:10]:
        image_bytes = await image.read()
        frame = decode_image(image_bytes)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if get_largest_face(gray_frame) is None:
            continue

        encoded, buffer = cv2.imencode(".jpg", frame)
        if encoded:
            image_path = person_folder / f"{uuid.uuid4().hex}.jpg"
            image_path.write_bytes(buffer.tobytes())
            saved_images += 1

    if not saved_images:
        raise HTTPException(
            status_code=422,
            detail="No clear face was detected in the selected photo. Try a front-facing image.",
        )

    return {
        "name": display_name,
        "saved_images": saved_images,
        "message": f"{display_name} is ready for recognition.",
    }


@app.post("/api/recognize")
async def recognize_face(image: UploadFile = File(...)):
    frame = decode_image(await image.read())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_faces(gray_frame)
    detected_face = get_largest_face(gray_frame)

    if detected_face is None:
        return {
            "match": False,
            "detected_faces": 0,
            "message": "No face detected. Use a clear, front-facing image.",
        }

    recognizer, label_to_name = train_recognizer()
    label, distance = recognizer.predict(detected_face)
    confidence = max(0, min(100, 100 - float(distance)))

    if distance > 65 or label not in label_to_name:
        return {
            "match": False,
            "confidence": round(confidence, 1),
            "detected_faces": len(faces),
            "message": "Face not recognized. Add more clear samples and try again.",
        }

    person_name = label_to_name[label]
    timestamp = save_attendance(person_name, confidence)
    return {
        "match": True,
        "name": person_name,
        "confidence": round(confidence, 1),
        "detected_faces": len(faces),
        "timestamp": timestamp,
        "message": "Identity verified and attendance recorded.",
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_FOLDER), html=True), name="frontend")
