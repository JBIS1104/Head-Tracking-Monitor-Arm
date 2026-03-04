import cv2
import numpy as np
import requests

# The URL of your iPad's IP camera stream (MJPEG)
url = "http://10.204.104.189:8081//video"

# If your IP cam requires credentials, set them here
username = "admin"
password = "admin"

if username and password:
    url = f"http://{username}:{password}@10.204.104.189:8081/video"

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def mjpeg_frames(stream_url):
    auth = (username, password) if username and password else None
    response = requests.get(stream_url, stream=True, timeout=5, auth=auth)
    response.raise_for_status()
    buffer = b""
    for chunk in response.iter_content(chunk_size=1024):
        buffer += chunk
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                yield frame


# Connect to the video stream (prefer FFMPEG for MJPEG over HTTP)
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# Retry a few times in case the stream needs a moment to initialize
for _ in range(5):
    if cap.isOpened():
        break
    cap.release()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

use_mjpeg_fallback = not cap.isOpened()
if use_mjpeg_fallback:
    print("OpenCV could not open the stream. Falling back to MJPEG parser...")

if use_mjpeg_fallback:
    frame_iter = mjpeg_frames(url)

    def next_frame():
        return True, next(frame_iter)

else:

    def next_frame():
        return cap.read()


while True:
    # Capture frame-by-frame
    ret, frame = next_frame()
    if not ret:
        print("Error: Failed to read frame from stream.")
        break

    # Convert to grayscale (faster for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("IP Camera Face Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()