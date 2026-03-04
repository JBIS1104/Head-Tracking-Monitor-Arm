# Head-Tracking Monitor Arm (Face Tracking + Recognition)

This repository contains the core Python code for my Head-Tracking Monitor Arm project.

It combines:
- face recognition model training
- image validation/testing
- live recognition from an IP/MJPEG camera stream
- OpenCV fallback handling for MJPEG parsing when direct capture fails

## Repository Contents

- `detector.py` — main script for training, validation, image testing, and live recognition
- `live_import_display.py` — streamlined live recognition script with MJPEG fallback
- `Import Display.py` — additional display/import helper script
- `requirements.txt` — Python dependencies
- `training/` — labeled training images (`person_name/*.jpg`)
- `validation/` — validation images
- `output/` — generated model output (encodings)
- `unknown.jpg` — sample test image

## Tech Stack

- Python 3
- OpenCV (`cv2`)
- `face_recognition` / `dlib`
- NumPy
- Pillow
- Requests
- pigpio (for hardware-side integration)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Layout

Put training images in this format:

```text
training/
  person_a/
    img1.jpg
    img2.jpg
  person_b/
    img1.jpg
```

## Main Usage (`detector.py`)

### 1) Train encodings

```bash
python detector.py --train -m hog
```

This creates/updates:
- `output/encodings.pkl`

### 2) Validate model

```bash
python detector.py --validate -m hog
```

### 3) Test with a single image

```bash
python detector.py --test -f unknown.jpg -m hog
```

### 4) Live recognition from IP camera

```bash
python detector.py --live -m hog
```

Press `q` to quit the live window.

## Live Stream Notes

The scripts use camera URL + credentials defined in code (`IP_URL`, `IP_USERNAME`, `IP_PASSWORD`).

If OpenCV cannot open the stream directly, the code automatically falls back to MJPEG frame parsing (`requests` + JPEG boundary scanning).

## Performance Notes

- `hog` model: faster on CPU (recommended default)
- `cnn` model: more accurate but requires stronger GPU setup
- Frames are resized to 25% in live mode for faster processing

## Important

- `output/encodings.pkl` must exist before running live recognition
- Run training first if you see: `Encodings file not found. Run --train first.`

## Author

Junbyung Park
