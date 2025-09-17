# Wave Detector

Implementation of an algorithm for detecting a wave of the hand using YOLO object detection models and pattern recognition techniques.

---

## 🔍 What It Does

- Uses YOLO to detect a person’s hand (or hands) in a video or image feed.  
- Applies additional pattern recognition / logic to determine when a **wave** gesture is made.  
- Useful for gesture-controlled interfaces, human-computer interaction, or video analytics.

---

## 📂 Project Structure

Wave-Detector\
├── Wave_data # Dataset or example video/image data for training / testing \
├── Augmentation # Scripts or tools for image/video augmentation \
├── Work # Working scripts / experiments \
├── yolo-model  config / weights # YOLO model files, configuration, and pretrained weights \
└── README.md # (This file)

---

## ⚙️ Requirements & Setup

You’ll need:

- Python 3.x  
- YOLO object detection environment (e.g. `yolov5` or similar)  
- Required Python libraries: e.g. `opencv-python`, `numpy`, `torch` (for YOLO), etc.  

To set up:

```bash
git clone https://github.com/VuKe77/Wave-Detector.git
cd Wave-Detector

# (optional) set up a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
Run the HandDetection.py
Output will indicate when a wave gesture is detected (via console and overlay on video frames).
Detection is performed ONLINE! (Could be adjusted to do offline detection)

## 🧠 How It Works

The YOLO model detects bounding boxes around hands.
Then a pattern recognition component together with CSRT and SORT algorithms determines whether the detected hand(s) is doing a wave.

