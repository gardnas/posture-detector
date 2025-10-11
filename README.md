# Smart Posture Detector

A real-time posture monitoring web application that uses computer vision to help maintain healthy posture while working. Built with Flask, OpenCV, and MediaPipe for body landmark detection.

## Features

- **Personalized Calibration** - Calibrate the system with your own good and bad posture positions for accurate detection
- **Real-Time Video Monitoring** - Live webcam feed with pose detection overlay showing shoulder angles
- **Session Statistics** - Track good vs poor posture percentages, session duration, and slouch event counts
- **Interactive Dashboard** - Visual charts and real-time feedback on posture status

## Tech Stack

**Backend:** Python, Flask, OpenCV, MediaPipe, NumPy  
**Frontend:** HTML/CSS, JavaScript, Chart.js

## Prerequisites

- Python 3.8+
- Webcam
- Modern web browser

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/gardnas/posture-detector.git
cd posture-detector
```

2. Run the setup script
```bash
chmod +x setup.sh
./setup.sh
```

3. Start the application
```bash
source venv/bin/activate
python app.py
```

4. Open http://localhost:8080 in your browser

## Usage

1. **Calibrate** - Click "Start Calibration" and hold good posture for 5 seconds, then bad posture for 5 seconds
2. **Monitor** - Click "Start Monitoring" to begin tracking your posture in real-time
3. **Review** - Check the dashboard for statistics on your posture throughout the session

## How It Works

The application uses MediaPipe to detect body landmarks (ear, shoulder, hip) and calculates the angle at the shoulder. During calibration, it records your good and bad posture angles, then sets a threshold as the midpoint. During monitoring, if your current angle falls below this threshold, it flags poor posture.

The Flask backend streams video via MJPEG and provides a REST API for calibration and monitoring. The frontend updates statistics every 500ms and displays real-time feedback with color-coded indicators.

## License

MIT License

## Author

Spencer G - [@gardnas](https://github.com/gardnas)
