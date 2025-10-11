from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import math
import time
import json
from threading import Lock
from datetime import datetime

app = Flask(__name__)
CORS(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global state
class PostureDetectorState:
    def __init__(self):
        self.camera = None
        self.pose = None
        self.good_posture_angle = None
        self.bad_posture_angle = None
        self.calibrated_threshold = None
        self.is_calibrating = False
        self.calibration_phase = None
        self.is_monitoring = False
        self.lock = Lock()
        
        # Statistics
        self.session_start = None
        self.total_time = 0
        self.good_posture_time = 0
        self.bad_posture_time = 0
        self.slouch_count = 0
        self.last_state = "good"
        self.current_angle = 0
        self.posture_history = []
        
state = PostureDetectorState()

def calculate_angle(a, b, c):
    """Calculate angle formed by three points."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    
    angle_radians = math.acos(max(-1, min(1, dot_product / (mag_ba * mag_bc))))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def initialize_camera():
    """Initialize camera and pose detector."""
    if state.camera is None:
        state.camera = cv2.VideoCapture(0)
        state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if state.pose is None:
        state.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    return state.camera is not None

def process_frame(frame, draw_landmarks=True):
    """Process frame and detect posture."""
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = state.pose.process(rgb_frame)
    
    angle = None
    posture_status = "unknown"
    
    if results.pose_landmarks:
        if draw_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        
        landmarks = results.pose_landmarks.landmark
        
        def to_pixel_coords(lm):
            return int(lm.x * w), int(lm.y * h)
        
        right_ear = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value])
        right_shoulder = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        right_hip = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        
        angle = calculate_angle(right_ear, right_shoulder, right_hip)
        state.current_angle = angle
        
        # Determine posture status
        if state.calibrated_threshold is not None:
            if angle < state.calibrated_threshold:
                posture_status = "bad"
            else:
                posture_status = "good"
    
    return frame, angle, posture_status

def generate_frames():
    """Generate frames for video streaming."""
    initialize_camera()
    
    while True:
        with state.lock:
            if state.camera is None:
                break
            
            success, frame = state.camera.read()
            if not success:
                break
            
            # Process frame
            frame, angle, posture_status = process_frame(frame)
            
            # Update statistics during monitoring
            if state.is_monitoring and posture_status != "unknown":
                current_time = time.time()
                if state.session_start is None:
                    state.session_start = current_time
                
                # Track posture changes
                if posture_status != state.last_state:
                    if posture_status == "bad" and state.last_state == "good":
                        state.slouch_count += 1
                    state.last_state = posture_status
                
                # Update time tracking
                elapsed = 1.0 / 30  # Approximate frame time
                state.total_time += elapsed
                if posture_status == "good":
                    state.good_posture_time += elapsed
                else:
                    state.bad_posture_time += elapsed
            
            # Add overlay text
            if state.is_calibrating:
                cv2.putText(frame, f"Calibrating {state.calibration_phase}...",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 0), 2, cv2.LINE_AA)
            elif state.is_monitoring:
                color = (0, 255, 0) if posture_status == "good" else (0, 0, 255)
                status_text = "Good Posture" if posture_status == "good" else "Slouching!"
                cv2.putText(frame, status_text,
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1, color, 2, cv2.LINE_AA)
                
                if angle is not None:
                    cv2.putText(frame, f"Angle: {angle:.1f}°",
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_calibration', methods=['POST'])
def start_calibration():
    """Start calibration process."""
    with state.lock:
        initialize_camera()
        state.is_calibrating = True
        state.is_monitoring = False
        state.calibration_phase = "GOOD POSTURE"
        return jsonify({"status": "success", "phase": "good"})

@app.route('/api/calibrate_good', methods=['POST'])
def calibrate_good():
    """Calibrate good posture."""
    data = request.json
    angles = data.get('angles', [])
    
    if len(angles) > 0:
        state.good_posture_angle = sum(angles) / len(angles)
        return jsonify({
            "status": "success",
            "angle": state.good_posture_angle,
            "next_phase": "bad"
        })
    
    return jsonify({"status": "error", "message": "No angles collected"}), 400

@app.route('/api/calibrate_bad', methods=['POST'])
def calibrate_bad():
    """Calibrate bad posture."""
    data = request.json
    angles = data.get('angles', [])
    
    if len(angles) > 0:
        state.bad_posture_angle = sum(angles) / len(angles)
        
        # Calculate threshold
        if state.good_posture_angle and state.bad_posture_angle:
            state.calibrated_threshold = (state.good_posture_angle + state.bad_posture_angle) / 2
        else:
            state.calibrated_threshold = 15
        
        state.is_calibrating = False
        
        return jsonify({
            "status": "success",
            "angle": state.bad_posture_angle,
            "threshold": state.calibrated_threshold,
            "good_angle": state.good_posture_angle
        })
    
    return jsonify({"status": "error", "message": "No angles collected"}), 400

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start posture monitoring."""
    with state.lock:
        if state.calibrated_threshold is None:
            return jsonify({"status": "error", "message": "Please calibrate first"}), 400
        
        # Reset statistics
        state.is_monitoring = True
        state.session_start = time.time()
        state.total_time = 0
        state.good_posture_time = 0
        state.bad_posture_time = 0
        state.slouch_count = 0
        state.last_state = "good"
        state.posture_history = []
        
        return jsonify({"status": "success"})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop posture monitoring."""
    with state.lock:
        state.is_monitoring = False
        return jsonify({"status": "success"})

@app.route('/api/current_angle')
def current_angle():
    """Get current angle reading."""
    return jsonify({
        "angle": state.current_angle,
        "is_calibrating": state.is_calibrating,
        "calibration_phase": state.calibration_phase
    })

@app.route('/api/statistics')
def statistics():
    """Get current session statistics."""
    good_percentage = 0
    bad_percentage = 0
    
    if state.total_time > 0:
        good_percentage = (state.good_posture_time / state.total_time) * 100
        bad_percentage = (state.bad_posture_time / state.total_time) * 100
    
    return jsonify({
        "is_monitoring": state.is_monitoring,
        "is_calibrated": state.calibrated_threshold is not None,
        "total_time": state.total_time,
        "good_posture_time": state.good_posture_time,
        "bad_posture_time": state.bad_posture_time,
        "good_percentage": good_percentage,
        "bad_percentage": bad_percentage,
        "slouch_count": state.slouch_count,
        "current_angle": state.current_angle,
        "threshold": state.calibrated_threshold,
        "good_angle": state.good_posture_angle,
        "bad_angle": state.bad_posture_angle
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the system."""
    with state.lock:
        state.is_calibrating = False
        state.is_monitoring = False
        state.good_posture_angle = None
        state.bad_posture_angle = None
        state.calibrated_threshold = None
        state.session_start = None
        state.total_time = 0
        state.good_posture_time = 0
        state.bad_posture_time = 0
        state.slouch_count = 0
        state.last_state = "good"
        
        return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
