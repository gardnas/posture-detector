import cv2
import mediapipe as mp
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# These are teh calibration angles
GOOD_POSTURE_ANGLE = None
BAD_POSTURE_ANGLE = None
CALIBRATED_THRESHOLD = None

def calculate_angle(a, b, c):
    """
    Calculates the angle formed by three points a, b, c (with b as the vertex).
    Points a, b, c are in (x, y) format.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    
    angle_radians = math.acos(dot_product / (mag_ba * mag_bc))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def calibrate_posture(cap, pose):
    """
    Guides the user through two calibration phases:
    1) Good Posture
    2) Bad Posture
    
    Then returns the average 'good' angle and 'bad' angle.
    """
    global GOOD_POSTURE_ANGLE, BAD_POSTURE_ANGLE
    
    # Helper function to gather angles over a period of time
    def gather_angles(label, seconds=5):
        angles = []
        start_time = time.time()
        while (time.time() - start_time) < seconds:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            cv2.putText(frame, f"Calibrating {label} posture...",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def to_pixel_coords(lm):
                    return int(lm.x * w), int(lm.y * h)
                
                # Right ear, shoulder, hip
                right_ear = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value])
                right_shoulder = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                right_hip = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                
                angle = calculate_angle(right_ear, right_shoulder, right_hip)
                angles.append(angle)
                
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # press the ESC key to quit at any time
                break
        
        if len(angles) > 0:
            return sum(angles)/len(angles)
        return None
    
    # Gather GOOD POSTURE angles
    print("Starting GOOD POSTURE calibration for ~5 seconds...")
    good_angle_avg = gather_angles(label="GOOD POSTURE", seconds=5)
    
    # Gather BAD POSTURE angles
    print("Starting BAD POSTURE calibration for ~5 seconds...")
    bad_angle_avg = gather_angles(label="BAD POSTURE", seconds=5)
    
    # Return the average angles
    GOOD_POSTURE_ANGLE = good_angle_avg
    BAD_POSTURE_ANGLE = bad_angle_avg
    print(f"[DEBUG] GOOD_POSTURE_ANGLE = {GOOD_POSTURE_ANGLE}, BAD_POSTURE_ANGLE = {BAD_POSTURE_ANGLE}")

def main():
    global CALIBRATED_THRESHOLD
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        # Step 1 & 2: Calibrate
        calibrate_posture(cap, pose)
        
        # Define a threshold based on the midpoint between GOOD and BAD posture angles
        # we can change this logic as you see fit
        if GOOD_POSTURE_ANGLE and BAD_POSTURE_ANGLE:
            CALIBRATED_THRESHOLD = (GOOD_POSTURE_ANGLE + BAD_POSTURE_ANGLE)/2
            print(f"[DEBUG] CALIBRATED_THRESHOLD = {CALIBRATED_THRESHOLD}")
        else:
            # Fallback/default value if calibration failed
            CALIBRATED_THRESHOLD = 15 
        
        # Close the calibration window
        cv2.destroyWindow('Calibration')
        
        # Step 3: Monitor posture
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
                
                def to_pixel_coords(lm):
                    return int(lm.x * w), int(lm.y * h)
                
                # Grab landmarks
                landmarks = results.pose_landmarks.landmark
                right_ear = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value])
                right_shoulder = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                right_hip = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                
                # Calculate angle
                angle_shoulder = calculate_angle(right_ear, right_shoulder, right_hip)
                
                # Compare angle to calibrated threshold
                if angle_shoulder < CALIBRATED_THRESHOLD:
                    cv2.putText(frame, "Slouching Detected!",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Good Posture",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Display angle info
                cv2.putText(frame, f"Angle: {angle_shoulder:.1f}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Smart Posture Detector - Monitoring', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit at any time
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
