# server.py
import os
import sys
import time
import json
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from threading import Thread

# Add project path for your modules
sys.path.append('../')

# Import your gesture model and utilities
from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
import utils as ut

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Define a global variable for gesture commands (or use socketio.emit)
def send_gesture(gesture):
    # Broadcast the gesture to all connected clients
    socketio.emit('gesture_command', {'gesture': gesture})
    print("Sent gesture:", gesture)

# Route for the demo page (if needed)
@app.route('/')
def index():
    # Optionally render a page or simply return a message.
    return "Gesture Control Server Running."

# (Optional) if you already have a route to serve webcam video:
@app.route('/webcam')
def webcam_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# The modified generate_video function now sends gestures via SocketIO
def generate_video():
    ################################################### VARIABLES INITIALIZATION ###########################################################
    WIDTH = 1028 // 2
    HEIGHT = 720 // 2
    cap = ut.cv.VideoCapture(0)
    cap.set(ut.cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(ut.cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    TRAINING_KEYPOINTS = [i for i in range(0, 21, 4)]
    SMOOTH_FACTOR = 6
    PLOCX, PLOCY = 0, 0
    CLOX, CLOXY = 0, 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils

    GESTURE_RECOGNIZER_PATH = '../models/model.pth'
    model.load_state_dict(ut.torch.load(GESTURE_RECOGNIZER_PATH))
    LABEL_PATH = '../data/label.csv'
    labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()

    CONF_THRESH = 0.9
    GESTURE_HISTORY = ut.deque([])
    GEN_COUNTER = 0

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)
    ABSENCE_COUNTER = 0
    ABSENCE_COUNTER_THRESH = 20

    SHAPE_PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    SLEEP_COUNTER = 0
    SLEEP_COUNTER_THRESH = 20
    EAR_THRESH = 0.21
    EAR_HISTORY = ut.deque([])

    # For cooldown on Play/Pause gesture:
    COOLDOWN_PERIOD = 1.0
    last_toggle_time = 0

    ################################################### INITIALIZATION END ###########################################################
    while True:
        current_time = time.time()
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = ut.cv.flip(frame, 1)
        frame_rgb = ut.cv.cvtColor(frame, ut.cv.COLOR_BGR2RGB)
        det_zone, m_zone = ut.det_mouse_zones(frame)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coordinates_list = ut.calc_landmark_coordinates(frame_rgb, hand_landmarks)
                important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]
                preprocessed = ut.pre_process_landmark(important_points)
                d0 = ut.calc_distance(coordinates_list[0], coordinates_list[5])
                pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
                distances = ut.normalize_distances(d0, ut.get_all_distances(pts_for_distances))
                features = ut.np.concatenate([preprocessed, distances])
                conf, pred = ut.predict(features, model)
                gesture = labels[pred]

                if ut.cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= CONF_THRESH:
                    gest_hist = ut.track_history(GESTURE_HISTORY, gesture)
                    before_last = gest_hist[-2] if len(gest_hist) >= 2 else gest_hist[0]

                    # Instead of executing pyautogui, we now send the gesture via WebSocket
                    if gesture == 'Play_Pause':
                        if current_time - last_toggle_time > COOLDOWN_PERIOD:
                            send_gesture('Play_Pause')
                            last_toggle_time = current_time
                            GESTURE_HISTORY.clear()
                    elif gesture == 'Right_click' and before_last != 'Right_click':
                        send_gesture('Right_click')
                    elif gesture == 'Left_click' and before_last != 'Left_click':
                        send_gesture('Left_click')
                    elif gesture == 'Move_mouse':
                        x, y = ut.mouse_zone_to_screen(coordinates_list[9], m_zone)
                        # For mouse movement, you might want to send coordinates
                        send_gesture(json.dumps({'gesture': 'Move_mouse', 'x': x, 'y': y}))
                    elif gesture == 'Vol_up_gen':
                        send_gesture('Vol_up_gen')
                    elif gesture == 'Vol_down_gen':
                        send_gesture('Vol_down_gen')
                    elif gesture == 'Vol_up_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            send_gesture('Vol_up_ytb')
                    elif gesture == 'Vol_down_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            send_gesture('Vol_down_ytb')
                    elif gesture == 'Forward':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            send_gesture('Forward')
                    elif gesture == 'Backward':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            send_gesture('Backward')
                    elif gesture == 'fullscreen' and before_last != 'fullscreen':
                        send_gesture('fullscreen')
                    elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
                        send_gesture('Cap_Subt')
                    elif gesture == 'Neutral':
                        GEN_COUNTER = 0

                    ut.cv.putText(frame, f'{gesture} | {conf: .2f}', (int(WIDTH*0.05), int(HEIGHT*0.07)),
                                  ut.cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, ut.cv.LINE_AA)
        
        # (For brevity, sleepiness and absence detection sections remain unchanged.)
        # Encode the frame and yield it for the webcam feed.
        _, buffer = ut.cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Start gesture processing in a background thread
def start_video_feed():
    thread = Thread(target=lambda: list(generate_video()))
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    # Optionally, start the video feed thread (if you want to use the /webcam endpoint)
    start_video_feed()
    # Run the server with SocketIO
    socketio.run(app, debug=True)
