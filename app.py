import cv2
import mediapipe as mp
import time
from flask import Flask, render_template, Response, jsonify
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# --- INISIALISASI MODEL ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Global variable untuk sinkronisasi data ke Frontend
current_gestures = {
    "left_hand": {"gesture": "None", "confidence": 0},
    "right_hand": {"gesture": "None", "confidence": 0}
}

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         #Jari Jempol
    (0,5), (5,6), (6,7), (7,8),         #Jari Telunjuk
    (0,9), (9,10), (10,11), (11,12),    #Jari Tengah
    (0,13), (13,14), (14,15), (15,16),  #Jari Manis
    (0,17), (17,18), (18,19), (19,20),  #Jari Kelingking
    (5,9), (9,13), (13,17)              #Pangkal Jari/Telapak Tangan
]

def recognize_gesture(hand_landmarks, handedness):
    fingers = []
    # Cek 4 jari (Telunjuk, Tengah, Manis, Kelingking)
    for i in [8, 12, 16, 20]:
        if hand_landmarks[i].y < hand_landmarks[i-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Logika Jempol (Disesuaikan dengan status Handedness yang sudah dikoreksi)
    is_right_hand = (handedness == "Right")
    thumb_open = 1
    if is_right_hand:
        # Tangan kanan: Jempol buka jika koordinat x titik 4 < titik 3 (karena sudah di-flip)
        if hand_landmarks[4].x > hand_landmarks[3].x: thumb_open = 0
    else:
        # Tangan kiri: kebalikannya
        if hand_landmarks[4].x < hand_landmarks[3].x: thumb_open = 0

    # --- LOGIKA GESTURE ---
    if hand_landmarks[4].y < hand_landmarks[3].y and sum(fingers) == 0 and thumb_open == 1:
        return "Thumbs Up"
    if hand_landmarks[4].y > hand_landmarks[3].y and sum(fingers) == 0 and thumb_open == 1:
        return "Thumbs Down"
    if fingers == [1, 0, 0, 0] and thumb_open == 0:
        return "Pointing Up"
    if fingers == [0, 1, 0, 0] and thumb_open == 0:
        return "Middle Finger"
    if fingers == [0, 0, 1, 0] and thumb_open == 0:
        return "Ring Finger"
    if fingers == [0, 0, 0, 1] and thumb_open == 0:
        return "Pinky Finger"
    if sum(fingers) == 0 and thumb_open == 0:
        return "Fist"
    if sum(fingers) == 4 and thumb_open == 1:
        return "Hello"
    if fingers == [1, 1, 0, 0] and thumb_open == 0:
        return "Peace Sign"
    if fingers == [1, 0, 0, 1] and thumb_open == 1:
        return "I Love You"
    if fingers == [1, 0, 0, 1] and thumb_open == 0:
        return "Metal"
    if fingers == [0, 1, 1, 1]:
        return "Okay"
    
    thumb_pinky = abs(hand_landmarks[4].x - hand_landmarks[20].x)
    if thumb_pinky > 0.2 and fingers == [0, 0, 0, 1]:
        return "Call Me"
    
    return "Mendeteksi..."


# --- FUNGSI BUAT NAMPILIN GAMBAR LANDMARK & TEKS ---
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        # FLIP FRAME (Efek Cermin) agar user tidak bingung
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # --- MULAI HITUNG WAKTU INFERENSI ---
        start_time = time.time()
        results = detector.detect(mp_image)
        end_time = time.time()
        

        # --- SELESAI HITUNG WAKTU INFERENSI ---
        # Hitung durasi dalam milidetik dan FPS
        inference_time_ms = (end_time - start_time) * 1000
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # Reset data sementara
        current_gestures["left_hand"] = {"gesture": "None", "confidence": 0}
        current_gestures["right_hand"] = {"gesture": "None", "confidence": 0}

        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # KOREKSI TANGAN: Karena frame di-flip, MediaPipe akan mendeteksi terbalik.
                # Jika MP bilang "Left", sebenarnya itu "Right" di mata user yang sudah di-flip.
                mp_handedness = results.handedness[idx][0].category_name
                actual_handedness = "Right" if mp_handedness == "Left" else "Left"
                
                gesture_text = recognize_gesture(hand_landmarks, actual_handedness)
                score = int(results.handedness[idx][0].score * 100)

                # Update status untuk API (Sidebar HTML)
                key = "left_hand" if actual_handedness == "Left" else "right_hand"
                current_gestures[key] = {"gesture": gesture_text, "confidence": score}

                # Gambar Landmarks
                for start, end in HAND_CONNECTIONS:
                    p1, p2 = hand_landmarks[start], hand_landmarks[end]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 255), 2)
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)

                # Label di atas tangan dalam video
                cv2.putText(frame, f"{actual_handedness}: {gesture_text}", 
                            (int(hand_landmarks[0].x*w), int(hand_landmarks[0].y*h)-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Info Inferensi & FPS
        cv2.putText(frame, f"Inf: {inference_time_ms:.1f}ms | FPS: {int(fps)}", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- AKSES INDEX.HTML ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture')
def get_gesture():
    return jsonify(current_gestures)


# --- AKSES KAMERA JIKA START ---
@app.route('/start')
def start(): return "OK"

# --- AKSES KAMERA JIKA STOP ---
@app.route('/stop')
def stop(): return "OK"

if __name__ == '__main__':
    app.run(debug=True, port=5000)