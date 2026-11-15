import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- INISIALISASI MODEL ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         #Jari Jempol
    (0,5), (5,6), (6,7), (7,8),         #Jari Telunjuk
    (0,9), (9,10), (10,11), (11,12),    #Jari Tengah
    (0,13), (13,14), (14,15), (15,16),  #Jari Manis
    (0,17), (17,18), (18,19), (19,20),  #Jari Kelingking
    (5,9), (9,13), (13,17)              #Pangkal Jari/Telapak Tangan
]

# --- FUNGSI BUAT NGENALIN GESTURE ---
def recognize_gesture(hand_landmarks):
    # Landmark ujung jari: 4(Jempol), 8(Telunjuk), 12(Tengah), 16(Manis), 20(Kelingking)

    fingers = []
    # Cek 4 jari (Telunjuk, Tengah, Manis, Kelingking) apakah terbuka
    for i in [8, 12, 16, 20]:
        if hand_landmarks[i].y < hand_landmarks[i-2].y:
            fingers.append(1) # Terbuka
        else:
            fingers.append(0) # Tertutup

    # Cek tangan kanan atau kiri berdasarkan posisi titik 5 (telunjuk) dan 17 (kelingking)
    is_right_hand = hand_landmarks[5].x > hand_landmarks[17].x
    thumb_open = 1
    if is_right_hand:
        # Tangan Kanan
        if hand_landmarks[4].x < hand_landmarks[3].x: thumb_open = 0
    else:
        # Tangan Kiri
        if hand_landmarks[4].x > hand_landmarks[3].x: thumb_open = 0


    # --- LOGIKA GESTURE ---
    # 1. Thumbs Up (Jempol di atas, jari lain tertutup)
    if hand_landmarks[4].y < hand_landmarks[3].y and sum(fingers) == 0 and thumb_open == 1:
        return "Thumbs Up"
    
    # 2. Thumbs Down (Jempol di bawah, jari lain tertutup)
    if hand_landmarks[4].y > hand_landmarks[3].y and sum(fingers) == 0 and thumb_open == 1:
        return "Thumbs Down"

    # 3. Pointing Up (Cuma telunjuk yg buka)
    if fingers == [1, 0, 0, 0] and thumb_open == 0:
        return "Pointing Up/Index Finger"
    
    '''
    # 4. Pointing Down 
    if fingers == [1, 0, 0, 0] and thumb_open == 0 and hand_landmarks[8].y > hand_landmarks[6].y:
        return "Pointing Down"
    '''
    
    # 4. Jari Tengah
    if fingers == [0, 1, 0, 0] and thumb_open == 0:
        return "Middle Finger"
    
    # 5. Jari Manis
    if fingers == [0, 0, 1, 0] and thumb_open == 0:
        return "Ring Finger"
    
    # 6. Jari Kelingking
    if fingers == [0, 0, 0, 1] and thumb_open == 0:
        return "Pinky Finger"

    # 7. Fist (Semua jari tertutup)
    if sum(fingers) == 0 and thumb_open == 0:
        return "Fist"

    # 8. Hello/Open Hand (Semua jari terbuka)
    if sum(fingers) == 4 and thumb_open == 1:
        return "Hello"

    # 9. Peace Sign (Telunjuk & Tengah buka, lainnya tutup)
    if fingers == [1, 1, 0, 0] and thumb_open == 0:
        return "Peace Sign"
    
    # 10. ILoveU (Telunjuk & Kelingking buka, Tengah & Manis tutup)
    if fingers == [1, 0, 0, 1] and thumb_open == 1:
        return "I Love You"

    # 11. Metal (cuma Telunjuk & Kelingking yg buka)
    if fingers == [1, 0, 0, 1] and thumb_open == 0:
        return "Metal"
    
    # 12. OK 
    if fingers == [0, 1, 1, 1]:
        return "Okay"
    
    # 13. Call/Telepon (Jempol sama Telunjuk aja)
    thumb_pinky = abs(hand_landmarks[4].x - hand_landmarks[20].x) #Si jempol sama telunjuk nempel, klo ga pake ini dia bacanya malah kelingking
    if thumb_pinky > 0.2 and fingers == [0, 0, 0, 1]:
        return "Call Me"
    
    return "Mendeteksi...."

# --- FUNGSI BUAT NAMPILIN GAMBAR LANDMARK & TEKS ---
def detect_hand_gesture(image, detector_model):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector_model.detect(mp_image)

   # --- MULAI HITUNG WAKTU INFERENSI ---
    start_time = time.time() 
    results = detector_model.detect(mp_image)
    end_time = time.time()
    
    # --- SELESAI HITUNG WAKTU INFERENSI ---
    # Hitung durasi dalam milidetik dan FPS
    inference_time_ms = (end_time - start_time) * 1000
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            gesture_text = recognize_gesture(hand_landmarks)

            # Gambar Garis
            for start, end in HAND_CONNECTIONS:
                p1, p2 = hand_landmarks[start], hand_landmarks[end]
                cv2.line(image, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 0, 255), 2)

            # Gambar Titik
            for lm in hand_landmarks:
                cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)

            # Tampilkan Teks Gesture
            cv2.putText(image, gesture_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # --- TAMPILKAN INFO INFERENSI DI INTERFACE ---
    info_text = f"Inference: {inference_time_ms:.1f} ms | FPS: {int(fps)}"
    cv2.putText(image, info_text, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    return image

# --- AKSES KAMERA ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1) # Mirror
    frame = detect_hand_gesture(frame, detector)
    
    cv2.imshow("Hand Gesture Recognition", frame)   #Ini judul buat di kamera windows-nya
    
    if cv2.waitKey(1) & 0xFF == ord('q'):           #Kalo mau close pencet keyword Q
        break

cap.release()
cv2.destroyAllWindows()