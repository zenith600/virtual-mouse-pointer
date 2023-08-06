import cv2
import mediapipe as mp
import pyautogui
import keyboard

pyautogui.FAILSAFE = False
num_screens = pyautogui.screenshot().size


head_scaling_factor = 10.0
cursor_scaling_factor = 130.0
cursor_smoothing_alpha = 0.03
cursor_deadzone = 2


LEFT_EYE_CLOSED_THRESHOLD = 1
RIGHT_CLICK_COUNTER_THRESHOLD = 0


mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
def euclidean_distance(pt1, pt2):
    return ((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2) ** 0.5


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=False)

prev_head_x, prev_head_y = None, None
cursor_x, cursor_y = pyautogui.position()
right_click_counter = 0

while True:

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        head_landmarks = [landmarks[10], landmarks[33], landmarks[152], landmarks[263]]
        for id, landmark in enumerate(head_landmarks):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                if prev_head_x is not None and prev_head_y is not None:
                    delta_x = (landmark.x - prev_head_x) * frame_w * head_scaling_factor
                    delta_y = (landmark.y - prev_head_y) * frame_h * head_scaling_factor
                    if abs(delta_x) < cursor_deadzone:
                        delta_x = 2
                    if abs(delta_y) < cursor_deadzone:
                        delta_y = 2
                    cursor_vx = int(delta_x * cursor_scaling_factor)
                    cursor_vy = int(delta_y * cursor_scaling_factor)
                    cursor_x = int((1 - cursor_smoothing_alpha) * cursor_x + cursor_smoothing_alpha * (cursor_x + cursor_vx))
                    cursor_y = int((1 - cursor_smoothing_alpha) * cursor_y + cursor_smoothing_alpha * (cursor_y + cursor_vy))
                    pyautogui.moveTo(cursor_x, cursor_y)
                prev_head_x, prev_head_y = landmark.x, landmark.y

    cv2.imshow('Head Controlled Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('z'):
        break


cam.release()
cv2.destroyAllWindows()