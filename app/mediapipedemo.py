

from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# The script is inside app/, while the model is in the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "hand_landmarker.task"

CAMERA_INDEX = 0
OUTPUT_PATH = PROJECT_ROOT / "mediapipe_landmarks_screenshot.png"


# MediaPipe's 21-point hand skeleton connections.
HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


def draw_hand_landmarks(frame, result):
    """Draw detected hand landmarks and connections on the frame."""

    if result is None or not result.hand_landmarks:
        return frame

    height, width = frame.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Keep coordinates within the image.
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            points.append((x, y))

        # Draw the skeleton lines first.
        for start_index, end_index in HAND_CONNECTIONS:
            start_point = points[start_index]
            end_point = points[end_index]

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 180, 0),  # Blue/cyan in BGR
                3,
                cv2.LINE_AA,
            )

        # Draw the 21 landmark points.
        for point in points:
            cv2.circle(
                frame,
                point,
                6,
                (0, 255, 255),  # Yellow in BGR
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                point,
                8,
                (0, 80, 255),  # Orange border
                2,
                cv2.LINE_AA,
            )

    return frame


def open_camera():
    """Open the webcam using DirectShow on Windows, with a fallback."""

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    return camera


def main():
    if not TASK_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the MediaPipe model:\n{TASK_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(TASK_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    camera = open_camera()

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Close the FastAPI server or "
            "any other application currently using the camera."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    print("Camera opened.")
    print("Press S to save a screenshot.")
    print("Press Q or Esc to close.")

    last_display_frame = None
    start_time = time.perf_counter()

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a camera frame.")
                    break

                # Mirror the preview like a normal webcam.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                display_frame = draw_hand_landmarks(
                    frame.copy(),
                    result,
                )

                detected_hands = len(result.hand_landmarks)

                cv2.putText(
                    display_frame,
                    f"Hands detected: {detected_hands}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    "S: save screenshot    Q: close",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                last_display_frame = display_frame

                cv2.imshow(
                    "MediaPipe Hand Landmarks",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    if last_display_frame is not None:
                        saved = cv2.imwrite(
                            str(OUTPUT_PATH),
                            last_display_frame,
                        )

                        if saved:
                            print(f"Screenshot saved to:\n{OUTPUT_PATH}")
                        else:
                            print("Could not save the screenshot.")

                elif key == ord("q") or key == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()