python
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# -------------------------------------------------
# File paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_IMAGE = PROJECT_ROOT / "webapp_screenshot.png"
OUTPUT_IMAGE = PROJECT_ROOT / "webapp_screenshot_with_landmarks.png"
TASK_PATH = PROJECT_ROOT / "hand_landmarker.task"


# -------------------------------------------------
# Drawing function
# -------------------------------------------------

def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()

    if detection_result is None:
        return annotated_image

    if not detection_result.hand_landmarks:
        print("No hands were detected.")
        return annotated_image

    landmark_style = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 255, 255),   # Yellow landmarks
        thickness=2,
        circle_radius=4
    )

    connection_style = mp.solutions.drawing_utils.DrawingSpec(
        color=(255, 120, 0),   # Blue/cyan connections
        thickness=3
    )

    for hand_landmarks in detection_result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            )
            for landmark in hand_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            landmark_style,
            connection_style
        )

    return annotated_image


# -------------------------------------------------
# Main program
# -------------------------------------------------

def main():
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(
            f"Screenshot not found:\n{INPUT_IMAGE}\n\n"
            "Rename your screenshot to webapp_screenshot.png "
            "and place it beside this script."
        )

    if not TASK_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe task file not found:\n{TASK_PATH}"
        )

    screenshot = cv2.imread(str(INPUT_IMAGE))

    if screenshot is None:
        raise RuntimeError("The screenshot could not be opened.")

    print("Select the camera area.")
    print("Drag a rectangle around the webcam preview.")
    print("Press ENTER or SPACE when finished.")

    x, y, width, height = cv2.selectROI(
        "Select camera area",
        screenshot,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    if width == 0 or height == 0:
        raise RuntimeError("No camera area was selected.")

    camera_crop = screenshot[
        y:y + height,
        x:x + width
    ].copy()

    rgb_crop = cv2.cvtColor(
        camera_crop,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_crop
    )

    base_options = python.BaseOptions(
        model_asset_path=str(TASK_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    annotated_crop = draw_landmarks_on_image(
        camera_crop,
        result
    )

    screenshot[
        y:y + height,
        x:x + width
    ] = annotated_crop

    success = cv2.imwrite(
        str(OUTPUT_IMAGE),
        screenshot
    )

    if not success:
        raise RuntimeError("The output image could not be saved.")

    print()
    print("Finished.")
    print("Detected hands:", len(result.hand_landmarks))
    print("Saved image:")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()

