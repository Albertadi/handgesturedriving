import os
import glob
import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create the hands processing object
hands = mp_hands.Hands(
    static_image_mode=True,       # True = treat each image as a static image (no tracking)
    max_num_hands=1,             # Adjust if you expect multiple hands per image
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks_from_image(image_path):
    """
    Given the path to an image containing a single hand gesture,
    returns a list of 42 floating-point coordinates for the 21 landmarks,
    or None if no hand is detected.
    """
    # Read image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return None
    
    # Convert BGR to RGB for Mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with Mediapipe
    result = hands.process(image_rgb)
    
    if not result.multi_hand_landmarks:
        # No hand detected
        return None
    
    # We only asked for max_num_hands=1, so just use the first hand
    hand_landmarks = result.multi_hand_landmarks[0]
    
    # Extract the 21 landmark coordinates
    # Mediapipe gives normalized coordinates [0,1]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        landmark_list.append(lm.x)
        landmark_list.append(lm.y)
        # If you want z as well, you can add lm.z. Typically for 2D classification, x & y suffice.
    
    return landmark_list  # length = 42

def build_landmark_dataset(base_dir):
    """
    Scans through subfolders of `base_dir`.
    Each subfolder name is treated as a label.
    Collects 21 x,y landmarks for each image and returns a Pandas DataFrame.
    """
    data_rows = []
    
    # Each subfolder in base_dir is a gesture label
    gesture_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for gesture_label in gesture_dirs:
        gesture_path = os.path.join(base_dir, gesture_label)
        # Get all images in this gesture's folder
        image_paths = glob.glob(os.path.join(gesture_path, "*.jpg")) + \
                      glob.glob(os.path.join(gesture_path, "*.png")) + \
                      glob.glob(os.path.join(gesture_path, "*.jpeg"))
        
        for img_path in image_paths:
            landmarks = extract_landmarks_from_image(img_path)
            if landmarks is not None:
                # Create a row: [x1, y1, x2, y2, ..., x21, y21, label]
                row = landmarks + [gesture_label]
                data_rows.append(row)
            else:
                print(f"No hand detected in {img_path}")
    
    # Create a DataFrame
    # We'll name columns: x1, y1, x2, y2, ..., x21, y21, label
    column_names = []
    for i in range(1, 22):
        column_names.append(f"x{i}")
        column_names.append(f"y{i}")
    column_names.append("label")
    
    df = pd.DataFrame(data_rows, columns=column_names)
    
    return df

def main():
    base_dir = "dataset"  # your dataset folder
    df = build_landmark_dataset(base_dir)
    
    print("Extracted landmark dataset:")
    print(df.head())
    
    # Save to CSV
    df.to_csv("hand_landmarks_dataset.csv", index=False)
    print("Saved dataset to hand_landmarks_dataset.csv")

if __name__ == "__main__":
    main()