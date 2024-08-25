import os
import sys
import json
import cv2
from pose import process_image, interpolate_landmarks, plot_and_save_landmarks, plot_all_landmarks
import subprocess

def pose_estimation(image_path1, image_path2):
    if not (os.path.exists(image_path1) and os.path.exists(image_path2)):
        print("invalid path")
        return

    landmarks1 = process_image(image_path1)
    landmarks2 = process_image(image_path2)

    interpolated_landmarks = interpolate_landmarks(landmarks1, landmarks2, alpha=0.5)
    
    output_interpolated_json_path = 'output/json/interpolated_pose_landmarks.json'
    os.makedirs(os.path.dirname(output_interpolated_json_path), exist_ok=True)
    with open(output_interpolated_json_path, 'w') as f:
        json.dump(interpolated_landmarks, f, indent=4)

    print(f"補間されたJSONファイルが保存されました。\nJSON: {output_interpolated_json_path}")

    image1 = cv2.imread(image_path1)
    height, width, _ = image1.shape

    plot_and_save_landmarks(landmarks1, 'output/vis/landmarks1.png', width, height)
    plot_and_save_landmarks(landmarks2, 'output/vis/landmarks2.png', width, height)
    plot_and_save_landmarks(interpolated_landmarks, 'output/vis/interpolated_landmarks.png', width, height)
    
    return landmarks1, landmarks2, interpolated_landmarks

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <image_path1> <image_path2>")
        sys.exit(1)
    
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    landmarks1, landmarks2, interpolated_landmarks = pose_estimation(image_path1, image_path2)