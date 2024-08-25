import cv2
import mediapipe as mp
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def process_image(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        draw_and_save_image(image, results.pose_landmarks, image_path)
        landmark_data = extract_landmarks(results.pose_landmarks, image.shape[1], image.shape[0])
        save_landmarks_as_json(landmark_data, image_path)
        print(f"画像とJSONファイルが保存されました。\n画像: {get_output_image_path(image_path)}\nJSON: {get_output_json_path(image_path)}")
    else:
        landmark_data = []
    
    pose.close()
    return landmark_data

def draw_and_save_image(image, pose_landmarks, image_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(image, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    output_image_path = get_output_image_path(image_path)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)

def extract_landmarks(pose_landmarks, width, height):
    return [{
        'x': landmark.x * width,
        'y': landmark.y * height,
        'z': landmark.z * width,  # Assuming z uses the same scale as x
        'visibility': landmark.visibility
    } for landmark in pose_landmarks.landmark]

def save_landmarks_as_json(landmark_data, image_path):
    output_json_path = get_output_json_path(image_path)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(landmark_data, f, indent=4)

def get_output_image_path(image_path):
    return f'output/vis/{os.path.basename(image_path)}'

def get_output_json_path(image_path):
    return f'output/json/{os.path.splitext(os.path.basename(image_path))[0]}_pose_landmarks.json'

def interpolate_landmarks(landmarks1, landmarks2, alpha=0.5):
    if len(landmarks1) != len(landmarks2):
        raise ValueError("The number of landmarks in both inputs must be the same.")
    
    interpolated_landmarks = []
    for l1, l2 in zip(landmarks1, landmarks2):
        interpolated_landmark = {
            'x': l1['x'] * (1 - alpha) + l2['x'] * alpha,
            'y': l1['y'] * (1 - alpha) + l2['y'] * alpha,
            'z': l1['z'] * (1 - alpha) + l2['z'] * alpha,
            'visibility': l1['visibility'] * (1 - alpha) + l2['visibility'] * alpha
        }
        interpolated_landmarks.append(interpolated_landmark)
    
    return interpolated_landmarks

def plot_landmarks(landmarks, ax, title, width, height, label):
    print("length of keypoints :", len(landmarks))
    x = [lm['x'] for lm in landmarks]
    y = [lm['y'] for lm in landmarks]
    ax.scatter(x, y, label=label)
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        ax.text(x_i, y_i, str(i), fontsize=12, ha='right')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.set_title(title)

def plot_and_save_landmarks(landmarks, filename, width, height):
    fig, ax = plt.subplots()
    plot_landmarks(landmarks, ax, 'Pose Landmarks', width, height, 'Landmarks')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def plot_all_landmarks(landmarks1, landmarks2, interpolated_landmarks, filename, width, height):
    fig, ax = plt.subplots()
    plot_landmarks(landmarks1, ax, 'Pose Landmarks', width, height, 'Landmarks 1')
    plot_landmarks(landmarks2, ax, 'Pose Landmarks', width, height, 'Landmarks 2')
    plot_landmarks(interpolated_landmarks, ax, 'Pose Landmarks', width, height, 'Interpolated Landmarks')
    ax.legend()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    image_path1 = 'test001.png'
    image_path2 = 'test002.png'
    
    if not (os.path.exists(image_path1) and os.path.exists(image_path2)):
        print("invalid path")
        exit(1)

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

    plot_all_landmarks(landmarks1, landmarks2, interpolated_landmarks, 'output/vis/all_landmarks.png', width, height)

    print("3つのJSONを1つの画像として保存しました。\n画像: output/vis/all_landmarks.png")
