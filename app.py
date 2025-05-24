from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
import os
import json
from openai import OpenAI

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("model_new_final.h5")

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the actions/signs the model can recognize
actions = np.array(["Book","Do","Eat","Go","Good","Hello","Home","Hungry","I","Morning","No","Not","Pizza", "Place", "Read","School","Student","Teacher","Thank You", "This", "Tomorrow","Want", "What", "Yes", "Yesterday","You"])

# Initialize OpenAI client for language processing
client = OpenAI(
    base_url="",
    api_key=""
)

# Function to translate raw sign language to proper English
def translate_to_english(raw_signs):
    if not raw_signs:
        return "No signs detected"
    
    # Join the signs into a raw input string
    raw_input = " ".join(raw_signs)
    
    # Create messages for the language model
    messages = [
        {"role": "system", "content": "You are an AI skilled at translating raw sign language input into grammatically correct English sentences. Remember that when a word is repeated twice, it means that the word is in plural form not that it is 2 in quantity."},
        {"role": "user", "content": "Translate the following sign language into proper English sentences."},
        {"role": "assistant", "content": "Raw Input: 'HOME RAIN HEAVY.'\nTranslation: 'It is raining heavily in my home area.'"},
        {"role": "assistant", "content": "Raw Input: 'I TOMORROW EAT FRUIT FRUIT.'\nTranslation: 'Tomorrow I will eat fruits.'"},
        {"role": "assistant", "content": "Raw Input: 'CLASS STUDENTS SIT.'\nTranslation: 'There are students sitting in the class.'"},
        {"role": "assistant", "content": "Raw Input: 'I TONIGHT HOME GO LATE.'\nTranslation: 'I will go home late tonight.'"},
        {"role": "assistant", "content": "Raw Input: 'YOU HUNGRY?'\nTranslation: 'Are you feeling hungry?'"},
        {"role": "user", "content": f"Raw Input: {raw_input}"},
    ]
    
    try:
        # Call the language model
        response = client.chat.completions.create(
            model="",
            messages=messages,
            max_tokens=50  # Increased for longer sentences
        )
        
        # Extract the translation
        translation = response.choices[0].message.content
        
        # If the response includes both "Raw Input" and "Translation", extract just the translation part
        if "Translation:" in translation:
            translation = translation.split("Translation:")[1].strip().strip("'\"")
        
        return translation
    except Exception as e:
        print(f"Error in translation API: {e}")
        # Fallback to simple sentence formation
        return f"{' '.join(raw_signs)}"

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = pose[:69]
    return np.concatenate([pose,lh,rh])

def normalize_keypoints(keypoints, center_keypoint, reference_distance):
    keypoints = keypoints.reshape(-1, 3)
    relative_keypoints = keypoints - center_keypoint
    relative_keypoints = relative_keypoints / reference_distance
    return relative_keypoints.flatten()

def preprocess_hand_keypoints(hand_keypoints):
    if np.any(hand_keypoints):
        wrist_keypoint = hand_keypoints[0:3]
        relative_hand_keypoints = (hand_keypoints.reshape(-1, 3) - wrist_keypoint)
    else:
        relative_hand_keypoints = np.zeros(21 * 3)
    return relative_hand_keypoints.flatten()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    content_type = video_file.content_type
    temp_path = 'temp_video.webm' if 'webm' in content_type else 'temp_video.mp4'
    print(f"Content type: {content_type}, using temp path: {temp_path}")
    video_file.save(temp_path)
    
    # Get pause information if provided
    pause_info = []
    if 'pause_info' in request.form:
        try:
            pause_info = json.loads(request.form['pause_info'])
            print(f"Received pause_info: {pause_info}")
        except Exception as e:
            print(f"Error parsing pause_info: {e}")
    
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return jsonify({'error': 'Could not open video file'}), 500
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    sequences = []
    results_list = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        window = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_per_second = 15  # Target 15 frames per second
        frame_interval = int(fps / frames_per_second) if fps > frames_per_second else 1
        print(f"Video FPS: {fps}, processing every {frame_interval} frame(s)")
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every nth frame to achieve 15 fps
            if frame_index % frame_interval == 0:
                # Process frame only if not in pause
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                
                # Store prediction for each frame
                if len(keypoints) > 0:
                    # Normalize keypoints
                    center_keypoint = keypoints[0:3]
                    left_shoulder = keypoints[11*3:11*3+3]
                    right_shoulder = keypoints[12*3:12*3+3]
                    reference_distance = np.linalg.norm(left_shoulder - right_shoulder)
                    if not reference_distance:
                        reference_distance = 1
                        
                    normalized_pose = normalize_keypoints(keypoints[:69], center_keypoint, reference_distance)
                    left_hand = keypoints[69:69 + 21*3]
                    right_hand = keypoints[69 + 21*3:]
                    
                    relative_left_hand = preprocess_hand_keypoints(left_hand)
                    relative_right_hand = preprocess_hand_keypoints(right_hand)
                    
                    full_frame = np.concatenate([normalized_pose, relative_left_hand, relative_right_hand])
                    window.append(full_frame)
                    frame_count += 1
                    
                    # Process every 30 frames
                    if len(window) == 30:
                        print(f"Processing window of 30 frames")
                        window_array = np.array(window)
                        smoothed_window = uniform_filter1d(window_array, size=3, axis=0)
                        prediction = model.predict(np.expand_dims(smoothed_window, axis=0))
                        predicted_action = actions[np.argmax(prediction)]
                        confidence = float(np.max(prediction))
                        
                        print(f"Predicted: {predicted_action}, confidence: {confidence}")
                        
                        # Only add predictions with confidence above threshold
                        if confidence > 0.5:  # Lowered from 0.7 to 0.5
                            results_list.append({
                                'sign': predicted_action,
                                'confidence': confidence
                            })
                        # Clear window for next batch
                        window = []
            
            frame_index += 1
    
    cap.release()
    os.remove(temp_path)
    
    print(f"Processed {frame_count} frames, got {len(results_list)} raw predictions")
    
    # Filter out duplicate consecutive signs
    filtered_results = []
    prev_sign = None
    for result in results_list:
        if result['sign'] != prev_sign:
            filtered_results.append(result)
            prev_sign = result['sign']
    
    # Extract just the sign names for translation
    sign_names = [result['sign'] for result in filtered_results]
    
    # Create a basic sentence by joining the signs
    basic_sentence = " ".join(sign_names)
    
    # Use the language model to translate to proper English
    translated_sentence = translate_to_english(sign_names)
    
    print(f"Results list: {filtered_results}")
    print(f"Basic sentence: {basic_sentence}")
    print(f"Translated sentence: {translated_sentence}")
    
    # Return filtered predictions and both sentences
    return jsonify({
        'predictions': filtered_results,
        'sentence': translated_sentence,
        'raw_sentence': basic_sentence
    })

if __name__ == '__main__':
    # For development only - not secure for production
    app.run(host='127.0.0.1', port=5000, debug=True)
