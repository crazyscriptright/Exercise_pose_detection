import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# Load the model and label encoder
model = tf.keras.models.load_model('excercise.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Helper function to detect keypoints
def detect_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
        return np.array(keypoints).flatten(), results.pose_landmarks
    return None, None

# Helper function to display keypoints
def draw_pose_landmarks(image, pose_landmarks):
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return annotated_image

# Streamlit UI
st.title("🏋️ Exercise Pose Detection App")
st.write("Upload an image to classify the exercise pose using MediaPipe and a deep learning model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    keypoints, pose_landmarks = detect_keypoints(image)

    if keypoints is not None:
        keypoints_input = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints_input)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_pose_name = label_encoder.inverse_transform([predicted_class_index])[0]

        annotated_image = draw_pose_landmarks(image.copy(), pose_landmarks)

        st.subheader(f"✅ Predicted Pose: `{predicted_pose_name}`")
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.warning("No pose keypoints were detected. Try using a clearer image.")
