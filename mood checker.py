import os
import cv2
from deepface import DeepFace
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read frame from the webcam
    if not ret:
        break

    # Detect emotions using DeepFace
    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']  # Get the dominant emotion

        # Display the detected emotion on the frame
        cv2.putText(frame, f"Mood: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, "Error detecting emotion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Error: {e}")

    # Show the frame with mood detection
    cv2.imshow("Mood Checker", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
