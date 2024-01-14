import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from PIL import Image
import requests
import concurrent.futures
import time

# Load the model
new_model = tf.keras.models.load_model('iot_model.h5')

# Capture video from the default camera (0)
cap = cv2.VideoCapture(0)

def send_put_request(predicted_class, confidence):
    api_url = "https://iot-71ac0f336b84.herokuapp.com/api/field/update"  # Replace with your actual API endpoint

    # Set isCheck parameter based on the condition
    is_check = predicted_class in ["Mix", "Bug"]

    # Example data for the PUT request
    data = {"id": 6, "name": "Tarla - 01", "check": is_check}

    response = requests.put(api_url, json=data)

    if response.status_code == 200:
        print("PUT request successful")
    else:
        print(f"Failed to send PUT request. Status code: {response.status_code}")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (896, 896))
    cv2.imshow('Camera Feed', frame)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = new_model.predict(img_array)
    classes = ["Honey", "Mix", "Bug", "Healthy"]
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence}")

    # Conditionally send the PUT request
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(send_put_request, predicted_class, confidence)

    # Introduce a 4-second delay
    time.sleep(4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()