#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
import numpy as np

# Load the AI model
model = tf.keras.models.load_model("/home/weed_bot/src/your_package_name/weed_model.h5")
bridge = CvBridge()

def preprocess_image(cv_image):
    """
    Preprocess the OpenCV image to match the input shape and normalization
    requirements of the AI model.
    """
    # Resize to model's input size (assuming 224x224 as an example)
    input_size = (150, 150)  # Update according to your model's requirements
    resized_image = cv2.resize(cv_image, input_size)

    # Normalize the image (assuming the model requires normalization)
    normalized_image = resized_image / 255.0  # Scale pixel values to [0, 1]

    # Add batch dimension (1, height, width, channels)
    input_tensor = np.expand_dims(normalized_image, axis=0)

    return input_tensor

def image_callback(msg):
    try:
        # Convert the ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Preprocess the image for the model
        input_tensor = preprocess_image(cv_image)

        # Run inference
        predictions = model.predict(input_tensor)
        predicted_class = np.argmax(predictions)
        print(predictions, predicted_class)
        
        # Process the predictions (example: display them or overlay on the image)
        if predicted_class==0:
            print(f"Prediction: Crop")
        else:
            print("Prediction: Weed")

        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

def main():
    rospy.init_node('camera_ai_inference', anonymous=True)

    # Subscribe to the /camera/image_raw topic
    rospy.Subscriber("/camera/image_raw", Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
