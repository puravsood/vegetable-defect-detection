# This code corresponds to the mobilenetv2_basic model in saved_models. It classified in real time(no lags) however it was not very accurate in its classification which is why the model was then finetuned. The training file for this corresponding model is not available unfortunately. If you wish to train it again you can remove the finetuning from finetuned_model_training.py in src and run the code. 
import cv2
import tensorflow as tf
import numpy as np

# Load the full TF model (SavedModel format)
model_path = "/path/to/your/model/here"
model = tf.keras.models.load_model(model_path)

# Get input shape from the model
input_shape = model.input_shape
height, width = input_shape[1], input_shape[2]

# Label mapping
class_names = [
    'fresh_apple', 'fresh_banana', 'fresh_potato',
    'rotten_apple','rotten_banana', 'rotten_potato'
]

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess frame
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)

    # Run inference
    predictions = model.predict(input_data)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    label = class_names[pred_idx]

    # Display result
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Vegetable Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
