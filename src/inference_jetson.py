import cv2
import tensorflow as tf
import numpy as np

# Load the trained model (SavedModel format)
model_path = "/path/to/your/saved/model/here"
model = tf.keras.models.load_model(model_path)

# Get input shape
input_shape = model.input_shape
height, width = input_shape[1], input_shape[2]

# Class labels (only apples and bananas)
class_names = [
    'fresh_apple', 'rotten_apple',
    'fresh_banana', 'rotten_banana'
]

# Open USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess image
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)

    # Inference
    predictions = model.predict(input_data)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    label = class_names[pred_idx]

    # Check for rotten detection
    if 'rotten' in label:
        print(f"⚠️ Rotten fruit detected: {label} (Confidence: {confidence:.2f})")

    # Display label on camera window
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if 'rotten' in label else (0, 255, 0), 2)

    cv2.imshow("Fruit Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
