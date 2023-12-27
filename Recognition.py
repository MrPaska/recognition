import cv2
import tensorflow as tf
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

tflite_model = "best-fp16_2.tflite"
label_map_file = "labelmap.txt"

with open(label_map_file, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
module = Interpreter(model_path=tflite_model)
module.allocate_tensors()

# Getting input and output tensors
input_details = module.get_input_details()
output_details = module.get_output_details()

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Preprocessing the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
    frame_type = frame_resized.astype(np.float32)
    input_data = np.expand_dims(frame_type, axis=0)

    # Transposing dimensions
    # input_data = np.transpose(input_data, (0, 3, 1, 2))

    print(f"Frame.shape = {frame.shape}")
    print(f"Input_data.shape = {input_data.shape}")

    # Invoking model
    module.set_tensor(input_details[0]["index"], input_data)
    module.invoke()

    # Getting output
    boxes = module.get_tensor(output_details[0]["index"])[0]  # Bounding box coordinates
    class_indices = module.get_tensor(output_details[1]["index"])[0].astype(int)

    # Map class indices to class names
    classes = [labels[i] for i in class_indices]

    print(f"Output_pred_box: {boxes}")
    print(f"Output_dtl: {output_details}")

cap.release()
cv2.destroyAllWindows()
