import cv2
import tensorflow as tf
import numpy as np

tflite_model_file = "best-fp16_2.tflite"
with open(tflite_model_file, "rb") as fid:
    tflite_model = fid.read()

# Load TFLite model
module = tf.lite.Interpreter(model_content=tflite_model)
module.allocate_tensors()

# Getting input and output tensors
input_dtl = module.get_input_details()
output_dtl = module.get_output_details()

classes = ["momentinis", "vidutinis", "zona"]

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, img = cap.read()
    if success:
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Camera failed!")
    # Preprocessing the frame
    input_data = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
    #input_data = input_data.astype(np.float32)  # Convert to float32
    input_data = input_data.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # Transposing dimensions
    #input_data = np.transpose(input_data, (0, 3, 1, 2))

    print(img.shape)
    print(input_data.shape)
    print(module.get_input_details())

    # Invoking model
    module.set_tensor(input_dtl[0]["index"], input_data)
    module.invoke()

    # Getting output tensor
    module.set_tensor(input_dtl[0]['index'], input_data)
    module.invoke()
    pred_boxes = module.get_tensor(output_dtl[0]['index'])
    #pred_scores = module.get_tensor(output_dtl[1]['index'])
    # pred_classes = module.get_tensor(output_dtl[2]['index'])
    # num_boxes = int(module.get_tensor(output_dtl[3]['index'])[0])
    print(f"Output_pred_box: {pred_boxes}")
    print(f"Output_dtl: {output_dtl}")
cap.release()
cv2.destroyAllWindows()

