import cv2
import numpy as np
import pyttsx3
import serial

# Serial communication
ser = serial.Serial('COM4', baudrate=9600, timeout=1)

# YOLO file paths
YOLO_CONFIG_PATH = "yolov3.cfg"
YOLO_WEIGHTS_PATH = "yolov3.weights"
COCO_NAMES_PATH = "coco.names"

# Labels to announce
detect_labels = ['person', 'chair', 'car', 'dog', 'bicycle']

# Load class labels
try:
    with open(COCO_NAMES_PATH, "r") as f:
        LABELS = f.read().strip().split("\n")
except FileNotFoundError:
    print("❌ Error: COCO names file not found.")
    exit()

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Object detection function
def detect_objects(frame):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    detected_objects = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            label = LABELS[class_ids[i]]
            if label in detect_labels:
                detected_objects.append(label)
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, list(set(detected_objects))  # Unique objects only



def speak_and_send(objects):
    if not objects:
        return

    msg = "Detected: " + ", ".join(objects)
    print("🗣️", msg)
    engine.say(msg)
    engine.runAndWait()

    for obj in objects:
        if obj == "person":
            ser.write(b'A')
        elif obj == "chair":
            ser.write(b'B')
        elif obj == "car":
            ser.write(b'c')
        elif obj == "dog":
            ser.write(b'D')
        elif obj == "bicycle":
            ser.write(b'E')

# Main loop
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    print("🎥 Real-time detection started. Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detected_frame, detected_objects = detect_objects(frame)

        if frame_count % 30 == 0 and detected_objects:  # Speak every 30 frames (~1 sec)
            speak_and_send(detected_objects)

        cv2.imshow("YOLO Real-Time Detection", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
