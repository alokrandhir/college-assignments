
import cv2
from ultralytics import YOLO

# Load the YOLO object detection model
model = YOLO("yolo11n.pt")

# Open CCTV video
cap = cv2.VideoCapture("cctv.mp4")

if not cap.isOpened():
    print("Error: Cannot open cctv.mp4")
    raise SystemExit

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

# Save the detected video
out = cv2.VideoWriter(
    "detected_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("Object detection started...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects in each video frame
    results = model(frame, verbose=False)

    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

    # Save the processed frame
    out.write(annotated_frame)

    # Show the processed frame
    cv2.imshow("CCTV Object Detection", annotated_frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed!")
print("Saved video as detected_output.mp4")
