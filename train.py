from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    device="cpu",
    name="miltech_yolov8n",
    project="MILITARY_DETECTION_YOLO"
)

