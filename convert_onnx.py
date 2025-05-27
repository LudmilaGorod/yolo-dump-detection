from ultralytics import YOLO

model = YOLO("C:/Users/Ludmila/Desktop/dump/YOLO_outputs/v2_yolo8s_hyp/weights/best.pt")
model.export(format="onnx", opset=12, dynamic=True, simplify=True)
