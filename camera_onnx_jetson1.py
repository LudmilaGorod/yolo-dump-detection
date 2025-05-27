import cv2
import os
import numpy as np
from datetime import datetime
import onnxruntime as ort # для запуска модели в формате ONNX

# Путь к модели
onnx_path = r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs\v2_yolo8s_hyp\weights\best.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Папка, куда будем сохранять кадры с обнаруженными объектами
output_dir = r"C:\Users\Ludmila\Desktop\dump\detected_frames_jetson"
os.makedirs(output_dir, exist_ok=True)

# Инициализируем видеопоток с камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

def preprocess(frame):
    """
      Предобработка кадра: изменение размера, нормализация, изменение порядка каналов.
    """
    img = cv2.resize(frame, (640, 640)) # Приводим кадр к нужному размеру
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Конвертируем BGR → RGB
    img = img.astype(np.float32) / 255.0 # Нормализация значений [0,1]
    img = np.transpose(img, (2, 0, 1))  # Изменяет форму массива с (высота, ширина, каналы) → (каналы, высота, ширина)
    img = np.expand_dims(img, axis=0) #Добавляет новую размерность спереди — размерность батча. (Изменяет форму массива с (3, 640, 640) → (1, 3, 640, 640))
    return img

def postprocess(outputs, orig_shape, conf_thres=0.3, iou_thres=0.4):
    """
       Постобработка результатов модели:
       - отбор по порогу уверенности
       - преобразование координат
       - non-maximum suppression (NMS)

    outputs — результат работы модели (массив numpy)
    orig_shape — исходное разрешение кадра, чтобы потом правильно масштабировать координаты
    conf_thres — порог уверенности (если объект предсказан слишком неуверенно, мы его игнорируем)
    iou_thres — порог перекрытия (IoU) для подавления похожих боксов (используется в NMS)
    """
    predictions = outputs[0]  # shape: (1, 5, 8400)
    # Убираем размерность батча
    predictions = np.squeeze(predictions, axis=0)  # shape: (5, 8400)
    # Транспонируем массив, чтобы получить по одной строке на каждый предсказанный объект: одна строка = одна детекция
    predictions = predictions.transpose(1, 0)      # shape: (8400, 5)

    # Отделяем координаты и confidence
    boxes = predictions[:, :4] # cx, cy, w, h
    confidences = predictions[:, 4] # уверенность

    # Фильтруем по порогу уверенности
    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return []

    # Преобразуем координаты из формата YOLO (cx, cy, w, h) в (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # Масштабируем координаты под исходное разрешение кадра
    h, w = orig_shape[:2]
    scale_x, scale_y = w / 640, h / 640
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    # Применяем non-maximum suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )

    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((boxes_xyxy[i], confidences[i], 0))  # всегда класс 0
    return result


class_names = ["dump"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобрабатываем кадр и передаём в модель
    input_tensor = preprocess(frame)
    outputs = session.run([output_name], {input_name: input_tensor})
    detections = postprocess(outputs, frame.shape)

    # Перебираем все обнаруженные прямоугольники на кадре
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]} {score:.2f}"

        # Отображаем прямоугольник и метку
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Сохраняем кадр, если есть детекции
    if len(detections) > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(output_dir, f"detected_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        print(f"Сохранён кадр: {path}")

    # Показываем окно с детекцией в реальном времени
    cv2.imshow("Live Detection", frame)

    # Останавливаем по нажатию 'q' или 'й'
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('й')]:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()