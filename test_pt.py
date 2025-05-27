import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загружаем модель
model = YOLO(r"C:\Users\Ludmila\Desktop\custom_train\weights\best.pt")

# Папка с изображениями
input_dir = r"C:\Users\Ludmila\Desktop\dump\v2\test\images"
output_dir = r"C:\Users\Ludmila\Desktop\test_outputs1"

# Создаём папку для сохранения, если её нет
os.makedirs(output_dir, exist_ok=True)

# Получаем список всех файлов в папке
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Обрабатываем все изображения в папке
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, f"annotated_{image_file}")

    print(f"Обрабатываем: {image_path}")

    # Загружаем оригинальное изображение (BGR)
    orig_image = cv2.imread(image_path)

    # Запускаем предсказание
    results = model.predict(source=image_path, imgsz=640, conf=0.5)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # координаты
        classes = result.boxes.cls.cpu().numpy()  # номера классов
        scores = result.boxes.conf.cpu().numpy()  # вероятности

        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255)  # красный (BGR)
            thickness = 2

            # Нарисовать прямоугольник
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, thickness)

            # Составить подпись: имя класса + вероятность
            label = f"{model.names[int(cls)]} {score:.2f}"

            # Параметры текста
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_width, text_height = text_size

            # Координаты для текста ВНУТРИ рамки
            text_x = x1 + 2
            text_y = y1 + text_height + 2  # немного отступить вниз внутри рамки

            # Нарисовать фон под текстом внутри рамки
            cv2.rectangle(orig_image, (x1, y1), (x1 + text_width + 4, y1 + text_height + 8), color, -1)

            # Наложить текст
            cv2.putText(orig_image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

        # Сохраняем результат
        cv2.imwrite(output_path, orig_image)

        # Отображаем
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Detections: {image_file}")
        plt.show(block=False)




print("Обработка завершена.")