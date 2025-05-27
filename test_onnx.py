import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загружаем модель
model = YOLO(r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs\v2_yolo8s_hyp\weights\best.onnx")

# Папка с изображениями
input_dir = r"C:\Users\Ludmila\Desktop\Все_датасеты\ТБО\11.03.2025 Мусор возле ККУЦ\фото с дрона"
output_dir = r"C:\Users\Ludmila\Desktop\Все_датасеты\ТБО\11.03.2025 Мусор возле ККУЦ\annotated"

# Создаём папку для сохранения, если её нет
os.makedirs(output_dir, exist_ok=True)

# Получаем список всех файлов в папке
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Обрабатываем все изображения в папке
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, f"annotated_{image_file}")

    print(f"Обрабатываем: {image_path}")

    # Запускаем предсказание
    results = model.predict(source=image_path, imgsz=640)

    for result in results:
        # Получаем размеченное изображение
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Сохраняем изображение
        cv2.imwrite(output_path, annotated_image)
        print(f"Сохранено: {output_path}")

        # Отображаем изображение через Matplotlib (не закрывая предыдущие)
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Detections: {image_file}")
        plt.show(block=False)  # Не блокирует выполнение программы

print("Обработка завершена.")
