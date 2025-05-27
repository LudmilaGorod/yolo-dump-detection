import os
import torch
from ultralytics import YOLO


def tune_hyperparameters(image_size, project_dir):
    """Подбор гиперпараметров для YOLOv8."""
    # Загружаем предобученную модель YOLOv8s
    model = YOLO('yolov8s.pt')

    print("Начинается подбор гиперпараметров...")
    # Запуск подбора гиперпараметров
    model.tune(
        data=r"C:\Users\Ludmila\Desktop\dump\v2\data.yaml",
        epochs=60,
        imgsz=image_size,
        seed= 42,
        iterations=30,
        optimizer='SGD',
        val=True,
        plots=True,
        project=project_dir,
        name="v2_hyp3_yolo8s"
    )

    # Путь к лучшему hyp.yaml
    best_hyp_path = os.path.join(project_dir, "v2_hyp3_yolo8s", "best_hyperparameters.yaml")
    print("\n✅ Подбор завершён. Лучшие гиперпараметры:")
    if os.path.exists(best_hyp_path):
        with open(best_hyp_path, "r") as f:
            print(f.read())
    else:
        print("Файл hyp.yaml не найден")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # Папка, в которую сохраняются результаты подбора
    project_dir = r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs_2"
    image_size = 640

    # Запуск подбора гиперпараметров
    tune_hyperparameters(image_size, project_dir)
