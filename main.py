import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ultralytics import YOLO


def check_system_info():
    """Проверка информации о системе и вывод ее."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    torch.cuda.empty_cache()


def plot_sample_images(image_dir, label_dir, num_images=16):
    """Вывод случайных изображений из датасета с аннотациями."""
    image_files = os.listdir(image_dir)
    random_images = random.sample(image_files, num_images)

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i, image_file in enumerate(random_images):
        row, col = divmod(i, 4)
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = f.read().strip().split("\n")
            for label in labels:
                if len(label.split()) != 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, label.split())
                x_min = int((x_center - width / 2) * image.shape[1])
                y_min = int((y_center - height / 2) * image.shape[0])
                x_max = int((x_center + width / 2) * image.shape[1])
                y_max = int((y_center + height / 2) * image.shape[0])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row, col].axis('off')
    plt.show()


def train_model(image_size, project_dir):
    """Тренировка модели YOLO."""
    model = YOLO('yolov8s.pt')
    model.train(
        data=r"C:\Users\Ludmila\Desktop\dump\v2\data.yaml",
        epochs= 150,
        imgsz=image_size,
        seed=42,
        batch=8,
        lr0=0.01,
        project=project_dir,
        name="v2_yolo8s_150"
    )


def plot_training_metrics(csv_path):
    """Построение графиков метрик из тренировочного процесса."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
    sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0, 0])
    sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0, 1])
    sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1, 0])
    sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1, 1])
    sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2, 0])
    sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2, 1])
    sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3, 0])
    sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3, 1])
    sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4, 0])
    sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4, 1])

    for ax in axs.flat:
        ax.set_title(ax.get_ylabel().replace("/", " ").title())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    check_system_info()

    # Задаем пути
    train_images = r"C:\Users\Ludmila\Desktop\dump\v2\train\images"
    train_labels = r"C:\Users\Ludmila\Desktop\dump\v2\train\labels"
    project_dir = r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs"
    bect_hyp_path = r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs\v2_tune_yolo8x\best_hyperparameters.yaml"

    # Демонстрация изображений
    plot_sample_images(train_images, train_labels)

    # Тренировка модели
    height = 640
    train_model(height, project_dir)

    # Построение графиков метрик
    results_csv = os.path.join(project_dir, "v2_yolo8s_150", "results.csv")
    plot_training_metrics(results_csv)