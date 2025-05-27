import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ultralytics import YOLO
import yaml

def train_model(image_size, project_dir):
    """Функция для обучения модели YOLO с заданными гиперпараметрами."""

    # Загружаем словарь с гиперпараметрами из yaml-файла
    with open(r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs_2\v2_hyp3_yolo8s2\best_hyperparameters.yaml", "r") as f:
        hyp_dict = yaml.safe_load(f)

    # Загружаем предобученную модель YOLOv8s
    model = YOLO('yolov8s.pt')
    model.train(
        data=r"C:\Users\Ludmila\Desktop\dump\v2\data.yaml",
        epochs= 150,
        imgsz=image_size,
        seed=42,
        project=project_dir,
        name="v2_yolo8s_hyp3_150",
        **hyp_dict
    )


def plot_training_metrics(csv_path):
    """Построение графиков метрик из тренировочного процесса."""

    # Загружаем CSV-файл с результатами обучения
    df = pd.read_csv(csv_path)

    # Убираем лишние пробелы из названий колонок, если они есть
    df.columns = df.columns.str.strip()

    # Создаем 5 строк по 2 графика (итого 10 графиков)
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

    # Строим графики для каждой метрики
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

    # Добавляем заголовки на графики на основе названия метрик
    for ax in axs.flat:
        ax.set_title(ax.get_ylabel().replace("/", " ").title())

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Для корректной работы multiprocessing на Windows
    torch.multiprocessing.freeze_support()

    # Задаем директорию, куда сохраняются результаты обучения
    project_dir = r"C:\Users\Ludmila\Desktop\dump\YOLO_outputs"

    # Тренировка модели
    height = 640
    train_model(height, project_dir)

    # Путь к файлу с результатами обучения
    results_csv = os.path.join(project_dir, "v2_yolo8s_hyp3_150", "results.csv")
    # Строим графики по метрикам обучения
    plot_training_metrics(results_csv)