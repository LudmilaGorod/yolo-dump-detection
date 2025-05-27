import random
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import cv2
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from loguru import logger

# Импорт функции, преобразующей координаты bbox из относительных в абсолютные, импорт функций загрузки и сохранения аннотаций YOLO
from helpers import convert_bbox_to_absolute, load_yolo_annotations, save_yolo_annotations

def create_augmenter() -> iaa.Augmenter:
    """
    Создаёт последовательность аугментаций с использованием imgaug.
    Трансформации применяются в случайном порядке.
    """
    return iaa.Sequential([
        iaa.Affine(scale=(0.9, 1.1)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rotate((-15, 15)),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Multiply((0.8, 1.2)),
    ], random_order=True)


def apply_augmentation(image: np.ndarray, boxes: List[List[float]], augmenter: iaa.Augmenter) -> Tuple[
    np.ndarray, List[List[float]]]:
    """
    Применяет аугментации к изображению и соответствующим bounding box'ам.
    Возвращает изменённое изображение и обновлённые аннотации.
    """
    if not boxes:
        return image, []

    absolute_boxes = [convert_bbox_to_absolute(box, image.shape) for box in boxes]
    # Создаём объект bbox для imgaug
    bbs = BoundingBoxesOnImage(absolute_boxes, shape=image.shape)
    # Применяем аугментации
    augmented_image, augmented_bbs = augmenter(image=image, bounding_boxes=bbs)
    # Удаляем или обрезаем боксы, вышедшие за пределы изображения
    valid_bbs = augmented_bbs.remove_out_of_image().clip_out_of_image().bounding_boxes
    filtered_boxes = []
    for bb in valid_bbs:
        # Конвертируем координаты обратно в YOLO-формат
        x_center = ((bb.x1 + bb.x2) / 2) / augmented_image.shape[1]
        y_center = ((bb.y1 + bb.y2) / 2) / augmented_image.shape[0]
        w = (bb.x2 - bb.x1) / augmented_image.shape[1]
        h = (bb.y2 - bb.y1) / augmented_image.shape[0]
        class_id = bb.label
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
            filtered_boxes.append([x_center, y_center, w, h, class_id])

    return augmented_image, filtered_boxes

def process_single_augmentation(task: Tuple[Path, Path, Path, Path, iaa.Augmenter, int]):
    """
    Читает изображение, применяет аугментации,
    визуализирует результат, сохраняет изображение и метки.

    """
    image_path, label_path, output_images_dir, output_labels_dir, augmenter, img_index = task
    image = cv2.imread(str(image_path))

    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        return

    boxes = load_yolo_annotations(label_path)
    if not boxes:
        logger.warning(f"No bounding boxes found in {label_path}")
        return

    # Аугментация изображения и обновление координат боксов
    augmented_image, valid_boxes_with_labels = apply_augmentation(image, boxes, augmenter)

    if not valid_boxes_with_labels:
        logger.warning(f"All bounding boxes were filtered out for {image_path}. Skipping save.")
        return

    # Визуализация изображения
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Augmented Image")
    plt.show()

    # Сохраняем аугментированное изображение
    augmented_image_file = output_images_dir / f"aug_{img_index}.png"

    if augmented_image is None or augmented_image.size == 0:
        logger.error(f"Augmented image is empty! Cannot save: {augmented_image_file}")
    else:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(augmented_image_file), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved augmented image: {augmented_image_file}")

    # Сохраняем обновлённые аннотации
    augmented_label_file = output_labels_dir / f"aug_{img_index}.txt"
    save_yolo_annotations(augmented_label_file, valid_boxes_with_labels)
    logger.info(f"Saved augmented label: {augmented_label_file}")


def run_augmentation(dataset_dir: Path, num_samples: int) -> Path:
    """
    Основная функция для запуска аугментации. Обрабатывает случайные изображения
    из тренировочного набора и сохраняет результаты.
    """

    from tqdm import tqdm # отображение прогресса
    from tempfile import TemporaryDirectory # создание временной папки
    import shutil # копирование файлов и директорий

    # Поиск директорий с изображениями и метками
    train_images_dir = dataset_dir / "images"
    train_labels_dir = dataset_dir / "labels"
    if not train_images_dir.exists() or not train_labels_dir.exists():
        train_images_dir = dataset_dir / "train" / "images"
        train_labels_dir = dataset_dir / "train" / "labels"

    if not train_images_dir.exists() or not train_labels_dir.exists():
        logger.warning(f"train/images or train/labels not found in {dataset_dir}. Skipping segmentation.")
        return dataset_dir

    # Создание временной копии датасета
    temp_dir_obj = TemporaryDirectory()
    temp_dir = Path(temp_dir_obj.name)
    for filename in os.listdir(dataset_dir):
        source = os.path.join(dataset_dir, filename)
        destination = os.path.join(temp_dir, filename)

        if os.path.isdir(source):
            shutil.copytree(source, destination)

    # Используем только temp_dir, ищем все файлы изображений и меток, соединяем их в пары: (изображение, метка).
    image_files = sorted(list(temp_dir.rglob("*.[jJ][pP][gG]")) + list(temp_dir.rglob("*.[pP][nN][gG]")))
    label_files = sorted(list(temp_dir.rglob("*.[tT][xX][tT]")))

    train_files = list(zip(image_files, label_files))

    total_images = len(train_files)
    if total_images == 0:
        logger.warning("Нет изображений для аугментации.")
        return dataset_dir

    tasks = []
    for i in range(num_samples):
        idx = np.random.randint(0, total_images)
        image_path, label_path = train_files[idx]
        seed = random.randint(0, 100000)
        augmenter = create_augmenter()
        augmenter.reseed(seed)
        tasks.append((image_path, label_path, train_images_dir, train_labels_dir, augmenter, i))

    # Запуск аугментации
    for task in tqdm(tasks, total=num_samples, desc="Augmentation"):
        process_single_augmentation(task)

    logger.info(f"Augmentation completed. Results are in {dataset_dir}")
    return dataset_dir

if __name__ == '__main__':
    dataset_path = Path(r"C:\Users\Ludmila\Desktop\dump\v1_1\train")  # Путь к датасету

    output_dir = r"C:\Users\Ludmila\Desktop\dump\v1_1\train\images"
    # Проверка доступности папки с изображениями на запись
    if not os.path.exists(output_dir):
        print(f"Ошибка: Папка {output_dir} не существует!")
    elif not os.access(output_dir, os.W_OK):
        print(f"Ошибка: Нет прав на запись в {output_dir}!")
    else:
        print("Папка доступна для записи.")

    num_samples = 10 # Количество аугментированных изображений

    run_augmentation(dataset_path, num_samples)
