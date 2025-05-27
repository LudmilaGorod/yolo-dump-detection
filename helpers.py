from typing import List, Tuple
from imgaug.augmentables.bbs import BoundingBox
from pathlib import Path

def convert_bbox_to_absolute(box: List[float], img_shape: Tuple[int, int]) -> BoundingBox:
    """
     Преобразует ограничивающую рамку (bbox) из YOLO-формата в абсолютные координаты.
    Вход:
        box: список [x_center, y_center, width, height, class_id] — координаты в формате YOLO
        img_shape: (высота, ширина) изображения
    Возвращает:
        BoundingBox из библиотеки imgaug — с абсолютными координатами и меткой класса.
    """
    x_center, y_center, w, h, class_id = box
    width, height = img_shape[1], img_shape[0]
    x_min = (x_center - w / 2) * width
    y_min = (y_center - h / 2) * height
    x_max = (x_center + w / 2) * width
    y_max = (y_center + h / 2) * height
    return BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id)

def load_yolo_annotations(label_path: Path) -> List[List[float]]:
    """
    Загружает аннотации из YOLO-файла .txt.
    Фильтрует некорректные строки (нечисловые значения или не 5 элементов).
    Возвращает список корректных bbox-ов в формате [x_center, y_center, w, h, class_id].
    """
    boxes = []
    with label_path.open("r") as f:
        for line in f:
            try:
                data = list(map(float, line.strip().split()))
            except ValueError:
                continue

            if len(data) != 5:
                continue

            class_id, x_center, y_center, w, h = data
            # Check that bounding box values are within expected range [0, 1].
            if all(0 <= val <= 1 for val in [x_center, y_center, w, h]):
                boxes.append([x_center, y_center, w, h, int(class_id)])
            else:
                pass
    return boxes


def save_yolo_annotations(output_label_path: Path, boxes_with_labels: List[List[float]]):
    """
     Сохраняет список bbox-ов в текстовый файл в YOLO-формате.
    Вход:
        output_label_path — путь к .txt-файлу, куда сохранить аннотации.
        boxes_with_labels — список bbox'ов в формате [x_center, y_center, width, height, class_id]
    """
    with open(output_label_path, "w") as f:
        for box in boxes_with_labels:
            x_center, y_center, width, height, class_id = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
