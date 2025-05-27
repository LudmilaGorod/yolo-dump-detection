import cv2
import re
import subprocess
import json
from ultralytics import YOLO
import math

def dms_to_decimal(dms_str, ref):
    match = re.match(r"(\d+) deg (\d+)' ([\d.]+)\" (\w)", dms_str)
    if not match:
        return None

    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

    if direction in ["S", "W"]:
        decimal *= -1

    return decimal

def show_resized_image(img, window_name="Detected", max_width=1200):
    height, width = img.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized = img
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_metadata(image_path):
    exiftool_path = r"C:\exiftool-13.18_64\exiftool.exe"
    try:
        result = subprocess.run(
            [exiftool_path, "-j", image_path], capture_output=True, text=True, check=True
        )
        metadata = json.loads(result.stdout)[0]

        gps_lat = dms_to_decimal(metadata.get("GPSLatitude", ""), metadata.get("GPSLatitudeRef", ""))
        gps_lon = dms_to_decimal(metadata.get("GPSLongitude", ""), metadata.get("GPSLongitudeRef", ""))
        rel_alt = float(metadata.get("RelativeAltitude", "0").replace("+", "").replace(" m", ""))

        print(f"[INFO] Координаты дрона: {gps_lat}, {gps_lon}, Высота: {rel_alt} м")

        return {
            "latitude": gps_lat,
            "longitude": gps_lon,
            "altitude": rel_alt
        }
    except Exception as e:
        print(f"[ERROR] Ошибка при чтении метаданных: {e}")
        return None

# Пути
image_path = r"C:\Users\Ludmila\Desktop\Все_датасеты\ТБО\10.04.2025 Мусор возле ККУЦ\DJI_0134.JPG"
model_path = r"C:/Users/Ludmila/Desktop/dump/YOLO_outputs/v2_yolo8s_hyp/weights/best.pt"

# Загрузка модели
model = YOLO(model_path)

# Загрузка изображения
orig_image = cv2.imread(image_path)

# Получение метаданных
meta = get_image_metadata(image_path)

# Предсказание
results = model.predict(source=image_path, imgsz=640, conf=0.5)

# Обработка результатов
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    print(f"[INFO] Найдено {len(boxes)} объектов")

    for box, cls, score in zip(boxes, classes, scores):

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        print(f" - Свалка: Центр рамки (x: {cx}, y: {cy})")

        # Расчёт географических координат центра свалки
        if meta and meta["latitude"] and meta["longitude"] and meta["altitude"]:
            image_h, image_w = orig_image.shape[:2]
            fov_degrees = 84  # угол обзора камеры по диагонали (для DJI Mini 2)
            fov_radians = fov_degrees * 3.14159265 / 180

            # Вычислим ширину охвата местности по ширине кадра
            ground_width_m = 2 * meta["altitude"] * (0.5 * fov_radians)  # приближённо
            meters_per_pixel = ground_width_m / image_w

            dx_meters = (cx - image_w / 2) * meters_per_pixel
            dy_meters = (cy - image_h / 2) * meters_per_pixel

            dlat = dy_meters / 111000  # 1° широты ≈ 111 км
            dlon = dx_meters / (111000 * abs(math.cos(math.radians(meta["latitude"]))))

            dump_lat = meta["latitude"] - dlat  # минус, т.к. пиксели растут вниз
            dump_lon = meta["longitude"] + dlon

            print(f"   Геокоординаты свалки: lat = {dump_lat:.6f}, lon = {dump_lon:.6f}")

        # Нарисовать прямоугольник и текст
        color = (0, 0, 255)
        thickness = 2
        label = f"{model.names[int(cls)]} {score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(orig_image, (x1, y1), (x1 + text_size[0] + 4, y1 + text_size[1] + 8), color, -1)
        cv2.putText(orig_image, label, (x1 + 2, y1 + text_size[1] + 2), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
        # Сначала белая окантовка
        cv2.circle(orig_image, (int(cx), int(cy)), radius=10, color=(255, 255, 255), thickness=-1)
        # Затем поверх — красный круг
        cv2.circle(orig_image, (int(cx), int(cy)), radius=8, color=(0, 0, 255), thickness=-1)

print(f"Размер изображения: {orig_image.shape[1]} x {orig_image.shape[0]}")

# показать изображение
show_resized_image(orig_image)
