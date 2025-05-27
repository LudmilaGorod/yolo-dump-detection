import os
import shutil
import random

# Папка с исходными изображениями
image_dir = r"C:\Users\Ludmila\Desktop\dump\v2_all\images"

# Папка с соответствующими аннотациями (YOLO-формат: *.txt)
label_dir = r"C:\Users\Ludmila\Desktop\dump\v2_all\labels"

# Базовая директория для нового разбиения датасета
dataset_base = r"C:\Users\Ludmila\Desktop\dump\v2_div"

# Задаём структуру новых папок для train/val/test
split_dirs = {
    "train": {
        "images": os.path.join(dataset_base, "train", "images"),
        "labels": os.path.join(dataset_base, "train", "labels")
    },
    "val": {
        "images": os.path.join(dataset_base, "val", "images"),
        "labels": os.path.join(dataset_base, "val", "labels")
    },
    "test": {
        "images": os.path.join(dataset_base, "test", "images"),
        "labels": os.path.join(dataset_base, "test", "labels")
    },
}

# Создаём папки, если они ещё не существуют
for split in split_dirs.values():
    for path in split.values():
        os.makedirs(path, exist_ok=True)

# Получаем список всех файлов изображений с нужными расширениями
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Перемешиваем список изображений для случайного разбиения
random.shuffle(image_files)

# Считаем общее количество изображений
num_total = len(image_files)

# Определяем количество файлов для train/val/test
n_train = int(num_total * 0.7)  # 70% — обучение
n_val = int(num_total * 0.2)    # 20% — валидация
n_test = num_total - n_train - n_val  # оставшиеся 10% — тест

# Формируем словарь с разделёнными наборами данных
dataset_splits = {
    "train": image_files[:n_train],
    "val": image_files[n_train:n_train + n_val],
    "test": image_files[n_train + n_val:]
}

# Функция для перемещения изображений и соответствующих аннотаций
def move_files(file_list, split_name):
    for file_name in file_list:
        # Полный путь к изображению
        src_image = os.path.join(image_dir, file_name)
        dst_image = os.path.join(split_dirs[split_name]["images"], file_name)

        # Формируем имя и путь к аннотации, заменив расширение на .txt
        label_name = os.path.splitext(file_name)[0] + ".txt"
        src_label = os.path.join(label_dir, label_name)
        dst_label = os.path.join(split_dirs[split_name]["labels"], label_name)

        # Перемещаем изображение
        shutil.move(src_image, dst_image)

        # Перемещаем аннотацию, если она существует
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

# Запускаем перенос файлов по каждому из трёх наборов
for split_name, files in dataset_splits.items():
    move_files(files, split_name)

# Выводим сообщение по завершению
print("Разделение данных завершено!")
