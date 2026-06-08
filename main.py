import os
import glob
import cv2
from ultralytics import YOLO


# =====================================
# Настройки
# =====================================

IMAGES_DIR = "images"
OUTPUT_DIR = "output"

TARGET_CLASS = "cell phone"
CONFIDENCE_THRESHOLD = 0.5


# =====================================
# Вспомогательные функции
# =====================================

def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Не удалось загрузить изображение: {image_path}"
        )

    return image


def get_image_files():
    """
    Получить все jpg-файлы из папки images.
    """

    image_files = glob.glob(
        os.path.join(IMAGES_DIR, "*.jpg")
    )

    image_files.sort()

    return image_files



# Обработка одного изображения
def process_image(model, image_path):

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    print(f"\n[INFO] Обработка {image_name}")

    image = load_image(image_path)

    original_image = image.copy()

    results = model(image)

    detected_count = 0

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name != TARGET_CLASS:
                continue

            detected_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            print(
                f"[FOUND] Телефон #{detected_count} "
                f"(confidence={confidence:.2f})"
            )

            # Рисуем рамку

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Подпись

            label = f"PHONE {confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Вырезаем объект

            crop = original_image[
                y1:y2,
                x1:x2
            ]

            crop_path = os.path.join(
                OUTPUT_DIR,
                f"{image_name}_phone_{detected_count}.jpg"
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            print(
                f"[SAVE] {crop_path}"
            )

    # Сохраняем изображение с рамками

    detected_image_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_detected.jpg"
    )

    cv2.imwrite(
        detected_image_path,
        image
    )

    print(
        f"[INFO] Найдено телефонов: {detected_count}"
    )
    print(
        f"[SAVE] {detected_image_path}"
    )
    return detected_count


def main():

    print("[INFO] Загрузка модели YOLO...")

    model = YOLO("yolov8n.pt")

    print("[INFO] Модель загружена")

    create_output_directory()

    image_files = get_image_files()

    if not image_files:
        raise FileNotFoundError(
            "В папке images не найдено ни одного JPG-файла"
        )

    total_images = 0
    total_phones = 0

    for image_path in image_files:

        total_images += 1

        phones_found = process_image(
            model,
            image_path
        )

        total_phones += phones_found



    print(
        f"Обработано изображений: {total_images}"
    )

    print(
        f"Всего найдено телефонов: {total_phones}"
    )

    print(
        f"Результаты сохранены в папку '{OUTPUT_DIR}'"
    )




if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print(f"[ERROR] {error}")