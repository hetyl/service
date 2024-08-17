import cv2
from augs import transform
import yaml
import numpy as np

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

alphabet_list = config['ALPHABET']
alphabet = ''.join(alphabet_list)
idx_to_char = {k: v for k, v in enumerate(alphabet_list, start=0)}


def recognize_OCR_onnx(image, ocr_session):
    img = image.astype(np.float32).transpose(2, 0, 1)  # Преобразуем в NCHW
    img = np.expand_dims(img, axis=0)  # Добавляем размерность batch

    ort_inputs = {ocr_session.get_inputs()[0].name: img}
    ort_outs = ocr_session.run(None, ort_inputs)

    # Проверяем структуру выходных данных
    print("Ort outputs:", len(ort_outs))

    # Предполагаем, что выходы содержат вероятности (batch_size, sequence_length, num_classes)
    #probs = ort_outs[0]
    probs = ort_outs

    probs = np.exp(probs - np.max(probs, axis=-1, keepdims=True))  # Стабильный softmax
    probs /= np.sum(probs, axis=-1, keepdims=True)

    # Извлекаем наиболее вероятные символы
    pred_indices = np.argmax(probs, axis=-1).flatten()


    # Декодируем индексы в символы
    pred_chars = [idx_to_char.get(index, '-') for index in pred_indices]

    # Формируем строку и удаляем символы '-'
    pred_str = ''.join(pred_chars).replace('-', '')
    return pred_str

    return pred_str


def get_result(im_path, yolo_model, OCR_model):
    image = cv2.imread(im_path)
    image = cv2.resize(image, (512, 512))

    output = []

    results = yolo_model.predict(source=image, conf=0.4)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            print(x1, y1, x2, y2)

            # Обрезка изображения по bounding box
            cropped_image = image[y1:y2, x1:x2]

            # Проверка ориентации и поворот, если бокс вертикальный
            if (x2 - x1) < (y2 - y1):
                cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

            height, width = cropped_image.shape[:2]
            new_height = int(height * 0.4)

            # Обрезаем изображение (оставляем только нижнюю часть)
            cropped_image = cropped_image[height - new_height:height, 0:width]

            augmented_image = transform(image=cropped_image)['image']
            code = recognize_OCR_onnx(augmented_image, OCR_model)
            output.append({'bbox': {"x_min": x1, "x_max": x2, "y_min": y1, "y_max": y2}, "value": code})
    return output
