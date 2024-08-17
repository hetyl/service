Проект по детекции штрихкода и распознаванию его цифр. 

Для готового использования:
В файле model_inference измените пути на ваши.
Далее будет выведен результат.

Для воспроизведения результатов:
1) скачайте датасет по ссылке https://disk.yandex.ru/d/pRFNuxLQUZcDDg
2) подготовьте данные для yolo при помощи файла create_yolo_data.py
3) обучите модель yolo:
yolo task=detect mode=train model=yolov10x.pt data=yolo_dataset.yaml epochs=100 imgsz=512 batch=8 half=True
4) при помощи файла cut_bboxes.py подготовьте данные для обучения OCR модели
5) при помощи файла train_ocr.py обучите модель OCR
6) в файле model_inference измените пути на ваши и проверьте результат.
