import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import onnxruntime as ort
from training.detection_OCR import get_result  # Убедитесь, что этот модуль доступен

# Загрузка моделей
yolo_path = 'best.pt'  # Путь к модели YOLO
ocr_onnx_path = 'ocr_model.onnx'  # Путь к модели OCR в формате ONNX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO(yolo_path)
ocr_session = ort.InferenceSession(ocr_onnx_path)

# Создание интерфейса Streamlit
st.title("Штрих-код детектор и OCR")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Чтение изображения
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Сохранение изображения временно
    temp_image_path = "./temp_image.jpg"
    image.save(temp_image_path)

    # Выполнение инференса
    with st.spinner("Обработка изображения..."):
        result = get_result(temp_image_path, yolo_model, ocr_session, device)

    # Показ результатов
    st.subheader("Результаты")
    if result:
        st.json(result)
    else:
        st.write("Штрих-коды не обнаружены.")
