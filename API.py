from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import onnxruntime as ort
from detect_OCR import get_result

# Создаем экземпляр FastAPI
app = FastAPI()

# Загружаем модели
yolo_path = 'best.pt'  # Замените на путь к вашей модели YOLO
#OCR_path = './ocr_models/multihead/cer_0.4720965325832367.pth'  # Замените на путь к вашей модели OCR

#OCR_model = torch.load(OCR_path, map_location=device)
yolo_model = YOLO(yolo_path)

ocr_onnx_path = 'ocr_model.onnx'  # Замените на путь к вашей модели OCR в формате ONNX
ocr_session = ort.InferenceSession(ocr_onnx_path)

# Маршрут для инференса изображения
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Читаем изображение
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Сохраняем изображение временно, чтобы передать в инференс-функцию
    temp_image_path = "./temp_image.jpg"
    image.save(temp_image_path)

    # Выполняем инференс
    result = get_result(temp_image_path, yolo_model, ocr_session)

    # Возвращаем результат в виде JSON
    return JSONResponse(content={"barcodes": result})


test_im = './data/images/test_im.jpg'
print(get_result(test_im, yolo_model, ocr_session))


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    #uvicorn.run(app, host="0.0.0.0", port=8000)

