import torch

# Загрузите вашу модель OCR
model = torch.load('./ocr_models/multihead/cer_0.4720965325832367.pth', map_location=torch.device('cpu'))

# Создайте фиктивный ввод для модели (соответствует входным данным модели)
dummy_input = torch.randn(1, 3, 240, 300)  # Измените размеры, если требуется

# Экспортируйте модель в ONNX
torch.onnx.export(model, dummy_input, "ocr_model.onnx", opset_version=11)
