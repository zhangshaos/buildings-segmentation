import os
from ultralytics import YOLO


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = YOLO('yolov8n-seg.pt')
    model.train(data='./data/sz_dataset.yaml',
                epochs=10,
                imgsz=(640,480),
                save=True,
                device=0,
                project='buildings-seg',
                name='exp0',
                pretrained=True)
    model.export(format='onnx')
    pass