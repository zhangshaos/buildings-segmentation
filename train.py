from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')
    model.train(model='yolov8n-seg.pt',
                data='./data/sz_dataset.yaml',
                epochs=3,
                imgsz=640,
                save=True,
                device=0,
                project='buildings-seg',
                name='exp0',
                pretrained=True)
    model.export(format='onnx')
    pass