import os
import sys
import time
import numpy as np
import PIL.Image as Image
import onnxruntime as ort
from ultralytics import YOLO


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: test.py <path-to-model.onnx> <path-to-picture>')
        exit(1)
    model_path, img_path = sys.argv[1], sys.argv[2]
    if model_path.endswith('.onnx'):
        use_onnx = True
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        use_onnx = False
    else:
        print(f'Unknown model filetype: {model_path}.')
        exit(1)
    if use_onnx:
        # ONNX推理
        # 测试集选择一张图片进行测试
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        print(f'onnx runtime device: {ort.get_device()}')
        opt = ort.SessionOptions()
        # opt.enable_profiling = True
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # opt.optimized_model_filepath = f'optmized_yolov8n-seg.onnx'
        ort_session = ort.InferenceSession(model_path, opt)
        in_s = ort_session.get_inputs()
        out_s = ort_session.get_outputs()
        # ort_input = {'images': img.convert_to(1,3,480,640}
        # t0 = time.time_ns()
        # ort_outs = ort_session.run(None, ort_input)
        # t1 = time.time_ns()
        # print(f'onnx inference cost time: {(t1 - t0) * 1e-6} ms.')
        # norm_out = ort_outs[3]
    else:
        yolo = YOLO(model_path)
        res = yolo.predict(img_path,
                           device='cpu',
                           show=False,
                           save=True,
                           save_txt=True,
                           save_conf=True,
                           save_crop=True)
    pass


