from train_common import run_training, MODELS, VARIANTS

for model_name in MODELS:          # yolov8n, yolo11n
    for variant in VARIANTS:      
        run_training(variant, model_name)