# train.py

import os
import yaml
import torch
from ultralytics import YOLO
from ultralytics.model.yolo.loss import ComputeLoss  # Ensure correct path

import logging

logging.basicConfig(level=logging.DEBUG)

def get_model_path(model_type, size):
    base_path = os.path.join('.', 'ultralytics', 'cfg', 'models', 'v8')
    model_paths = {
        'basic': f"yolov8{size}.yaml",
        'sa': f"yolov8{size}_SA.yaml",
        'ca': f"yolov8{size}_CA.yaml",
        'rescbam': f"yolov8{size}_ResCBAM.yaml",
        'scsa': f"yolov8{size}_SCSA_CWM.yaml"
    }
    model_file = model_paths.get(model_type.lower())
    if not model_file:
        raise ValueError(f"Invalid model type: {model_type}")
    return os.path.join(base_path, model_file)

def get_experiment_name(model_type, size):
    name_map = {
        'basic': 'Basic',
        'sa': 'SA',
        'ca': 'CA',
        'rescbam': 'ResCBAM',
        'scsa': 'SCSA'
    }
    model_name = name_map.get(model_type.lower())
    if not model_name:
        raise ValueError(f"Invalid model type: {model_type}")
    return f"{model_name}({size})"

def load_class_weights(data_path):
    """Load class weights from the dataset.yaml file"""
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    class_weights = data.get('class_weights', [1.0] * data['nc'])
    return torch.tensor(class_weights).float()

def train(model_type, data_path, img_size, size):
    model_path = get_model_path(model_type, size)
    experiment_name = get_experiment_name(model_type, size)
    
    # Load class weights from dataset.yaml
    class_weights_tensor = load_class_weights(data_path)
    
    os.makedirs('./logs', exist_ok=True)
    
    model = YOLO(model_path)
    
    # Pass class weights to the model's loss function
    # Assuming the model has a 'model' attribute containing the architecture and loss
    if hasattr(model.model, 'loss'):
        model.model.loss = ComputeLoss(model.model, class_weights=class_weights_tensor)
    else:
        # Alternative approach: access the loss function differently
        try:
            model.loss_fn = ComputeLoss(model.model, class_weights=class_weights_tensor)
        except AttributeError:
            raise AttributeError("Model does not have a 'loss' attribute. Please check the model's architecture.")
    
    # Start training
    model.train(
        data=data_path,
        imgsz=img_size,
        epochs=100,
        name=experiment_name,
        project='./logs'
        # Additional parameters can be added here if needed
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model")
    parser.add_argument('model_type', help="Model type: basic, sa, ca, rescbam, scsa")
    parser.add_argument('data_path', help="Path to the dataset YAML file")
    parser.add_argument('--img_size', type=int, default=512, help="Image size for training (default: 512)")
    parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l', 'x'], help="Model size: n, s, m, l, x (default: s)")
    args = parser.parse_args()
    train(args.model_type, args.data_path, args.img_size, args.size)

