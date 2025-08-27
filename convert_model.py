#!/usr/bin/env python3

import torch
import torch.onnx
import urllib.request
import os

def download_yolov8_onnx():
    """Download a pre-converted YOLOv8n ONNX model"""
    urls = [
        "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov8n.onnx",
        "https://raw.githubusercontent.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/main/models/yolov8n.onnx",
        "https://huggingface.co/spaces/jameslahm/yolov8_onnx/resolve/main/yolov8n.onnx"
    ]
    
    for url in urls:
        try:
            print(f"Trying to download from: {url}")
            urllib.request.urlretrieve(url, 'detector.onnx')
            
            # Check file size
            size = os.path.getsize('detector.onnx')
            print(f"Downloaded file size: {size} bytes")
            
            if size > 1000000:  # More than 1MB
                print("Successfully downloaded YOLOv8n ONNX model!")
                return True
            else:
                print("Downloaded file too small, trying next URL...")
                os.remove('detector.onnx')
                
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    return False

def create_dummy_classifier():
    """Create a dummy classifier ONNX model for testing"""
    import torch.nn as nn
    
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 5)  # 5 classes: red, yellow, green, off, unknown
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            x = torch.softmax(self.fc(x), dim=1)
            return x
    
    model = SimpleClassifier()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    torch.onnx.export(
        model,
        dummy_input,
        'classifier.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Created dummy classifier ONNX model!")

if __name__ == "__main__":
    print("Downloading models for Traffic Light Recognition system...")
    
    # Try to download YOLOv8n ONNX
    if not download_yolov8_onnx():
        print("Failed to download detector model from all sources")
        print("You can manually download YOLOv8n ONNX from:")
        print("   - https://github.com/ultralytics/ultralytics")
        print("   - Convert from PyTorch using: yolo export model=yolov8n.pt format=onnx")
    
    # Create dummy classifier
    try:
        create_dummy_classifier()
    except Exception as e:
        print(f"Failed to create classifier: {e}")
    
    print("\nNext steps:")
    print("1. Place models in: app/src/main/assets/models/")
    print("2. detector.onnx - YOLOv8n for traffic light detection")  
    print("3. classifier.onnx - Traffic light classification model")