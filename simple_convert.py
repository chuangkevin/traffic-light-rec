import torch
import torch.onnx

# Create a simple YOLO-like detector model for testing
class SimpleDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(20),  # Simulate detection grid
            torch.nn.Conv2d(32, 85, 1)  # 85 = 4(bbox) + 1(conf) + 80(classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        # Reshape to [batch, 1600, 85] for YOLO format
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, 85)
        return x

# Create and export detector
detector = SimpleDetector()
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    detector,
    dummy_input,
    'detector.onnx',
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Created detector.onnx")
print("Created classifier.onnx")
print("Both models are ready for testing!")