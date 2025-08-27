# 🧠 AI 模型設定指南

本專案需要兩個 ONNX 模型檔案才能運行：

## 📥 模型檔案下載

### 1. 偵測模型 (detector.onnx)
使用 **YOLOv8n** 進行紅綠燈偵測：

```bash
# 安裝 ultralytics
pip install ultralytics

# 匯出 YOLOv8n 模型為 ONNX 格式
yolo export model=yolov8n.pt format=onnx

# 重命名檔案
mv yolov8n.onnx detector.onnx
```

### 2. 分類模型 (classifier.onnx)  
有兩種選擇：

#### 選擇 A：從 Roboflow Universe 下載 (推薦)
1. 訪問 [Roboflow Universe](https://universe.roboflow.com/)
2. 搜尋 "traffic light classification" 
3. 選擇適合的資料集並匯出為 ONNX 格式
4. 重命名為 `classifier.onnx`

#### 選擇 B：自訓練 MobileNetV3-S 模型
```python
# 使用 PyTorch 訓練範例
import torch
import torchvision.models as models

# 載入 MobileNetV3-Small 並修改輸出層
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)

# 訓練模型... (需要紅綠燈分類資料集)
# 類別：0=RED, 1=YELLOW, 2=GREEN, 3=OFF, 4=UNKNOWN

# 匯出為 ONNX
dummy_input = torch.randn(1, 3, 64, 64)
torch.onnx.export(model, dummy_input, "classifier.onnx")
```

## 📂 模型放置位置

將下載好的模型檔案放到以下位置：

```
app/src/main/assets/models/
├── detector.onnx      # YOLOv8n 偵測模型
└── classifier.onnx    # MobileNetV3-S 分類模型
```

## ⚙️ 模型規格要求

### 偵測模型 (detector.onnx)
- **輸入尺寸**: 640x640x3 (RGB)
- **輸出格式**: YOLO 格式 (bbox + confidence + class)
- **支援類別**: traffic light (COCO class 9)

### 分類模型 (classifier.onnx)  
- **輸入尺寸**: 64x64x3 (RGB)
- **輸出格式**: 5個類別的機率分佈
- **類別對應**:
  - 0: 紅燈 (RED)
  - 1: 黃燈 (YELLOW) 
  - 2: 綠燈 (GREEN)
  - 3: 關閉 (OFF)
  - 4: 未知 (UNKNOWN)

## 🔧 測試與驗證

放置模型後，可以透過以下方式驗證：

1. **建構專案**: `./gradlew build`
2. **安裝到裝置**: 連接 POCO F5 Pro 並執行
3. **檢查 Log**: 查看 `InferenceEngine` 的初始化訊息

## 📊 效能調整

針對 **Snapdragon 8+ Gen1** 優化：

- 偵測每 3 幀執行一次
- 分類每幀執行  
- 可調整 `FrameAnalyzer.kt` 中的間隔設定

## 🚨 注意事項

1. **檔案大小**: ONNX 模型已加入 `.gitignore`，不會上傳到 Git
2. **授權問題**: 確認使用的預訓練模型符合授權要求
3. **效能測試**: 建議先用小模型測試，再上升到生產模型

## 🔗 相關資源

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime Android](https://onnxruntime.ai/docs/get-started/with-android.html)
- [Roboflow Universe](https://universe.roboflow.com/)