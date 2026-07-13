# 🧠 模型設定指南

本專案使用 **comma.ai openpilot** 的兩個 ONNX 模型:

```
app/src/main/assets/models/
├── openpilot_driving_vision.onnx   # vision model(~89MB)
└── openpilot_driving_policy.onnx   # policy model(~27MB)
```

## 來源

模型來自 [commaai/openpilot](https://github.com/commaai/openpilot) 的
`selfdrive/modeld/models/`(driving_vision / driving_policy)。
留意 openpilot 的授權條款(MIT)與模型版本——輸出張量的欄位索引
(pose/wide euler/road transform/hidden state 起點)與 core 內常數綁定,
換模型版本時需同步核對 `CalibrationFusion` 與 `DrivingPipeline` 的偏移常數。

## 輸入格式

### vision model
- `img`、`big_img`:各 `1×12×128×256` uint8
  = 連續兩幀 × 6 通道 YUV12(Y 四相位 + U + V,各 128×256)
- 影像須先經相機校正 warp 到 MEDMODEL(fl=910)/ SBIGMODEL(fl=455)幀
- 打包實作:`core/frame/YuvPacker.kt`;warp:`core/geometry/CameraProjection.kt`

### policy model
- `features_buffer`:`1×25×512`(vision hidden state 滑動緩衝)
- `desire_pulse`:`1×25×8`(全零)
- `traffic_convention`:`1×2` = `[1, 0]`(靠右行駛)

## 輸出解析

policy 輸出前 33×15 為規劃軌跡(x, y, ..., v@idx3, a@idx6),
解析邏輯在 `core/plan/PlanParser.kt`。STOP/GO 判斷閾值同檔案。

## 驗證

```bash
./gradlew :core:test                 # 單元測試
./gradlew :desktop-harness:run --args="frames out"   # 真實影片回歸
```
