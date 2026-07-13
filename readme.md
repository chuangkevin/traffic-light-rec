# 🚦 車用 AI 行車提示系統(openpilot on Android)

在 Android 手機上跑 **comma.ai openpilot** 的 driving vision + policy 模型,
對駕駛提供 STOP/GO 語音提示(紅燈/前車停等情境減速、可起步時提示)。
本專案僅作行車輔助與 AI 研究用途,**不是自動駕駛**。

## 專案架構

```
:core             純 Kotlin/JVM,無 Android 依賴 —— 所有 AI 邏輯
 ├─ geometry/     Mat3、相機投影矩陣(含 roll 傾斜修正)
 ├─ frame/        影像轉正(rotate90)、warp、YUV12 打包
 ├─ calib/        IMU 傾斜計算 + 模型自校正融合
 ├─ plan/         policy 輸出解析(STOP/GO/HOLD)
 └─ pipeline/     端到端 DrivingPipeline(app 與桌機共用)

:app              Android 殼 —— CameraX、IMU 感測器、ONNX session、Overlay、音效
:desktop-harness  桌機回歸 —— 用行車影片跑同一套 pipeline,不需手機
```

## 功能特色

- **openpilot 模型推理**:雙幀 YUV12 輸入、25 幀特徵緩衝、policy 規劃解析
- **手機傾斜自動修正**:重力感測器計算 roll/pitch,歪斜擺放時影像自動轉正後才進模型;
  劇烈移動(重新擺放手機)自動重置時序緩衝
- **直向/橫向自動調整**:任意方向皆可用,幀依 rotationDegrees 轉正、
  UI 跟隨重排,切換過渡期靜音 1 秒
- **動態 FOV**:從 Camera2 characteristics 讀取實際鏡頭視角(含變焦),取代寫死常數
- **模型自校正**:行駛中由模型 pose 輸出低通細化 pitch/yaw/height

## 開發與驗證(不需要手機)

```bash
# 單元測試(core 42 tests + harness 傾斜回歸)
./gradlew :core:test :desktop-harness:test

# 編譯 APK
./gradlew :app:assembleDebug
```

### 桌機影片回歸

用真實行車影片驗證整條 pipeline(詳見 [desktop-harness/README.md](desktop-harness/README.md)):

```bash
ffmpeg -i drive.mp4 -vf fps=20 -q:v 2 frames/%05d.jpg
./gradlew :desktop-harness:run --args="frames out 0 93.4"      # baseline
./gradlew :desktop-harness:run --args="frames out-tilt8 8 93.4" # 8° 傾斜模擬
```

輸出 `events.csv`(逐幀 action/速度)與標注影片幀。
已驗證:comma 公開路段 400 幀,8° 傾斜輸入經 IMU 修正後
action 序列與正立基準 **100% 一致**(速度平均差 0.084 m/s)。

## 上機煙霧測試清單

桌機已驗證邏輯,手機端只需確認:

1. 相機開啟、畫面有 overlay
2. STOP/GO 有音效
3. 直向 ↔ 橫向切換不 crash、UI 正確重排
4. 手機故意歪 10° 擺放,行駛中 overlay 路徑仍貼合路面

## 模型

`app/src/main/assets/models/`:
- `openpilot_driving_vision.onnx`(89MB)— vision model,輸入 2×6×128×256 YUV12 雙幀
- `openpilot_driving_policy.onnx`(27MB)— policy model,輸入 25 幀特徵緩衝

來源與轉換見 [MODEL_SETUP.md](MODEL_SETUP.md)。

## 環境需求

- JDK 17、Android SDK(platform 35)
- minSdk 26 / targetSdk 35
- 測試裝置:POCO F5 Pro(Snapdragon 8+ Gen1)
- ONNX Runtime 1.19.2(android + JVM)、CameraX 1.4.0

## 🔒 注意事項

- ✅ **輔助功能** — 僅作行車輔助提示
- ⛔ **非自動駕駛** — 不可依賴本系統進行駕駛決策
- 🚗 **駕駛責任** — 實際駕駛以真實號誌與路況為準
- 📱 **使用安全** — 避免裝置遮蔽駕駛視線

### 已知限制

openpilot 模型**沒有明確的紅綠燈顏色輸出**;STOP/GO 來自端到端 policy 的
規劃速度(模型「想停」可能是紅燈、停止線或前車)。若需明確紅綠燈狀態,
規劃中的 Phase 2 將參照 dragonpilot SightSense 架構外掛 YOLO 偵測器。
