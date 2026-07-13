# 設計:openpilot pipeline 補強 — 手機傾斜自動修正 + 直橫向支援

日期:2026-07-13
狀態:已核准(v2)

## 背景與目標

本 App 在手機上跑 openpilot 的 driving vision + policy 模型,對駕駛提供
STOP/GO 語音提示(紅燈/前車停等情境減速、起步情境提示)。現有
`InferenceEngine.kt` 已完成雙幀 YUV12 打包、25 幀特徵緩衝、
模型 pose 輸出自校正(pitch/yaw 低通)與 warp 矩陣。

本次目標:

1. **手機傾斜角度自動修正**:手機任意角度擺放(歪斜/俯仰)時,
   影像在進模型前自動轉正。
2. **直向/橫向自動調整**:螢幕不鎖向,直橫向皆可用,UI 與偵測跟隨調整。
3. **可在桌機完整驗證**:不上機即可驗證 pipeline 正確性。

目標裝置:POCO F5 Pro(Snapdragon 8+ Gen1, Android 13+)。

## 不變的部分

- openpilot vision + policy 雙模型推理與 STOP/GO/HOLD 判斷邏輯
- 雙幀 YUV12 打包格式(對齊 openpilot `frames_to_tensor`)
- 25 幀 hidden-state 特徵緩衝、desire/traffic_convention 輸入
- 模型自校正(pose/road transform 輸出 → pitch/yaw/height 低通濾波)
- TTS 播報與 Overlay 顯示架構

## 新增 1:IMU 輔助傾斜修正

現有弱點:

- warp 的 roll 固定為 0 → 手機左右歪斜時,模型輸入的地平線是斜的。
- 模型自校正需累積 20 樣本;冷啟動期間 pitch 寫死 5.5°。

設計:

- 訂閱 `TYPE_ROTATION_VECTOR`(fallback:`TYPE_GRAVITY`)。
- **Roll**:由重力方向即時計算「相機相對地平線的滾轉角」,
  經低通濾波後餵入 `rotationFromEuler(roll = imuRoll)`。
- **Pitch**:IMU 值作為校正初始值(取代寫死 5.5°),模型自校正
  繼續以低通細化(模型量測的是相對路面、比 IMU 相對重力更準)。
- **Yaw**:IMU 無法量測(相對車頭方向),維持純模型自校正。
- IMU 濾波:一階低通(alpha 依取樣率調整),車輛震動視為高頻噪音。
- 突變偵測:roll/pitch 短時間變化超過閾值(手機被重新擺放)→
  重置雙幀緩衝與特徵緩衝、校正樣本計數歸零。

## 新增 2:直向/橫向自動調整

- Activity 解除方向鎖定(`sensor`),`configChanges` 處理旋轉不重建。
- 每一幀依 `ImageProxy.imageInfo.rotationDegrees` + 顯示方向
  **先轉正成正立(地平線水平)影像**再進 warp;模型層不感知方向。
- source intrinsics 依實際(轉正後)幀尺寸重算;
  相機水平 FOV 從 Camera2 characteristics 讀取(焦距+感光元件尺寸),
  取代寫死的 72°。
- 方向改變(90° 跳變)→ 視野劇變 → 自動重置時序緩衝。
- Overlay 座標:用 warp 逆矩陣把模型座標映回螢幕,直橫向共用同一數學。
- UI:`layout/` 與 `layout-land/` 各一套;狀態列直向置頂、橫向靠側。

## 新增 3:可測架構(不上機驗證)

模組劃分:

```
core(純 Kotlin/JVM,無 Android 依賴)
 ├─ geometry/   warp 矩陣、rotationFromEuler、3x3 運算、逆矩陣
 ├─ frame/      YUV12 打包、center-crop、影像抽象(PixelReader 介面)
 ├─ calib/      校正狀態(模型低通 + IMU 融合 + 突變重置)
 ├─ plan/       policy 輸出解析、STOP/GO 判斷、時間插值
 └─ pipeline/   端到端組裝(給 app 與 desktop-harness 共用)
app(Android)
 ├─ CameraX 取像、rotationDegrees 轉正、Bitmap→PixelReader
 ├─ IMU 感測器接入 → core/calib
 └─ Overlay + TTS
desktop-harness(JVM main,ONNX Runtime 桌面版)
 └─ 讀行車影片 → 同一套 core pipeline → 標注影片 + 事件 log
```

Android `Bitmap` 不進 core;core 以 `PixelReader`(width/height/getPixel)
介面操作影像,app 端以 Bitmap 實作、桌機以 BufferedImage 實作。

測試計畫:

1. **單元測試(JUnit)**:3x3 數學、rotationFromEuler、YUV 打包
   (與參考實作比對)、plan 解析(合成 policy 向量)、校正濾波收斂、
   IMU 突變重置。
2. **桌機影片回歸**:desktop-harness 跑 comma 公開行車片段,
   輸出 STOP/GO 事件 log(golden file 回歸比對)+ 標注影片(人工驗收)。
3. **傾斜修正驗證**:同一影片幀人工旋轉已知 roll(如 8°)後餵入
   帶 IMU 修正的 pipeline,斷言 warp 輸出與未旋轉版本近似
   (像素 diff 閾值)且 STOP/GO 事件一致。
4. **方向切換測試**:模擬直→橫幀序列,斷言緩衝重置、無座標錯亂。
5. **上機煙霧測試(唯一手動項)**:相機開啟、有框有聲音、
   實際震動下 IMU 濾波不抖。

## 錯誤處理

- IMU 不可用 → 退回純模型自校正(現行為)。
- 方向切換過渡期(緩衝重置後 1 秒)→ 暫停播報,Overlay 顯示「調整中」。
- 模型載入失敗 → 明確錯誤畫面,不 crash。

## 非目標(Phase 2+)

- 明確紅綠燈顏色辨識(需另掛 YOLO,參照 dragonpilot SightSense 架構)
- 前車距離的顯式輸出(目前依賴 policy 隱含行為)
- NNAPI/GPU delegate 效能調校
