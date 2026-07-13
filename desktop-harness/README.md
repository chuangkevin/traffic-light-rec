# desktop-harness

在桌機上跑與手機完全相同的 core pipeline,驗證 openpilot 模型行為,不需要 Android 裝置。

## 準備測試幀

任一段行車影片(dashcam 視角、有等紅燈起步的路段最好):

- comma2k19 公開資料集(https://github.com/commaai/comma2k19)
- 或任何 YouTube 行車影片(yt-dlp 下載)

轉成幀(15fps 足夠):

    ffmpeg -i drive.mp4 -vf fps=15 -q:v 2 frames/%05d.jpg

## 執行

    ./gradlew :desktop-harness:run --args="frames out"

輸出:
- `out/events.csv` — 每幀 action/confidence/速度
- `out/annotated/*.png` — 疊加 STOP/GO 標籤的幀

合回影片檢視:

    ffmpeg -framerate 15 -i out/annotated/%05d.png -c:v libx264 out/annotated.mp4

## 傾斜模擬(驗證 roll 修正)

    ./gradlew :desktop-harness:run --args="frames out-tilt8 8"

把每幀旋轉 8° 並餵入 roll=8° 的 IMU 樣本;`out-tilt8/events.csv`
的 action 序列應與 `out/events.csv` 大致一致(warp 會把歪斜修正回來)。
純幾何層的自動驗證見 `src/test/.../TiltRegressionTest.kt`。
