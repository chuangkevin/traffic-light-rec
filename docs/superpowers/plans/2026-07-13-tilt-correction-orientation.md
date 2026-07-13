# Tilt Correction + Orientation Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 openpilot pipeline 在手機任意傾斜角度與直/橫向下都輸入正確轉正的影像,且整條 pipeline 可在桌機驗證。

**Architecture:** 把 `InferenceEngine.kt` 內的純邏輯(3x3 幾何、warp、YUV 打包、plan 解析、校正濾波)抽到新的純 JVM `:core` 模組,新增 IMU roll/pitch 融合;`:app` 變成薄殼(CameraX/IMU/ORT session/UI);新增 `:desktop-harness` 用桌面版 ONNX Runtime 跑同一套 core 對行車影片做回歸。

**Tech Stack:** Kotlin 1.9.10, Gradle 8.13, ONNX Runtime 1.19.2(android + JVM), CameraX 1.4.0, JUnit 4.

## Global Constraints

- 套件名:core 內為 `com.example.trafficlight.core.*`
- YUV12 打包格式必須逐 byte 等同現有 `packOpenpilotFrame`(openpilot `frames_to_tensor` 順序:Y-TL, Y-BL, Y-TR, Y-BR, U, V)
- 模型檔不動:`app/src/main/assets/models/openpilot_driving_{vision,policy}.onnx`
- 所有既有常數值保留:MEDMODEL_FL=910, MEDMODEL_CX=256, MEDMODEL_CY=47.6, SBIGMODEL_FL=455, SBIGMODEL_CX=256, SBIGMODEL_CY=151.8, POLICY_FRAMES=25, FEATURE_LEN=512, PLAN_WIDTH=15
- `:core` 與其測試不得 import 任何 `android.*` / `androidx.*`
- 每個 task 結尾 commit;commit message 結尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- 測試指令一律 `./gradlew :core:test`(或 harness 對應),不需要 Android 裝置

---

### Task 1: 建立 :core 模組與 Mat3 幾何

**Files:**
- Modify: `settings.gradle`(最後一行 `include ':app'` 改為 `include ':app', ':core'`)
- Modify: `build.gradle`(plugins 區塊加 `id 'org.jetbrains.kotlin.jvm' version '1.9.10' apply false`)
- Create: `core/build.gradle`
- Create: `core/src/main/kotlin/com/example/trafficlight/core/geometry/Mat3.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/geometry/Mat3Test.kt`

**Interfaces:**
- Produces: `Mat3(val m: FloatArray)`(row-major 9 元素)、`operator fun times(o: Mat3): Mat3`、`fun invert(): Mat3?`、`fun map(x: Float, y: Float): Pair<Float, Float>`(齊次除法)、`companion { identity(), rotationFromEuler(roll,pitch,yaw), intrinsics(fl,cx,cy) }`

- [ ] **Step 1: core/build.gradle**

```groovy
plugins {
    id 'org.jetbrains.kotlin.jvm'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

dependencies {
    testImplementation 'junit:junit:4.13.2'
}
```

同步修改 `settings.gradle` 與根 `build.gradle`(如上)。

- [ ] **Step 2: 寫失敗測試 Mat3Test.kt**

```kotlin
package com.example.trafficlight.core.geometry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class Mat3Test {
    private fun assertMat(expected: FloatArray, actual: Mat3, eps: Float = 1e-4f) {
        for (i in 0..8) assertEquals("index $i", expected[i], actual.m[i], eps)
    }

    @Test fun identityTimesIsSame() {
        val a = Mat3(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 10f))
        assertMat(a.m, Mat3.identity() * a)
    }

    @Test fun invertRecoversIdentity() {
        val a = Mat3(floatArrayOf(2f, 0f, 1f, 0f, 3f, 0f, 0f, 0f, 1f))
        val inv = a.invert()
        assertNotNull(inv)
        assertMat(Mat3.identity().m, a * inv!!)
    }

    @Test fun singularReturnsNull() {
        val a = Mat3(FloatArray(9) { 1f })
        assertEquals(null, a.invert())
    }

    @Test fun eulerZeroIsIdentity() {
        assertMat(Mat3.identity().m, Mat3.rotationFromEuler(0f, 0f, 0f))
    }

    @Test fun rollRotatesAroundX() {
        // openpilot 順序 yaw*pitch*roll;roll=90° 時 (0,1,0)->(0,0,1)
        val r = Mat3.rotationFromEuler((Math.PI / 2).toFloat(), 0f, 0f)
        val (x, y) = r.map(0f, 1f) // map 對 (0,1,1) 齊次;此處直接驗證矩陣元素
        assertEquals(0f, r.m[4], 1e-4f)  // m[1][1] = cos90 = 0
        assertEquals(-1f, r.m[5], 1e-4f) // m[1][2] = -sin90
    }

    @Test fun mapAppliesHomogeneousDivide() {
        val scale2 = Mat3(floatArrayOf(2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f))
        val (x, y) = scale2.map(3f, 4f)
        assertEquals(3f, x, 1e-4f)
        assertEquals(4f, y, 1e-4f)
    }
}
```

- [ ] **Step 3: 跑測試確認編譯失敗**

Run: `./gradlew :core:test`
Expected: FAIL(`Unresolved reference: Mat3`)

- [ ] **Step 4: 實作 Mat3.kt**

從 `InferenceEngine.kt:514-580` 的 `rotationFromEuler`/`multiply3x3`/`invert3x3`/`identity3x3` 移植,改為 row-major FloatArray:

```kotlin
package com.example.trafficlight.core.geometry

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class Mat3(val m: FloatArray) {
    init { require(m.size == 9) }

    operator fun times(o: Mat3): Mat3 {
        val r = FloatArray(9)
        for (row in 0..2) for (col in 0..2) {
            r[row * 3 + col] = m[row * 3] * o.m[col] +
                m[row * 3 + 1] * o.m[3 + col] +
                m[row * 3 + 2] * o.m[6 + col]
        }
        return Mat3(r)
    }

    fun map(x: Float, y: Float): Pair<Float, Float> {
        val w = m[6] * x + m[7] * y + m[8]
        return Pair(
            (m[0] * x + m[1] * y + m[2]) / w,
            (m[3] * x + m[4] * y + m[5]) / w
        )
    }

    fun invert(): Mat3? {
        val det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
            m[1] * (m[3] * m[8] - m[5] * m[6]) +
            m[2] * (m[3] * m[7] - m[4] * m[6])
        if (!det.isFinite() || abs(det) < 1e-6f) return null
        val i = 1f / det
        return Mat3(floatArrayOf(
            (m[4] * m[8] - m[5] * m[7]) * i, (m[2] * m[7] - m[1] * m[8]) * i, (m[1] * m[5] - m[2] * m[4]) * i,
            (m[5] * m[6] - m[3] * m[8]) * i, (m[0] * m[8] - m[2] * m[6]) * i, (m[2] * m[3] - m[0] * m[5]) * i,
            (m[3] * m[7] - m[4] * m[6]) * i, (m[1] * m[6] - m[0] * m[7]) * i, (m[0] * m[4] - m[1] * m[3]) * i
        ))
    }

    companion object {
        fun identity() = Mat3(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f))

        fun intrinsics(fl: Float, cx: Float, cy: Float) =
            Mat3(floatArrayOf(fl, 0f, cx, 0f, fl, cy, 0f, 0f, 1f))

        /** openpilot 順序:yaw * pitch * roll */
        fun rotationFromEuler(roll: Float, pitch: Float, yaw: Float): Mat3 {
            val cr = cos(roll); val sr = sin(roll)
            val cp = cos(pitch); val sp = sin(pitch)
            val cy = cos(yaw); val sy = sin(yaw)
            val rollM = Mat3(floatArrayOf(1f, 0f, 0f, 0f, cr, -sr, 0f, sr, cr))
            val pitchM = Mat3(floatArrayOf(cp, 0f, sp, 0f, 1f, 0f, -sp, 0f, cp))
            val yawM = Mat3(floatArrayOf(cy, -sy, 0f, sy, cy, 0f, 0f, 0f, 1f))
            return yawM * (pitchM * rollM)
        }
    }
}
```

- [ ] **Step 5: 跑測試確認通過**

Run: `./gradlew :core:test`
Expected: PASS(6 tests)

- [ ] **Step 6: Commit**

```bash
git add settings.gradle build.gradle core
git commit -m "feat(core): add pure-JVM core module with Mat3 geometry"
```

---

### Task 2: 相機投影矩陣(含 roll)

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/geometry/CameraProjection.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/geometry/CameraProjectionTest.kt`

**Interfaces:**
- Consumes: `Mat3`
- Produces: `object ModelFrames`(常數)與
  `fun sourceFromModelFrame(sourceWidth: Float, sourceHeight: Float, horizontalFovDeg: Float, rollRad: Float, pitchRad: Float, yawRad: Float, bigModelFrame: Boolean): Mat3`
  —— 將模型幀像素座標映射到來源影像像素座標(即 inverse-mapping 取樣用)。

- [ ] **Step 1: 寫失敗測試**

```kotlin
package com.example.trafficlight.core.geometry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraProjectionTest {
    // 模型幀中心(cx, cy)在零角度時應映射到來源影像中心
    @Test fun zeroAnglesMapsPrincipalPointToImageCenter() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, bigModelFrame = false)
        val (x, y) = m.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        assertEquals(960f, x, 1.0f)
        assertEquals(540f, y, 1.0f)
    }

    // roll > 0 時,模型幀主點左右兩側的取樣點在來源影像中的 y 應不同(地平線傾斜補償)
    @Test fun rollTiltsSampling() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0.14f, 0f, 0f, bigModelFrame = false)
        val (_, yLeft) = m.map(ModelFrames.MEDMODEL_CX - 100f, ModelFrames.MEDMODEL_CY)
        val (_, yRight) = m.map(ModelFrames.MEDMODEL_CX + 100f, ModelFrames.MEDMODEL_CY)
        assertTrue("roll should skew sampling rows", kotlin.math.abs(yLeft - yRight) > 10f)
    }

    // roll = 0 時左右對稱
    @Test fun zeroRollIsSymmetric() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, bigModelFrame = false)
        val (_, yLeft) = m.map(ModelFrames.MEDMODEL_CX - 100f, ModelFrames.MEDMODEL_CY)
        val (_, yRight) = m.map(ModelFrames.MEDMODEL_CX + 100f, ModelFrames.MEDMODEL_CY)
        assertEquals(yLeft, yRight, 0.5f)
    }

    // pitch 增加(相機朝上修正)時取樣區應往影像下方移動或至少改變
    @Test fun pitchShiftsVertically() {
        val m0 = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, false)
        val m1 = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0.1f, 0f, false)
        val (_, y0) = m0.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        val (_, y1) = m1.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        assertTrue(kotlin.math.abs(y0 - y1) > 20f)
    }

    @Test fun bigFrameUsesWiderIntrinsics() {
        val med = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, false)
        val big = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, true)
        // 大幀焦距一半 → 同一模型像素位移對應兩倍來源位移
        val (xm, _) = med.map(ModelFrames.MEDMODEL_CX + 50f, ModelFrames.MEDMODEL_CY)
        val (xb, _) = big.map(ModelFrames.SBIGMODEL_CX + 50f, ModelFrames.SBIGMODEL_CY)
        assertTrue((xb - 960f) > (xm - 960f) * 1.5f)
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*CameraProjectionTest*'`
Expected: FAIL(Unresolved reference)

- [ ] **Step 3: 實作 CameraProjection.kt**

移植 `InferenceEngine.kt:476-512`(`sourceFromModelFrameMatrix`/`calibFromModelFrame`),
差異:(a) FOV 變參數;(b) `deviceFromCalib` 的 roll 不再固定 0:

```kotlin
package com.example.trafficlight.core.geometry

import kotlin.math.tan

object ModelFrames {
    const val MEDMODEL_FL = 910f
    const val MEDMODEL_CX = 256f
    const val MEDMODEL_CY = 47.6f
    const val SBIGMODEL_FL = 455f
    const val SBIGMODEL_CX = 256f
    const val SBIGMODEL_CY = 151.8f
    const val MODEL_WIDTH = 512
    const val MODEL_HEIGHT = 256
}

private val VIEW_FROM_DEVICE = Mat3(floatArrayOf(
    0f, 1f, 0f,
    0f, 0f, 1f,
    1f, 0f, 0f
))

fun sourceFromModelFrame(
    sourceWidth: Float,
    sourceHeight: Float,
    horizontalFovDeg: Float,
    rollRad: Float,
    pitchRad: Float,
    yawRad: Float,
    bigModelFrame: Boolean
): Mat3 {
    val fovRad = Math.toRadians(horizontalFovDeg.toDouble() / 2.0)
    val sourceFl = sourceWidth / (2f * tan(fovRad).toFloat())
    val sourceIntrinsics = Mat3.intrinsics(sourceFl, sourceWidth / 2f, sourceHeight / 2f)
    val deviceFromCalib = Mat3.rotationFromEuler(rollRad, pitchRad, yawRad)
    val cameraFromCalib = sourceIntrinsics * VIEW_FROM_DEVICE * deviceFromCalib

    val modelFl = if (bigModelFrame) ModelFrames.SBIGMODEL_FL else ModelFrames.MEDMODEL_FL
    val modelCx = if (bigModelFrame) ModelFrames.SBIGMODEL_CX else ModelFrames.MEDMODEL_CX
    val modelCy = if (bigModelFrame) ModelFrames.SBIGMODEL_CY else ModelFrames.MEDMODEL_CY
    val calibFromModel = (Mat3.intrinsics(modelFl, modelCx, modelCy) * VIEW_FROM_DEVICE).invert()
        ?: Mat3.identity()
    return cameraFromCalib * calibFromModel
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `./gradlew :core:test --tests '*CameraProjectionTest*'`
Expected: PASS(5 tests)

- [ ] **Step 5: Commit**

```bash
git add core
git commit -m "feat(core): camera projection matrix with roll support"
```

---

### Task 3: IntImage — 90° 轉正 + warp 取樣

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/frame/IntImage.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/frame/IntImageTest.kt`

**Interfaces:**
- Consumes: `Mat3`, `sourceFromModelFrame`
- Produces: `class IntImage(val width: Int, val height: Int, val pixels: IntArray)`(ARGB int)、
  `fun IntImage.rotate90(degrees: Int): IntImage`(degrees ∈ {0,90,180,270},順時針,對應 CameraX rotationDegrees 語意:輸出為正立影像)、
  `fun IntImage.warpToModelFrame(sourceFromModel: Mat3, big: Boolean): IntImage`(輸出 512×256,最近鄰取樣,出界填黑)

- [ ] **Step 1: 寫失敗測試**

```kotlin
package com.example.trafficlight.core.frame

import com.example.trafficlight.core.geometry.Mat3
import com.example.trafficlight.core.geometry.ModelFrames
import org.junit.Assert.assertEquals
import org.junit.Test

class IntImageTest {
    // 2x1 影像:左紅右綠
    private val redGreen = IntImage(2, 1, intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt()))

    @Test fun rotate0IsIdentity() {
        val r = redGreen.rotate90(0)
        assertEquals(0xFFFF0000.toInt(), r.pixels[0])
    }

    @Test fun rotate90MakesPortrait() {
        val r = redGreen.rotate90(90)
        assertEquals(1, r.width)
        assertEquals(2, r.height)
        // 順時針 90°:原 (0,0)紅 → (width-1-0, 0) = (0,0);原(1,0)綠 → (0,1)
        assertEquals(0xFFFF0000.toInt(), r.pixels[0])
        assertEquals(0xFF00FF00.toInt(), r.pixels[1])
    }

    @Test fun rotate180Reverses() {
        val r = redGreen.rotate90(180)
        assertEquals(0xFF00FF00.toInt(), r.pixels[0])
        assertEquals(0xFFFF0000.toInt(), r.pixels[1])
    }

    @Test fun rotate360ViaTwo180sMatches() {
        val r = redGreen.rotate90(180).rotate90(180)
        assertEquals(redGreen.pixels.toList(), r.pixels.toList())
    }

    @Test fun warpIdentityMatrixSamplesDirectly() {
        // 用單位矩陣:模型像素 (x,y) 直接取來源 (x,y)
        val src = IntImage(ModelFrames.MODEL_WIDTH, ModelFrames.MODEL_HEIGHT,
            IntArray(ModelFrames.MODEL_WIDTH * ModelFrames.MODEL_HEIGHT) { it })
        val out = src.warpToModelFrame(Mat3.identity(), big = false)
        assertEquals(src.pixels[5000], out.pixels[5000])
    }

    @Test fun warpOutOfBoundsIsBlack() {
        // 平移超出來源範圍 → 黑
        val shift = Mat3(floatArrayOf(1f, 0f, 99999f, 0f, 1f, 0f, 0f, 0f, 1f))
        val src = IntImage(4, 4, IntArray(16) { 0xFFFFFFFF.toInt() })
        val out = src.warpToModelFrame(shift, big = false)
        assertEquals(0xFF000000.toInt(), out.pixels[0])
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*IntImageTest*'`
Expected: FAIL

- [ ] **Step 3: 實作 IntImage.kt**

```kotlin
package com.example.trafficlight.core.frame

import com.example.trafficlight.core.geometry.Mat3
import com.example.trafficlight.core.geometry.ModelFrames

class IntImage(val width: Int, val height: Int, val pixels: IntArray) {
    init { require(pixels.size == width * height) }
}

/** 順時針旋轉,degrees ∈ {0, 90, 180, 270}。用於把 CameraX 幀轉正。 */
fun IntImage.rotate90(degrees: Int): IntImage {
    return when (((degrees % 360) + 360) % 360) {
        0 -> this
        90 -> {
            val out = IntArray(pixels.size)
            for (y in 0 until height) for (x in 0 until width) {
                out[x * height + (height - 1 - y)] = pixels[y * width + x]
            }
            IntImage(height, width, out)
        }
        180 -> {
            val out = IntArray(pixels.size)
            for (i in pixels.indices) out[pixels.size - 1 - i] = pixels[i]
            IntImage(width, height, out)
        }
        270 -> {
            val out = IntArray(pixels.size)
            for (y in 0 until height) for (x in 0 until width) {
                out[(width - 1 - x) * height + y] = pixels[y * width + x]
            }
            IntImage(height, width, out)
        }
        else -> throw IllegalArgumentException("unsupported rotation: $degrees")
    }
}

/** inverse mapping:對每個模型幀像素,經 sourceFromModel 找來源像素,最近鄰取樣。 */
fun IntImage.warpToModelFrame(sourceFromModel: Mat3, big: Boolean): IntImage {
    val w = ModelFrames.MODEL_WIDTH
    val h = ModelFrames.MODEL_HEIGHT
    val out = IntArray(w * h) { 0xFF000000.toInt() }
    for (y in 0 until h) {
        for (x in 0 until w) {
            val (sx, sy) = sourceFromModel.map(x.toFloat(), y.toFloat())
            val ix = (sx + 0.5f).toInt()
            val iy = (sy + 0.5f).toInt()
            if (ix in 0 until width && iy in 0 until height) {
                out[y * w + x] = pixels[iy * width + ix]
            }
        }
    }
    return IntImage(w, h, out)
}
```

(`big` 參數目前僅供呼叫端語意,取樣流程相同——矩陣本身已含大小幀差異。)

- [ ] **Step 4: 跑測試確認通過**

Run: `./gradlew :core:test --tests '*IntImageTest*'`
Expected: PASS(6 tests)

- [ ] **Step 5: Commit**

```bash
git add core
git commit -m "feat(core): IntImage with rotate90 and model-frame warp"
```

---

### Task 4: YUV12 打包

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/frame/YuvPacker.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/frame/YuvPackerTest.kt`

**Interfaces:**
- Consumes: `IntImage`
- Produces: `fun packYuv12(img: IntImage): ByteArray`(輸入必須是 512×256,輸出 6·128·256 = 196608 bytes,佈局同 `InferenceEngine.packOpenpilotFrame`)

- [ ] **Step 1: 寫失敗測試**

```kotlin
package com.example.trafficlight.core.frame

import org.junit.Assert.assertEquals
import org.junit.Test

class YuvPackerTest {
    private val w = 512
    private val h = 256
    private val half = (w / 2) * (h / 2) // 32768

    @Test fun outputSizeIs6Planes() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFF000000.toInt() })
        assertEquals(6 * half, packYuv12(img).size)
    }

    @Test fun whiteImageYIs255UvIs128() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFFFFFFFF.toInt() })
        val p = packYuv12(img)
        assertEquals(255, p[0].toInt() and 0xFF)          // Y plane 0 (top-left)
        assertEquals(128, p[4 * half].toInt() and 0xFF)   // U
        assertEquals(128, p[5 * half].toInt() and 0xFF)   // V
    }

    @Test fun pureRedHasHighV() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFFFF0000.toInt() })
        val p = packYuv12(img)
        val v = p[5 * half].toInt() and 0xFF
        assertEquals(255, v) // 0.5*255+128 clamp → 255
        val u = p[4 * half].toInt() and 0xFF
        assertEquals(84, u.toDouble(), 2.0) // -0.169*255+128 ≈ 84.9
    }

    @Test fun subsamplePositionsAreQuadrants() {
        // 只有像素 (1,0)(=top-right of 2x2 block 0) 是白,其他黑
        val px = IntArray(w * h) { 0xFF000000.toInt() }
        px[1] = 0xFFFFFFFF.toInt()
        val p = packYuv12(IntImage(w, h, px))
        assertEquals(0, p[0].toInt() and 0xFF)             // Y TL
        assertEquals(0, p[half].toInt() and 0xFF)          // Y BL
        assertEquals(255, p[2 * half].toInt() and 0xFF)    // Y TR ← 白像素在這
        assertEquals(0, p[3 * half].toInt() and 0xFF)      // Y BR
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*YuvPackerTest*'`
Expected: FAIL

- [ ] **Step 3: 實作 YuvPacker.kt**

逐行移植 `InferenceEngine.kt:402-455` 的 RGB→YUV 與打包(略去 Bitmap,直接吃 IntImage):

```kotlin
package com.example.trafficlight.core.frame

fun packYuv12(img: IntImage): ByteArray {
    val w = img.width
    val h = img.height
    require(w == 512 && h == 256) { "expected 512x256, got ${w}x$h" }
    val halfW = w / 2
    val halfH = h / 2
    val yPlane = IntArray(w * h)
    val uPlane = IntArray(halfW * halfH)
    val vPlane = IntArray(halfW * halfH)

    for (blockY in 0 until h step 2) {
        for (blockX in 0 until w step 2) {
            var uSum = 0
            var vSum = 0
            for (dy in 0..1) for (dx in 0..1) {
                val x = blockX + dx
                val y = blockY + dy
                val pixel = img.pixels[y * w + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                yPlane[y * w + x] = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
                uSum += (-0.169f * r - 0.331f * g + 0.5f * b + 128f).toInt().coerceIn(0, 255)
                vSum += (0.5f * r - 0.419f * g - 0.081f * b + 128f).toInt().coerceIn(0, 255)
            }
            val uvIndex = (blockY / 2) * halfW + (blockX / 2)
            uPlane[uvIndex] = uSum / 4
            vPlane[uvIndex] = vSum / 4
        }
    }

    val packed = ByteArray(6 * halfW * halfH)
    for (y in 0 until halfH) {
        for (x in 0 until halfW) {
            val base = y * halfW + x
            packed[base] = yPlane[(y * 2) * w + x * 2].toByte()
            packed[halfW * halfH + base] = yPlane[(y * 2 + 1) * w + x * 2].toByte()
            packed[2 * halfW * halfH + base] = yPlane[(y * 2) * w + x * 2 + 1].toByte()
            packed[3 * halfW * halfH + base] = yPlane[(y * 2 + 1) * w + x * 2 + 1].toByte()
            packed[4 * halfW * halfH + base] = uPlane[base].toByte()
            packed[5 * halfW * halfH + base] = vPlane[base].toByte()
        }
    }
    return packed
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `./gradlew :core:test --tests '*YuvPackerTest*'`
Expected: PASS(4 tests)

- [ ] **Step 5: Commit**

```bash
git add core
git commit -m "feat(core): YUV12 frame packing (openpilot frames_to_tensor layout)"
```

---

### Task 5: Plan 解析

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/plan/PlanParser.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/plan/PlanParserTest.kt`

**Interfaces:**
- Produces:
  ```kotlin
  enum class DrivingAction { STOP, GO, HOLD }
  data class PlanPoint(val x: Float, val y: Float)
  data class DrivingPlan(
      val shouldStop: Boolean, val shouldGo: Boolean, val confidence: Float,
      val nearVelocity: Float, val futureVelocity: Float,
      val desiredAcceleration: Float, val action: DrivingAction,
      val path: List<PlanPoint>
  )
  fun parseDrivingPlan(policyData: FloatArray): DrivingPlan
  ```
  行為與 `InferenceEngine.parseDrivingPlan`/`getOpenpilotDesiredAcceleration`/`interpolatePlanVelocity`/`modelTimeIndex` 完全一致(常數同 Global Constraints + STOPPING_VELOCITY_MPS=0.3, GO_ACCELERATION_MPS2=0.45, GO_VELOCITY_DELTA_MPS=0.35, MIN_STABLE_DELAY_S=0.3, MODEL_ACTION_T_S=0.075)。

- [ ] **Step 1: 寫失敗測試**

合成 policy 向量:33 個時間點 × 15 欄,索引 3 = 速度、索引 6 = 加速度、索引 0/1 = x/y。

```kotlin
package com.example.trafficlight.core.plan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PlanParserTest {
    private fun policyVector(vNow: Float, vEverywhereElse: Float, aNow: Float = 0f): FloatArray {
        val data = FloatArray(33 * 15)
        for (i in 0 until 33) {
            data[i * 15 + 3] = if (i == 0) vNow else vEverywhereElse
            data[i * 15 + 6] = aNow
            data[i * 15] = i.toFloat()      // path x
            data[i * 15 + 1] = 0f           // path y
        }
        return data
    }

    @Test fun stoppedPlanYieldsStop() {
        val plan = parseDrivingPlan(policyVector(vNow = 0f, vEverywhereElse = 0f))
        assertTrue(plan.shouldStop)
        assertEquals(DrivingAction.STOP, plan.action)
    }

    @Test fun acceleratingFromStandstillYieldsGo() {
        val plan = parseDrivingPlan(policyVector(vNow = 0.5f, vEverywhereElse = 5f))
        assertTrue(plan.shouldGo)
        assertFalse(plan.shouldStop)
        assertEquals(DrivingAction.GO, plan.action)
    }

    @Test fun cruisingYieldsHold() {
        val plan = parseDrivingPlan(policyVector(vNow = 15f, vEverywhereElse = 15f))
        assertEquals(DrivingAction.HOLD, plan.action)
    }

    @Test fun tooShortVectorIsHold() {
        assertEquals(DrivingAction.HOLD, parseDrivingPlan(FloatArray(10)).action)
    }

    @Test fun pathFiltersNegativeAndNonFinite() {
        val data = policyVector(15f, 15f)
        data[0] = -1f                      // x < 0 → 濾掉
        data[15] = Float.NaN               // 非有限 → 濾掉
        val plan = parseDrivingPlan(data)
        assertEquals(31, plan.path.size)
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*PlanParserTest*'`
Expected: FAIL

- [ ] **Step 3: 實作 PlanParser.kt**

移植 `InferenceEngine.kt:218-257, 335-360`(不含 calibration 欄位——calibration 由 Task 6 元件持有):

```kotlin
package com.example.trafficlight.core.plan

enum class DrivingAction { STOP, GO, HOLD }

data class PlanPoint(val x: Float, val y: Float)

data class DrivingPlan(
    val shouldStop: Boolean,
    val shouldGo: Boolean,
    val confidence: Float,
    val nearVelocity: Float,
    val futureVelocity: Float,
    val desiredAcceleration: Float,
    val action: DrivingAction,
    val path: List<PlanPoint> = emptyList()
)

private const val PLAN_VALUES = 33 * 15
private const val PLAN_WIDTH = 15
private const val PLAN_ACCELERATION_X = 6
private const val PLAN_VELOCITY_X = 3
private const val MIN_STABLE_DELAY_S = 0.3f
private const val MODEL_ACTION_T_S = 0.075f
private const val STOPPING_VELOCITY_MPS = 0.3f
private const val GO_ACCELERATION_MPS2 = 0.45f
private const val GO_VELOCITY_DELTA_MPS = 0.35f

val EMPTY_PLAN = DrivingPlan(false, false, 0f, 0f, 0f, 0f, DrivingAction.HOLD)

fun parseDrivingPlan(policyData: FloatArray): DrivingPlan {
    if (policyData.size < PLAN_VALUES) return EMPTY_PLAN

    val nearVelocity = policyData[PLAN_VELOCITY_X]
    val futureVelocity = policyData[16 * PLAN_WIDTH + PLAN_VELOCITY_X]
    val path = (0 until 33).map { i ->
        PlanPoint(policyData[i * PLAN_WIDTH], policyData[i * PLAN_WIDTH + 1])
    }.filter { it.x.isFinite() && it.y.isFinite() && it.x >= 0f }
    val accelerationNow = policyData[PLAN_ACCELERATION_X]
    val desiredAcceleration = desiredAcceleration(policyData, nearVelocity, accelerationNow)
    val shouldStop = nearVelocity < STOPPING_VELOCITY_MPS && desiredAcceleration < 0.1f
    val velocityDelta = futureVelocity - nearVelocity
    val shouldGo = !shouldStop && nearVelocity < 1.5f &&
        desiredAcceleration > GO_ACCELERATION_MPS2 && velocityDelta > GO_VELOCITY_DELTA_MPS
    val action = when {
        shouldStop -> DrivingAction.STOP
        shouldGo -> DrivingAction.GO
        else -> DrivingAction.HOLD
    }
    val confidence = when (action) {
        DrivingAction.STOP -> ((0.1f - desiredAcceleration) / 1.6f).coerceIn(0.65f, 1f)
        DrivingAction.GO -> ((desiredAcceleration - GO_ACCELERATION_MPS2) / 1.8f).coerceIn(0.65f, 1f)
        DrivingAction.HOLD -> 0.2f
    }
    return DrivingPlan(shouldStop, shouldGo, confidence, nearVelocity, futureVelocity,
        desiredAcceleration, action, path)
}

private fun desiredAcceleration(policyData: FloatArray, vNow: Float, aNow: Float): Float {
    val stableTargetVelocity = interpolatePlanVelocity(policyData, MIN_STABLE_DELAY_S)
    val vTarget = vNow + (MODEL_ACTION_T_S / MIN_STABLE_DELAY_S) * (stableTargetVelocity - vNow)
    return 2f * (vTarget - vNow) / MODEL_ACTION_T_S - aNow
}

private fun interpolatePlanVelocity(policyData: FloatArray, targetTimeS: Float): Float {
    var previousTime = 0f
    var previousVelocity = policyData[PLAN_VELOCITY_X]
    for (i in 1 until 33) {
        val time = 10f * (i / 32f) * (i / 32f)
        val velocity = policyData[i * PLAN_WIDTH + PLAN_VELOCITY_X]
        if (targetTimeS <= time) {
            val ratio = ((targetTimeS - previousTime) / (time - previousTime)).coerceIn(0f, 1f)
            return previousVelocity + (velocity - previousVelocity) * ratio
        }
        previousTime = time
        previousVelocity = velocity
    }
    return previousVelocity
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `./gradlew :core:test --tests '*PlanParserTest*'`
Expected: PASS(5 tests)

- [ ] **Step 5: Commit**

```bash
git add core
git commit -m "feat(core): extract driving plan parser"
```

---

### Task 6: CalibrationFusion — 模型自校正 + IMU 融合

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/calib/CalibrationFusion.kt`
- Create: `core/src/main/kotlin/com/example/trafficlight/core/calib/ImuTilt.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/calib/ImuTiltTest.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/calib/CalibrationFusionTest.kt`

**Interfaces:**
- Produces:
  ```kotlin
  // ImuTilt.kt — 由重力向量計算相機傾斜(裝置座標:x 右、y 上、z 出螢幕)
  data class Tilt(val rollDeg: Float, val pitchDeg: Float)
  fun tiltFromGravity(gx: Float, gy: Float, gz: Float, rotationDegrees: Int): Tilt

  // CalibrationFusion.kt
  data class CalibrationState(
      val rollDeg: Float, val pitchDeg: Float, val yawDeg: Float,
      val heightM: Float, val valid: Boolean, val sampleCount: Int
  )
  class CalibrationFusion(initialPitchDeg: Float = 5.5f) {
      fun onImuTilt(tilt: Tilt, timestampMs: Long): Boolean  // 回傳 true = 突變,需重置時序緩衝
      fun onModelOutputs(visionData: FloatArray): Boolean    // 回傳 true = warp 剛啟用,需重置
      fun state(): CalibrationState
      fun reset()
  }
  ```
- `tiltFromGravity` 語意:先把重力向量旋入「轉正後」的幀座標,rollDeg = 幀相對地平線的殘餘滾轉(逆時針為正),pitchDeg = 相機光軸相對水平面仰角(朝上為正)。

- [ ] **Step 1: 寫失敗測試 ImuTiltTest.kt**

```kotlin
package com.example.trafficlight.core.calib

import org.junit.Assert.assertEquals
import org.junit.Test

class ImuTiltTest {
    // 手機直立(直向)、相機水平朝前:重力沿裝置 +y
    @Test fun uprightPortraitIsZero() {
        val t = tiltFromGravity(0f, 9.81f, 0f, rotationDegrees = 90)
        assertEquals(0f, t.rollDeg, 0.5f)
        assertEquals(0f, t.pitchDeg, 0.5f)
    }

    // 橫向(頂朝左,rotationDegrees=0 慣例):重力沿裝置 +x
    @Test fun landscapeIsZero() {
        val t = tiltFromGravity(9.81f, 0f, 0f, rotationDegrees = 0)
        assertEquals(0f, t.rollDeg, 0.5f)
        assertEquals(0f, t.pitchDeg, 0.5f)
    }

    // 直向、往順時針歪 10°:gx = sin10°g, gy = cos10°g
    @Test fun tenDegreeRollDetected() {
        val g = 9.81f
        val gx = (g * kotlin.math.sin(Math.toRadians(10.0))).toFloat()
        val gy = (g * kotlin.math.cos(Math.toRadians(10.0))).toFloat()
        val t = tiltFromGravity(gx, gy, 0f, rotationDegrees = 90)
        assertEquals(10f, kotlin.math.abs(t.rollDeg), 1.0f)
    }

    // 直向、上仰 15°(頂往後倒):gz = sin15°g
    @Test fun pitchUpDetected() {
        val g = 9.81f
        val gy = (g * kotlin.math.cos(Math.toRadians(15.0))).toFloat()
        val gz = (g * kotlin.math.sin(Math.toRadians(15.0))).toFloat()
        val t = tiltFromGravity(0f, gy, gz, rotationDegrees = 90)
        assertEquals(15f, t.pitchDeg, 1.0f)
    }
}
```

- [ ] **Step 2: 寫失敗測試 CalibrationFusionTest.kt**

```kotlin
package com.example.trafficlight.core.calib

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationFusionTest {
    @Test fun imuRollIsLowPassed() {
        val f = CalibrationFusion()
        f.onImuTilt(Tilt(rollDeg = 10f, pitchDeg = 5f), 0L)
        val afterOne = f.state().rollDeg
        assertTrue(afterOne > 0f && afterOne < 10f)
        // 持續餵 3 秒(50Hz)→ 收斂到 10 附近
        var t = 0L
        repeat(150) { t += 20; f.onImuTilt(Tilt(10f, 5f), t) }
        assertEquals(10f, f.state().rollDeg, 1.0f)
    }

    @Test fun initialPitchComesFromImuBeforeModelCalib() {
        val f = CalibrationFusion(initialPitchDeg = 5.5f)
        var t = 0L
        repeat(150) { t += 20; f.onImuTilt(Tilt(0f, 8f), t) }
        assertEquals(8f, f.state().pitchDeg, 1.5f)
    }

    @Test fun suddenRollJumpSignalsReset() {
        val f = CalibrationFusion()
        var t = 0L
        repeat(150) { t += 20; f.onImuTilt(Tilt(0f, 0f), t) }
        // 突然 +20°(手機被重新擺放)
        val reset = f.onImuTilt(Tilt(20f, 0f), t + 20)
        assertTrue(reset)
        assertEquals(0, f.state().sampleCount)
    }

    @Test fun smallVibrationDoesNotReset() {
        val f = CalibrationFusion()
        var t = 0L
        var anyReset = false
        repeat(200) {
            t += 20
            val jitter = if (it % 2 == 0) 1.5f else -1.5f
            if (f.onImuTilt(Tilt(jitter, jitter), t)) anyReset = true
        }
        assertFalse(anyReset)
    }

    @Test fun modelOutputsRefinePitchAndActivateWarp() {
        val f = CalibrationFusion()
        // 合成 vision 輸出:pose x 平移可信、pitch/yaw std 小、height 合理
        val vision = FloatArray(1600)
        vision[87] = 5f          // poseTransX
        vision[87 + 6] = -3f     // ln(std) → std=e^-3 小
        vision[99] = 0f          // wideRoll
        vision[99 + 1] = Math.toRadians(3.0).toFloat()  // widePitch 3°
        vision[99 + 2] = 0f      // wideYaw
        vision[99 + 4] = -3f     // pitchStd
        vision[99 + 5] = -3f     // yawStd
        vision[105 + 2] = 1.4f   // roadHeight
        vision[105 + 8] = -3f    // heightStd
        var activated = false
        repeat(25) { if (f.onModelOutputs(vision)) activated = true }
        assertTrue(activated)          // 累積 20 樣本後 warp 啟用一次
        assertTrue(f.state().valid)
    }
}
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*calib*'`
Expected: FAIL

- [ ] **Step 4: 實作 ImuTilt.kt**

```kotlin
package com.example.trafficlight.core.calib

import kotlin.math.atan2
import kotlin.math.sqrt

data class Tilt(val rollDeg: Float, val pitchDeg: Float)

/**
 * 由重力向量(裝置座標:x 右、y 上、z 出螢幕)計算轉正後幀的殘餘滾轉與相機仰角。
 * rotationDegrees:CameraX rotationDegrees(幀轉正所需的順時針角度)。
 */
fun tiltFromGravity(gx: Float, gy: Float, gz: Float, rotationDegrees: Int): Tilt {
    // 裝置在螢幕平面內相對「直立」的角度(順時針為正)
    val screenAngleDeg = Math.toDegrees(atan2(gx.toDouble(), gy.toDouble())).toFloat()
    // 轉正後殘餘 roll:扣掉 90° 步進(rotationDegrees=90 表示直向)
    val stepDeg = when (((rotationDegrees % 360) + 360) % 360) {
        90 -> 0f     // 直向:直立即 0
        0 -> -90f    // 橫向(頂朝左)
        180 -> 90f   // 反向橫向
        270 -> 180f
        else -> 0f
    }
    var roll = screenAngleDeg + stepDeg
    while (roll > 180f) roll -= 360f
    while (roll < -180f) roll += 360f

    val gPlane = sqrt((gx * gx + gy * gy).toDouble()).toFloat()
    val pitch = Math.toDegrees(atan2(gz.toDouble(), gPlane.toDouble())).toFloat()
    return Tilt(roll, pitch)
}
```

(注意:`stepDeg` 對應關係以測試為準——`uprightPortraitIsZero` 與 `landscapeIsZero` 兩個測試錨定行為;若實機方向相反,只改此對照表,測試同步更新。)

- [ ] **Step 5: 實作 CalibrationFusion.kt**

模型自校正部分逐行移植 `InferenceEngine.kt:278-333`(`updateAutoCalibration`/`safeExp`/`lowPass`),IMU 部分新增:

```kotlin
package com.example.trafficlight.core.calib

import kotlin.math.abs
import kotlin.math.exp

data class CalibrationState(
    val rollDeg: Float,
    val pitchDeg: Float,
    val yawDeg: Float,
    val heightM: Float,
    val valid: Boolean,
    val sampleCount: Int
)

class CalibrationFusion(private val initialPitchDeg: Float = 5.5f) {

    private var imuRollDeg = 0f
    private var imuPitchDeg = initialPitchDeg
    private var imuInitialized = false
    private var lastImuTimestampMs = 0L

    private var modelPitchDeg = initialPitchDeg
    private var modelYawDeg = 0f
    private var heightM = 1.35f
    private var sampleCount = 0
    private var warpWasActive = false
    var valid = false
        private set

    companion object {
        private const val MIN_SAMPLES = 20
        private const val IMU_TAU_S = 0.5f            // 低通時間常數
        private const val SUDDEN_CHANGE_DEG = 6f       // 濾波值 vs 觀測值差異閾值
        private const val VISION_POSE_START = 87
        private const val VISION_WIDE_EULER_START = 99
        private const val VISION_ROAD_TRANSFORM_START = 105
    }

    /** 回傳 true = 偵測到手機被重新擺放,呼叫端須重置時序緩衝。 */
    fun onImuTilt(tilt: Tilt, timestampMs: Long): Boolean {
        if (!imuInitialized) {
            imuRollDeg = tilt.rollDeg
            imuPitchDeg = tilt.pitchDeg
            imuInitialized = true
            lastImuTimestampMs = timestampMs
            return false
        }
        val sudden = abs(tilt.rollDeg - imuRollDeg) > SUDDEN_CHANGE_DEG ||
            abs(tilt.pitchDeg - imuPitchDeg) > SUDDEN_CHANGE_DEG
        val dtS = ((timestampMs - lastImuTimestampMs).coerceAtLeast(1L)) / 1000f
        lastImuTimestampMs = timestampMs
        val alpha = (dtS / (IMU_TAU_S + dtS)).coerceIn(0f, 1f)
        imuRollDeg += (tilt.rollDeg - imuRollDeg) * alpha
        imuPitchDeg += (tilt.pitchDeg - imuPitchDeg) * alpha
        if (sudden && abs(tilt.rollDeg - imuRollDeg) > SUDDEN_CHANGE_DEG / 2) {
            // 真突變(非單次雜訊):快速跟上並要求重置
            imuRollDeg = tilt.rollDeg
            imuPitchDeg = tilt.pitchDeg
            reset()
            return true
        }
        return false
    }

    /** 移植自 InferenceEngine.updateAutoCalibration;回傳 true = warp 剛轉為啟用。 */
    fun onModelOutputs(visionData: FloatArray): Boolean {
        if (visionData.size <= VISION_ROAD_TRANSFORM_START + 8) return false
        val wasValid = valid

        val poseTransX = visionData[VISION_POSE_START]
        val poseStdX = safeExp(visionData[VISION_POSE_START + 6])
        val roadHeight = abs(visionData[VISION_ROAD_TRANSFORM_START + 2])
        val roadHeightStd = safeExp(visionData[VISION_ROAD_TRANSFORM_START + 8])
        val widePitch = visionData[VISION_WIDE_EULER_START + 1]
        val wideYaw = visionData[VISION_WIDE_EULER_START + 2]
        val widePitchStd = safeExp(visionData[VISION_WIDE_EULER_START + 4])
        val wideYawStd = safeExp(visionData[VISION_WIDE_EULER_START + 5])

        val poseReliable = poseTransX.isFinite() && abs(poseTransX) > 0.05f && poseStdX < 2.5f
        val eulerReliable = poseReliable &&
            widePitch.isFinite() && wideYaw.isFinite() &&
            widePitchStd.isFinite() && wideYawStd.isFinite() &&
            widePitchStd < 0.20f && wideYawStd < 0.20f

        var updated = false
        if (eulerReliable) {
            val obsPitch = Math.toDegrees(widePitch.toDouble()).toFloat()
            val obsYaw = Math.toDegrees(wideYaw.toDouble()).toFloat()
            if (obsPitch in -12f..12f) { modelPitchDeg = lowPass(modelPitchDeg, obsPitch, 0.015f); updated = true }
            if (obsYaw in -12f..12f) { modelYawDeg = lowPass(modelYawDeg, obsYaw, 0.015f); updated = true }
        }
        if (roadHeight.isFinite() && roadHeightStd.isFinite() && roadHeightStd < 0.60f && roadHeight in 0.7f..2.2f) {
            heightM = lowPass(heightM, roadHeight, 0.01f)
            updated = true
        }
        if (updated) sampleCount += 1
        valid = sampleCount >= MIN_SAMPLES
        val warpActivated = valid && !wasValid && !warpWasActive
        if (valid) warpWasActive = true
        return warpActivated
    }

    fun state(): CalibrationState {
        // pitch:模型校正 valid 前採 IMU,valid 後採模型細化值
        val pitch = if (valid) modelPitchDeg else imuPitchDeg
        return CalibrationState(imuRollDeg, pitch, modelYawDeg, heightM, valid, sampleCount)
    }

    fun reset() {
        sampleCount = 0
        valid = false
        warpWasActive = false
        modelPitchDeg = if (imuInitialized) imuPitchDeg else initialPitchDeg
        modelYawDeg = 0f
    }

    private fun safeExp(v: Float): Float =
        if (v.isFinite()) exp(v.coerceAtMost(11f)) else Float.POSITIVE_INFINITY

    private fun lowPass(prev: Float, obs: Float, alpha: Float): Float = prev + (obs - prev) * alpha
}
```

- [ ] **Step 6: 跑測試確認通過**

Run: `./gradlew :core:test --tests '*calib*'`
Expected: PASS(9 tests)

- [ ] **Step 7: Commit**

```bash
git add core
git commit -m "feat(core): IMU tilt extraction and calibration fusion with sudden-change reset"
```

---

### Task 7: DrivingPipeline — 端到端組裝

**Files:**
- Create: `core/src/main/kotlin/com/example/trafficlight/core/pipeline/DrivingPipeline.kt`
- Test: `core/src/test/kotlin/com/example/trafficlight/core/pipeline/DrivingPipelineTest.kt`

**Interfaces:**
- Consumes: 全部前置 task 的 API
- Produces:
  ```kotlin
  interface ModelRunner {
      /** stackedImg / stackedBigImg:2 幀 × 6ch × 128×256 = 393216 bytes */
      fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray
      fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray
  }
  data class PipelineResult(val plan: DrivingPlan, val calibration: CalibrationState)
  class DrivingPipeline(private val runner: ModelRunner, val calibration: CalibrationFusion = CalibrationFusion()) {
      fun processFrame(frame: IntImage, rotationDegrees: Int, horizontalFovDeg: Float): PipelineResult
      fun onImuTilt(tilt: Tilt, timestampMs: Long): Boolean  // 委派給 calibration,突變時重置
      fun resetTemporalBuffers()
      val bufferedFrameReady: Boolean  // 測試用:previousFrame 是否已存在
  }
  ```
  常數:`POLICY_FRAMES=25`, `FEATURE_LEN=512`, `VISION_HIDDEN_STATE_START=1064`, desire 全零 `25×8`, trafficConvention = `[1f, 0f]`(靠右駕駛)。

- [ ] **Step 1: 寫失敗測試**

```kotlin
package com.example.trafficlight.core.pipeline

import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.plan.DrivingAction
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeRunner : ModelRunner {
    var visionCalls = 0
    var lastStackedImg: ByteArray? = null
    var lastFeatures: FloatArray? = null
    override fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray {
        visionCalls++
        lastStackedImg = stackedImg
        return FloatArray(1600) { if (it >= 1064) 0.5f else 0f } // hidden state 0.5
    }
    override fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray {
        lastFeatures = featuresBuffer
        return FloatArray(33 * 15) // 全零 → STOP(v=0, a=0)
    }
}

class DrivingPipelineTest {
    private fun frame(w: Int = 640, h: Int = 480) = IntImage(w, h, IntArray(w * h) { 0xFF808080.toInt() })

    @Test fun processFrameReturnsParsedPlan() {
        val p = DrivingPipeline(FakeRunner())
        val r = p.processFrame(frame(), rotationDegrees = 0, horizontalFovDeg = 72f)
        assertEquals(DrivingAction.STOP, r.plan.action)
    }

    @Test fun stackedInputIsTwoFrames() {
        val runner = FakeRunner()
        val p = DrivingPipeline(runner)
        p.processFrame(frame(), 0, 72f)
        assertEquals(2 * 6 * 128 * 256, runner.lastStackedImg!!.size)
    }

    @Test fun featureBufferShiftsIn() {
        val runner = FakeRunner()
        val p = DrivingPipeline(runner)
        p.processFrame(frame(), 0, 72f)
        val f = runner.lastFeatures!!
        assertEquals(25 * 512, f.size)
        assertEquals(0f, f[0], 1e-6f)                    // 最舊幀仍為 0
        assertEquals(0.5f, f[24 * 512], 1e-6f)           // 最新幀 = hidden state
    }

    @Test fun rotationChangeResetsTemporalBuffers() {
        val p = DrivingPipeline(FakeRunner())
        p.processFrame(frame(), 0, 72f)
        assertTrue(p.bufferedFrameReady)
        p.processFrame(frame(480, 640), 90, 72f)          // 直向
        // 方向切換當下 buffer 被清,該幀重新起算 → 處理後又是 ready
        assertTrue(p.bufferedFrameReady)
        val runner2 = FakeRunner()
        val p2 = DrivingPipeline(runner2)
        p2.processFrame(frame(), 0, 72f)
        p2.processFrame(frame(480, 640), 90, 72f)
        // 切換後第一幀:stacked 的前半 == 後半(prior=current)
        val s = runner2.lastStackedImg!!
        val half = s.size / 2
        for (i in 0 until half step 5000) assertEquals(s[i], s[half + i])
    }

    @Test fun imuSuddenChangeResetsBuffers() {
        val p = DrivingPipeline(FakeRunner())
        p.processFrame(frame(), 0, 72f)
        var t = 0L
        repeat(150) { t += 20; p.onImuTilt(Tilt(0f, 0f), t) }
        assertTrue(p.onImuTilt(Tilt(25f, 0f), t + 20))
        assertFalse(p.bufferedFrameReady)
    }

    @Test fun portraitFrameIsRotatedUprightBeforeWarp() {
        // 直向幀(w<h)+ rotation 90 → 不 crash 且輸出 plan
        val p = DrivingPipeline(FakeRunner())
        val r = p.processFrame(frame(480, 640), 90, 60f)
        assertEquals(DrivingAction.STOP, r.plan.action)
    }
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `./gradlew :core:test --tests '*DrivingPipelineTest*'`
Expected: FAIL

- [ ] **Step 3: 實作 DrivingPipeline.kt**

```kotlin
package com.example.trafficlight.core.pipeline

import com.example.trafficlight.core.calib.CalibrationFusion
import com.example.trafficlight.core.calib.CalibrationState
import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.frame.packYuv12
import com.example.trafficlight.core.frame.rotate90
import com.example.trafficlight.core.frame.warpToModelFrame
import com.example.trafficlight.core.geometry.sourceFromModelFrame
import com.example.trafficlight.core.plan.DrivingPlan
import com.example.trafficlight.core.plan.parseDrivingPlan

interface ModelRunner {
    fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray
    fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray
}

data class PipelineResult(val plan: DrivingPlan, val calibration: CalibrationState)

class DrivingPipeline(
    private val runner: ModelRunner,
    val calibration: CalibrationFusion = CalibrationFusion()
) {
    companion object {
        const val POLICY_FRAMES = 25
        const val FEATURE_LEN = 512
        const val VISION_HIDDEN_STATE_START = 1064
        private const val PACKED_FRAME_SIZE = 6 * 128 * 256
    }

    private var previousMed: ByteArray? = null
    private var previousBig: ByteArray? = null
    private val featureBuffer = Array(POLICY_FRAMES) { FloatArray(FEATURE_LEN) }
    private var lastRotationDegrees: Int? = null

    val bufferedFrameReady: Boolean get() = previousMed != null

    fun onImuTilt(tilt: Tilt, timestampMs: Long): Boolean {
        val sudden = calibration.onImuTilt(tilt, timestampMs)
        if (sudden) resetTemporalBuffers()
        return sudden
    }

    fun resetTemporalBuffers() {
        previousMed = null
        previousBig = null
        for (f in featureBuffer) f.fill(0f)
    }

    fun processFrame(frame: IntImage, rotationDegrees: Int, horizontalFovDeg: Float): PipelineResult {
        if (lastRotationDegrees != null && rotationDegrees != lastRotationDegrees) {
            resetTemporalBuffers()
            calibration.reset()
        }
        lastRotationDegrees = rotationDegrees

        val upright = frame.rotate90(rotationDegrees)
        val cal = calibration.state()
        val rollRad = Math.toRadians(cal.rollDeg.toDouble()).toFloat()
        val pitchRad = Math.toRadians(cal.pitchDeg.toDouble()).toFloat()
        val yawRad = Math.toRadians(cal.yawDeg.toDouble()).toFloat()

        val medMatrix = sourceFromModelFrame(upright.width.toFloat(), upright.height.toFloat(),
            horizontalFovDeg, rollRad, pitchRad, yawRad, bigModelFrame = false)
        val bigMatrix = sourceFromModelFrame(upright.width.toFloat(), upright.height.toFloat(),
            horizontalFovDeg, rollRad, pitchRad, yawRad, bigModelFrame = true)

        val med = packYuv12(upright.warpToModelFrame(medMatrix, big = false))
        val big = packYuv12(upright.warpToModelFrame(bigMatrix, big = true))

        val stackedMed = ByteArray(PACKED_FRAME_SIZE * 2)
        (previousMed ?: med).copyInto(stackedMed, 0)
        med.copyInto(stackedMed, PACKED_FRAME_SIZE)
        val stackedBig = ByteArray(PACKED_FRAME_SIZE * 2)
        (previousBig ?: big).copyInto(stackedBig, 0)
        big.copyInto(stackedBig, PACKED_FRAME_SIZE)
        previousMed = med
        previousBig = big

        val visionData = runner.runVision(stackedMed, stackedBig)
        if (calibration.onModelOutputs(visionData)) {
            // warp 剛啟用:輸入幾何改變,清時序緩衝(保留本幀作為新起點)
            for (f in featureBuffer) f.fill(0f)
            previousMed = med
            previousBig = big
        }

        if (visionData.size >= VISION_HIDDEN_STATE_START + FEATURE_LEN) {
            for (i in 0 until POLICY_FRAMES - 1) {
                System.arraycopy(featureBuffer[i + 1], 0, featureBuffer[i], 0, FEATURE_LEN)
            }
            System.arraycopy(visionData, VISION_HIDDEN_STATE_START, featureBuffer[POLICY_FRAMES - 1], 0, FEATURE_LEN)
        }

        val features = FloatArray(POLICY_FRAMES * FEATURE_LEN)
        for (i in 0 until POLICY_FRAMES) {
            featureBuffer[i].copyInto(features, i * FEATURE_LEN)
        }
        val desire = FloatArray(POLICY_FRAMES * 8)
        val trafficConvention = floatArrayOf(1f, 0f)

        val policyData = runner.runPolicy(features, desire, trafficConvention)
        return PipelineResult(parseDrivingPlan(policyData), calibration.state())
    }
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `./gradlew :core:test`
Expected: PASS(全部 core 測試)

- [ ] **Step 5: Commit**

```bash
git add core
git commit -m "feat(core): end-to-end driving pipeline with orientation and tilt reset"
```

---

### Task 8: App 整合 — InferenceEngine 換 core、IMU、動態 FOV

**Files:**
- Modify: `app/build.gradle`(dependencies 加 `implementation project(':core')`)
- Rewrite: `app/src/main/java/com/example/trafficlight/inference/InferenceEngine.kt`
- Create: `app/src/main/java/com/example/trafficlight/sensor/ImuManager.kt`
- Create: `app/src/main/java/com/example/trafficlight/camera/CameraFov.kt`
- Modify: `app/src/main/java/com/example/trafficlight/analyzer/FrameAnalyzer.kt`
- Modify: `app/src/main/java/com/example/trafficlight/MainActivity.kt`

**Interfaces:**
- Consumes: core 的 `DrivingPipeline`, `ModelRunner`, `IntImage`, `Tilt`, `tiltFromGravity`, `PipelineResult`
- Produces: 新版 `InferenceEngine`:
  ```kotlin
  class InferenceEngine(context: Context) {
      suspend fun initialize(): Boolean
      suspend fun analyzeDrivingPlan(bitmap: Bitmap, rotationDegrees: Int, horizontalFovDeg: Float): DrivingPlanResult
      fun onImuTilt(tilt: Tilt, timestampMs: Long)
      fun release()
  }
  ```
  `DrivingPlanResult`/`PlanPoint`/`DrivingAction`/`CameraCalibrationEstimate` 保留為既有名稱的轉接(內部由 core 型別映射),`FrameAnalyzer`/`OverlayView` 介面不變。

- [ ] **Step 1: 重寫 InferenceEngine.kt**

保留對外資料類(供 OverlayView/FrameAnalyzer 使用),內部全部委派 core:

```kotlin
package com.example.trafficlight.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.pipeline.DrivingPipeline
import com.example.trafficlight.core.pipeline.ModelRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

// --- 對外型別維持不變(OverlayView / FrameAnalyzer 依賴) ---
data class DetectionResult(
    val bbox: RectF,
    val confidence: Float,
    val classId: Int,
    val label: String,
    val colorHint: Int? = null
)

data class ClassificationResult(
    val classId: Int,
    val confidence: Float,
    val probabilities: FloatArray
) {
    companion object {
        const val RED = 0
        const val YELLOW = 1
        const val GREEN = 2
        const val OFF = 3
        const val UNKNOWN = 4
    }
}

enum class DrivingAction { STOP, GO, HOLD }

data class PlanPoint(val x: Float, val y: Float)

data class CameraCalibrationEstimate(
    val pitchDeg: Float = 5.5f,
    val yawDeg: Float = 0f,
    val heightM: Float = 1.35f,
    val valid: Boolean = false,
    val sampleCount: Int = 0,
    val rollDeg: Float = 0f
)

data class DrivingPlanResult(
    val shouldStop: Boolean,
    val shouldGo: Boolean,
    val confidence: Float,
    val nearVelocity: Float,
    val futureVelocity: Float,
    val desiredAcceleration: Float,
    val action: DrivingAction,
    val path: List<PlanPoint> = emptyList(),
    val calibration: CameraCalibrationEstimate = CameraCalibrationEstimate()
)

class InferenceEngine(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var visionSession: OrtSession? = null
    private var policySession: OrtSession? = null
    private var pipeline: DrivingPipeline? = null

    companion object {
        private const val VISION_MODEL = "models/openpilot_driving_vision.onnx"
        private const val POLICY_MODEL = "models/openpilot_driving_policy.onnx"
    }

    private inner class OrtModelRunner : ModelRunner {
        override fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray {
            val env = ortEnvironment!!
            val vision = visionSession!!
            fun tensorOf(data: ByteArray): OnnxTensor {
                val buf = ByteBuffer.allocateDirect(data.size).order(ByteOrder.nativeOrder())
                buf.put(data); buf.rewind()
                return OnnxTensor.createTensor(env, buf, longArrayOf(1, 12, 128, 256), OnnxJavaType.UINT8)
            }
            val img = tensorOf(stackedImg)
            val bigImg = tensorOf(stackedBigImg)
            val out = vision.run(mapOf("img" to img, "big_img" to bigImg))
            val t = out.get(0) as OnnxTensor
            val data = readFloats(t)
            t.close(); out.close(); img.close(); bigImg.close()
            return data
        }

        override fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray {
            val env = ortEnvironment!!
            val policy = policySession!!
            fun tensorOf(data: FloatArray, shape: LongArray): OnnxTensor {
                val buf = FloatBuffer.allocate(data.size)
                buf.put(data); buf.rewind()
                return OnnxTensor.createTensor(env, buf, shape)
            }
            val desireT = tensorOf(desire, longArrayOf(1, 25, 8))
            val tcT = tensorOf(trafficConvention, longArrayOf(1, 2))
            val featT = tensorOf(featuresBuffer, longArrayOf(1, 25, 512))
            val out = policy.run(mapOf(
                "desire_pulse" to desireT,
                "traffic_convention" to tcT,
                "features_buffer" to featT
            ))
            val t = out.get(0) as OnnxTensor
            val data = readFloats(t)
            t.close(); out.close(); desireT.close(); tcT.close(); featT.close()
            return data
        }

        private fun readFloats(tensor: OnnxTensor): FloatArray {
            val buffer = tensor.floatBuffer
            buffer.rewind()
            val data = FloatArray(buffer.remaining())
            buffer.get(data)
            return data
        }
    }

    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val env = ortEnvironment ?: return@withContext false
            visionSession = env.createSession(context.assets.open(VISION_MODEL).readBytes())
            policySession = env.createSession(context.assets.open(POLICY_MODEL).readBytes())
            pipeline = DrivingPipeline(OrtModelRunner())
            Log.d("InferenceEngine", "openpilot models ready (core pipeline)")
            true
        } catch (e: Exception) {
            Log.e("InferenceEngine", "openpilot model init failed: ${e.message}", e)
            false
        }
    }

    fun onImuTilt(tilt: Tilt, timestampMs: Long) {
        pipeline?.onImuTilt(tilt, timestampMs)
    }

    suspend fun analyzeDrivingPlan(
        bitmap: Bitmap,
        rotationDegrees: Int,
        horizontalFovDeg: Float
    ): DrivingPlanResult = withContext(Dispatchers.Default) {
        val p = pipeline ?: return@withContext DrivingPlanResult(
            false, false, 0f, 0f, 0f, 0f, DrivingAction.HOLD)
        try {
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            val result = p.processFrame(IntImage(bitmap.width, bitmap.height, pixels), rotationDegrees, horizontalFovDeg)
            val plan = result.plan
            val cal = result.calibration
            DrivingPlanResult(
                shouldStop = plan.shouldStop,
                shouldGo = plan.shouldGo,
                confidence = plan.confidence,
                nearVelocity = plan.nearVelocity,
                futureVelocity = plan.futureVelocity,
                desiredAcceleration = plan.desiredAcceleration,
                action = when (plan.action) {
                    com.example.trafficlight.core.plan.DrivingAction.STOP -> DrivingAction.STOP
                    com.example.trafficlight.core.plan.DrivingAction.GO -> DrivingAction.GO
                    com.example.trafficlight.core.plan.DrivingAction.HOLD -> DrivingAction.HOLD
                },
                path = plan.path.map { PlanPoint(it.x, it.y) },
                calibration = CameraCalibrationEstimate(
                    cal.pitchDeg, cal.yawDeg, cal.heightM, cal.valid, cal.sampleCount, cal.rollDeg)
            )
        } catch (e: Exception) {
            Log.e("InferenceEngine", "inference failed: ${e.message}", e)
            DrivingPlanResult(false, false, 0f, 0f, 0f, 0f, DrivingAction.HOLD)
        }
    }

    fun release() {
        visionSession?.close()
        policySession?.close()
        ortEnvironment?.close()
    }
}
```

- [ ] **Step 2: 建立 ImuManager.kt**

```kotlin
package com.example.trafficlight.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.calib.tiltFromGravity

/** 訂閱重力感測器,轉成 core 的 Tilt 樣本。rotationDegrees 由外部(相機幀)提供。 */
class ImuManager(
    context: Context,
    private val onTilt: (Tilt, Long) -> Unit
) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gravitySensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    @Volatile var rotationDegrees: Int = 0

    val available: Boolean get() = gravitySensor != null

    fun start() {
        gravitySensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        val tilt = tiltFromGravity(event.values[0], event.values[1], event.values[2], rotationDegrees)
        onTilt(tilt, event.timestamp / 1_000_000L)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
```

- [ ] **Step 3: 建立 CameraFov.kt**

```kotlin
package com.example.trafficlight.camera

import android.hardware.camera2.CameraCharacteristics
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import kotlin.math.atan

/** 從 Camera2 characteristics 計算水平 FOV(度);讀不到時回傳 fallback。 */
@OptIn(ExperimentalCamera2Interop::class)
fun horizontalFovDeg(cameraInfo: CameraInfo, zoomRatio: Float, fallbackDeg: Float = 72f): Float {
    return try {
        val c2 = Camera2CameraInfo.from(cameraInfo)
        val focalLengths = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val sensorSize = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        if (focalLengths == null || focalLengths.isEmpty() || sensorSize == null) return fallbackDeg
        val fl = focalLengths[0]
        val baseFov = Math.toDegrees(2.0 * atan((sensorSize.width / (2f * fl)).toDouble())).toFloat()
        // 數位變焦縮小視角
        (baseFov / zoomRatio.coerceAtLeast(1f))
    } catch (e: Exception) {
        fallbackDeg
    }
}
```

`app/build.gradle` dependencies 需加(若未含):`implementation "androidx.camera:camera-camera2:1.4.0"`(已存在)與 `implementation project(':core')`。

- [ ] **Step 4: 修改 FrameAnalyzer.kt**

建構子加 `private val horizontalFovProvider: () -> Float`;
`runDetection` 改傳 rotation 與 FOV;`analyze()` 內取得 `imageRotation` 後傳入:

```kotlin
// runDetection 簽名與呼叫處改為:
private suspend fun runDetection(bitmap: Bitmap, rotationDegrees: Int, currentTime: Long) {
    allDetections = emptyList()
    val plan = inferenceEngine.analyzeDrivingPlan(bitmap, rotationDegrees, horizontalFovProvider())
    // ...其餘不變
}

// analyze() 內:
val imageRotation = image.imageInfo.rotationDegrees
if (shouldRunDetection) {
    runDetection(bitmap, imageRotation, currentTime)
}
```

注意:`createAnalysisResult` 傳給 OverlayView 的 `imageRotation` 改為 `0`、
`imageWidth/imageHeight` 改為轉正後尺寸(rotation 90/270 時交換寬高)——因為
core pipeline 已把幀轉正,overlay 的座標基準跟著改:

```kotlin
val uprightW = if (imageRotation == 90 || imageRotation == 270) bitmap.height else bitmap.width
val uprightH = if (imageRotation == 90 || imageRotation == 270) bitmap.width else bitmap.height
val result = createAnalysisResult(uprightW, uprightH, 0)
```

- [ ] **Step 5: 修改 MainActivity.kt**

1. `initComponents()` 中建立 ImuManager 並接到 engine:

```kotlin
private var imuManager: ImuManager? = null

// initComponents() 內:
imuManager = ImuManager(this) { tilt, ts -> inferenceEngine.onImuTilt(tilt, ts) }

// FrameAnalyzer 建構子補 FOV provider:
frameAnalyzer = FrameAnalyzer(
    inferenceEngine = inferenceEngine,
    stateMachine = stateMachine,
    roiSelector = roiSelector,
    onResultCallback = ::onAnalysisResult,
    onDebugCallback = ::updateDebugText,
    horizontalFovProvider = {
        cameraInfo?.let { horizontalFovDeg(it, currentZoomRatio) } ?: 72f
    }
)
```

2. `onResume`/`onPause` 啟停 IMU:

```kotlin
override fun onResume() {
    super.onResume()
    imuManager?.start()
}

override fun onPause() {
    super.onPause()
    imuManager?.stop()
}
```

3. `onAnalysisResult` 同步 rotation 給 ImuManager(取自 result.imageRotation 之前的來源——
   在 FrameAnalyzer 中直接呼叫即可):FrameAnalyzer 建構子再加
   `private val onRotationChanged: (Int) -> Unit = {}`,`analyze()` 內
   `onRotationChanged(imageRotation)`;MainActivity 傳
   `onRotationChanged = { imuManager?.rotationDegrees = it }`。

4. import 補 `com.example.trafficlight.sensor.ImuManager` 與
   `com.example.trafficlight.camera.horizontalFovDeg`。

- [ ] **Step 6: 編譯驗證**

Run: `./gradlew :app:assembleDebug :core:test`
Expected: BUILD SUCCESSFUL(APK 產出 + core 測試全綠)

- [ ] **Step 7: Commit**

```bash
git add app core
git commit -m "feat(app): wire core pipeline with IMU tilt correction and dynamic FOV"
```

---

### Task 9: 直/橫向 UI 與切換靜音

**Files:**
- Create: `app/src/main/res/layout-land/activity_main.xml`
- Modify: `app/src/main/res/layout/activity_main.xml`(僅若需要:確認 statusPanel 錨點在直向時位於頂部)
- Modify: `app/src/main/java/com/example/trafficlight/MainActivity.kt`
- Modify: `app/src/main/AndroidManifest.xml`

**Interfaces:**
- Consumes: 既有 view id(全部 id 必須在兩份 layout 中同名存在,MainActivity `findViewById` 才不會 NPE)

- [ ] **Step 1: Manifest 加 configChanges**

`<activity>` 加屬性,旋轉時不重建(相機與模型 session 保留):

```xml
android:configChanges="orientation|screenSize|screenLayout"
```

- [ ] **Step 2: 建立 layout-land/activity_main.xml**

複製 `layout/activity_main.xml` 全文,調整:statusPanel 從頂部橫條改為靠右側縱向(約束改為 `app:layout_constraintEnd_toEndOf="parent"` + `app:layout_constraintTop_toTopOf="parent"`,寬度 `wrap_content`、高度 `match_parent` 方向的等效約束)。所有 view id 保持不變。實作時以現有檔案為底本,僅改容器方向/約束,不增刪 id。

- [ ] **Step 3: MainActivity 處理旋轉重佈局**

`configChanges` 模式下旋轉會走 `onConfigurationChanged` 而不重建 Activity;
需重新 inflate 佈局並重綁 view:

```kotlin
override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
    super.onConfigurationChanged(newConfig)
    setContentView(R.layout.activity_main)
    initViews()
    // 重新掛 preview surface
    cameraProvider?.let { bindCameraUseCases() }
    // 切換後靜音 1 秒,避免緩衝重置期間亂報
    stateMachine.muteFor(1000L)
}
```

- [ ] **Step 4: StateMachine 加 muteFor**

`app/src/main/java/com/example/trafficlight/logic/StateMachine.kt` 加:

```kotlin
private var mutedUntilMs = 0L

fun muteFor(durationMs: Long) {
    mutedUntilMs = System.currentTimeMillis() + durationMs
}
```

並在觸發 `shouldAnnounce` 的判斷處(現有 `processClassification` 內
決定播報的位置)加上 `if (System.currentTimeMillis() < mutedUntilMs) return` 的前置檢查
(讀 StateMachine.kt 現況後放在狀態確立、即將設定 announce flag 之前)。

- [ ] **Step 5: 編譯驗證**

Run: `./gradlew :app:assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add app
git commit -m "feat(app): landscape layout and rotation-safe rebinding with announcement mute"
```

---

### Task 10: desktop-harness — 桌機影片回歸

**Files:**
- Modify: `settings.gradle`(include 加 `':desktop-harness'`)
- Create: `desktop-harness/build.gradle`
- Create: `desktop-harness/src/main/kotlin/com/example/trafficlight/harness/Main.kt`
- Create: `desktop-harness/src/main/kotlin/com/example/trafficlight/harness/DesktopModelRunner.kt`
- Create: `desktop-harness/README.md`

**Interfaces:**
- Consumes: core 全部;ONNX 模型檔於 `app/src/main/assets/models/`
- Produces: CLI:`./gradlew :desktop-harness:run --args="<framesDir> <outDir> [rollDeg]"`
  - 讀 `framesDir` 內按檔名排序的 PNG/JPG 幀
  - 輸出 `outDir/events.csv`(`frameIndex,action,confidence,nearVel,desiredAccel,calibValid`)
  - 輸出 `outDir/annotated/NNNNN.png`(疊加規劃路徑點與 STOP/GO 標籤)
  - 選配 `rollDeg`:先把每幀旋轉該角度再餵 pipeline(模擬歪斜手機),同時餵對應 IMU tilt

- [ ] **Step 1: desktop-harness/build.gradle**

```groovy
plugins {
    id 'org.jetbrains.kotlin.jvm'
    id 'application'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

application {
    mainClass = 'com.example.trafficlight.harness.MainKt'
}

dependencies {
    implementation project(':core')
    implementation 'com.microsoft.onnxruntime:onnxruntime:1.19.2'
    testImplementation 'junit:junit:4.13.2'
}
```

- [ ] **Step 2: DesktopModelRunner.kt**

與 app 的 OrtModelRunner 相同邏輯(ai.onnxruntime JVM 版 API 相同):

```kotlin
package com.example.trafficlight.harness

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.trafficlight.core.pipeline.ModelRunner
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class DesktopModelRunner(visionPath: File, policyPath: File) : ModelRunner, AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val vision: OrtSession = env.createSession(visionPath.readBytes())
    private val policy: OrtSession = env.createSession(policyPath.readBytes())

    override fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray {
        fun tensorOf(data: ByteArray): OnnxTensor {
            val buf = ByteBuffer.allocateDirect(data.size).order(ByteOrder.nativeOrder())
            buf.put(data); buf.rewind()
            return OnnxTensor.createTensor(env, buf, longArrayOf(1, 12, 128, 256), OnnxJavaType.UINT8)
        }
        tensorOf(stackedImg).use { img ->
            tensorOf(stackedBigImg).use { bigImg ->
                vision.run(mapOf("img" to img, "big_img" to bigImg)).use { out ->
                    return readFloats(out.get(0) as OnnxTensor)
                }
            }
        }
    }

    override fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray {
        fun tensorOf(data: FloatArray, shape: LongArray): OnnxTensor {
            val buf = FloatBuffer.allocate(data.size)
            buf.put(data); buf.rewind()
            return OnnxTensor.createTensor(env, buf, shape)
        }
        tensorOf(desire, longArrayOf(1, 25, 8)).use { d ->
            tensorOf(trafficConvention, longArrayOf(1, 2)).use { tc ->
                tensorOf(featuresBuffer, longArrayOf(1, 25, 512)).use { f ->
                    policy.run(mapOf("desire_pulse" to d, "traffic_convention" to tc, "features_buffer" to f)).use { out ->
                        return readFloats(out.get(0) as OnnxTensor)
                    }
                }
            }
        }
    }

    private fun readFloats(tensor: OnnxTensor): FloatArray {
        val buffer = tensor.floatBuffer
        buffer.rewind()
        val data = FloatArray(buffer.remaining())
        buffer.get(data)
        return data
    }

    override fun close() {
        vision.close()
        policy.close()
        env.close()
    }
}
```

- [ ] **Step 3: Main.kt**

```kotlin
package com.example.trafficlight.harness

import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.pipeline.DrivingPipeline
import java.awt.Color
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

fun bufferedToIntImage(img: BufferedImage): IntImage {
    val pixels = IntArray(img.width * img.height)
    img.getRGB(0, 0, img.width, img.height, pixels, 0, img.width)
    return IntImage(img.width, img.height, pixels)
}

/** 以影像中心旋轉 rollDeg(模擬歪斜的手機),尺寸不變、出界填黑。 */
fun rotateByDegrees(img: BufferedImage, rollDeg: Double): BufferedImage {
    val out = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.color = Color.BLACK
    g.fillRect(0, 0, img.width, img.height)
    val t = AffineTransform.getRotateInstance(
        Math.toRadians(rollDeg), img.width / 2.0, img.height / 2.0)
    g.drawImage(img, t, null)
    g.dispose()
    return out
}

fun main(args: Array<String>) {
    require(args.size >= 2) { "usage: <framesDir> <outDir> [rollDeg]" }
    val framesDir = File(args[0])
    val outDir = File(args[1]).apply { mkdirs() }
    val rollDeg = if (args.size >= 3) args[2].toDouble() else 0.0
    val annotatedDir = File(outDir, "annotated").apply { mkdirs() }

    val modelsDir = File("app/src/main/assets/models")
    val runner = DesktopModelRunner(
        File(modelsDir, "openpilot_driving_vision.onnx"),
        File(modelsDir, "openpilot_driving_policy.onnx")
    )
    val pipeline = DrivingPipeline(runner)

    val frames = framesDir.listFiles { f -> f.extension.lowercase() in setOf("png", "jpg", "jpeg") }
        ?.sortedBy { it.name } ?: emptyList()
    require(frames.isNotEmpty()) { "no frames in $framesDir" }

    val events = StringBuilder("frameIndex,action,confidence,nearVel,desiredAccel,calibValid\n")
    var timestampMs = 0L

    frames.forEachIndexed { index, file ->
        val original = ImageIO.read(file)
        val input = if (rollDeg != 0.0) rotateByDegrees(original, rollDeg) else original
        // 模擬 IMU:回報與影像旋轉相同的 roll(50Hz × 3 個樣本/幀 @ ~15fps)
        repeat(3) {
            timestampMs += 20
            pipeline.onImuTilt(Tilt(rollDeg.toFloat(), 0f), timestampMs)
        }
        val result = pipeline.processFrame(bufferedToIntImage(input), rotationDegrees = 0, horizontalFovDeg = 72f)
        val plan = result.plan

        events.append("$index,${plan.action},${"%.2f".format(plan.confidence)},")
        events.append("${"%.2f".format(plan.nearVelocity)},${"%.2f".format(plan.desiredAcceleration)},")
        events.append("${result.calibration.valid}\n")

        // 標注輸出
        val annotated = BufferedImage(input.width, input.height, BufferedImage.TYPE_INT_RGB)
        val g = annotated.createGraphics()
        g.drawImage(input, 0, 0, null)
        g.color = when (plan.action.name) {
            "STOP" -> Color.RED
            "GO" -> Color.GREEN
            else -> Color.GRAY
        }
        g.fillRect(10, 10, 180, 40)
        g.color = Color.WHITE
        g.drawString("${plan.action} v=${"%.1f".format(plan.nearVelocity)}", 20, 35)
        g.dispose()
        ImageIO.write(annotated, "png", File(annotatedDir, "%05d.png".format(index)))

        if (index % 20 == 0) println("frame $index/${frames.size}: ${plan.action}")
    }

    File(outDir, "events.csv").writeText(events.toString())
    runner.close()
    println("done. events: ${File(outDir, "events.csv").absolutePath}")
}
```

- [ ] **Step 4: README.md(取得測試影片)**

```markdown
# desktop-harness

在桌機上跑與手機完全相同的 core pipeline,驗證 openpilot 模型行為。

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
- `out/annotated/*.png` — 疊加 STOP/GO 標籤的幀(可用 ffmpeg 合回影片檢視)

    ffmpeg -framerate 15 -i out/annotated/%05d.png -c:v libx264 out/annotated.mp4

## 傾斜模擬(驗證 roll 修正)

    ./gradlew :desktop-harness:run --args="frames out-tilt8 8"

把每幀旋轉 8° 並餵入 roll=8° 的 IMU 樣本;`out-tilt8/events.csv`
應與 `out/events.csv` 的 action 序列大致一致(見 TiltRegressionTest)。
```

- [ ] **Step 5: 編譯驗證**

Run: `./gradlew :desktop-harness:build`
Expected: BUILD SUCCESSFUL

- [ ] **Step 6: Commit**

```bash
git add settings.gradle desktop-harness
git commit -m "feat(harness): desktop video regression harness with tilt simulation"
```

---

### Task 11: 傾斜修正回歸測試

**Files:**
- Create: `desktop-harness/src/test/kotlin/com/example/trafficlight/harness/TiltRegressionTest.kt`

**Interfaces:**
- Consumes: `DesktopModelRunner`, `DrivingPipeline`, `rotateByDegrees`, `bufferedToIntImage`
- 模型檔不存在時 skip(`org.junit.Assume`),CI 無模型也能過。

- [ ] **Step 1: 寫測試**

用合成幀(漸層+地平線圖案)證明:輸入旋轉 8° + IMU roll 8° 修正後,warp 輸出與未旋轉版本近似。這不需要 ONNX 模型,直接測 core 的 warp:

```kotlin
package com.example.trafficlight.harness

import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.frame.warpToModelFrame
import com.example.trafficlight.core.geometry.sourceFromModelFrame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage

class TiltRegressionTest {

    /** 產生上灰下黑的地平線合成幀 */
    private fun horizonFrame(w: Int = 1280, h: Int = 720): BufferedImage {
        val img = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.color = Color(180, 180, 220); g.fillRect(0, 0, w, h / 2)
        g.color = Color(40, 40, 40); g.fillRect(0, h / 2, w, h / 2)
        g.dispose()
        return img
    }

    private fun meanAbsDiff(a: IntImage, b: IntImage): Double {
        var sum = 0.0
        for (i in a.pixels.indices) {
            val pa = a.pixels[i] and 0xFF
            val pb = b.pixels[i] and 0xFF
            sum += kotlin.math.abs(pa - pb)
        }
        return sum / a.pixels.size
    }

    @Test
    fun rollCorrectionRecoversUprightWarp() {
        val upright = horizonFrame()
        val tilted = rotateByDegrees(upright, 8.0)

        val baseline = bufferedToIntImage(upright).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, 0f, 0f, 0f, false), false)

        // 無修正:歪斜輸入 + roll=0 warp → 差異大
        val uncorrected = bufferedToIntImage(tilted).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, 0f, 0f, 0f, false), false)

        // 有修正:歪斜輸入 + roll=8° warp → 接近 baseline
        val rollRad = Math.toRadians(8.0).toFloat()
        val corrected = bufferedToIntImage(tilted).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, rollRad, 0f, 0f, false), false)

        val diffUncorrected = meanAbsDiff(baseline, uncorrected)
        val diffCorrected = meanAbsDiff(baseline, corrected)
        assertTrue(
            "corrected ($diffCorrected) should beat uncorrected ($diffUncorrected)",
            diffCorrected < diffUncorrected * 0.5
        )
    }
}
```

注意:此測試錨定 roll 修正的**方向與有效性**。若 `diffCorrected` 反而更大,代表 roll 符號接反——修 `sourceFromModelFrame` 的 roll 傳入符號(或 `tiltFromGravity` 的正負),不要改測試閾值。

- [ ] **Step 2: 跑測試**

Run: `./gradlew :desktop-harness:test`
Expected: PASS

- [ ] **Step 3: 全量驗證**

Run: `./gradlew :core:test :desktop-harness:test :app:assembleDebug`
Expected: 全綠 + APK 產出

- [ ] **Step 4: Commit**

```bash
git add desktop-harness
git commit -m "test(harness): tilt-correction regression proving roll compensation"
```

---

### Task 12: 文件更新與收尾

**Files:**
- Modify: `readme.md`(更新專案結構、模組說明、桌機驗證流程;移除已過時的 YOLO/MobileNet 敘述,標明 openpilot pipeline 現況)
- Modify: `MODEL_SETUP.md`(改為 openpilot 模型說明:兩個檔名、來源 commaai/openpilot、輸入格式)

- [ ] **Step 1: 更新 readme.md**

重點段落:
- 架構圖:`:core`(純 JVM,可測)/`:app`(Android 殼)/`:desktop-harness`(桌機回歸)
- 新功能:IMU 傾斜自動修正、直/橫向自動調整、動態 FOV
- 驗證流程:`./gradlew :core:test :desktop-harness:test` + harness 影片流程(引用 desktop-harness/README.md)
- 上機煙霧測試清單:相機開啟、STOP/GO 有聲音、直橫向切換不 crash、手機故意歪 10° 擺放後 overlay 路徑仍貼合路面

- [ ] **Step 2: 全量最終驗證**

Run: `./gradlew :core:test :desktop-harness:test :app:assembleDebug`
Expected: 全綠

- [ ] **Step 3: Commit**

```bash
git add readme.md MODEL_SETUP.md
git commit -m "docs: update README for core/app/harness architecture"
```

---

## 驗收對照(spec → task)

| Spec 要求 | Task |
|---|---|
| IMU roll 修正進 warp | 2, 6, 7, 8 |
| IMU pitch 當校正初始值 | 6 |
| 突變偵測重置緩衝 | 6, 7 |
| 幀按 rotationDegrees 轉正(修直向 bug) | 3, 7, 8 |
| 方向切換重置時序緩衝 | 7 |
| 動態 FOV(取代寫死 72°) | 8 |
| 直橫向 UI | 9 |
| 切換過渡靜音 | 9 |
| 桌機影片回歸 + golden | 10 |
| 傾斜修正驗證(不上機) | 10, 11 |
| IMU 不可用 fallback | 8(ImuManager.available;engine 未收 IMU 樣本時 CalibrationFusion 走純模型路徑) |
| 文件 | 12 |
