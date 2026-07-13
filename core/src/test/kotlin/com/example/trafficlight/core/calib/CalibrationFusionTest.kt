package com.example.trafficlight.core.calib

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationFusionTest {
    @Test fun imuRollIsLowPassed() {
        val f = CalibrationFusion()
        f.onImuTilt(Tilt(rollDeg = 10f, pitchDeg = 5f), 0L)
        // 第一個樣本直接初始化;之後閾值內的變化(10→6,差 4°<6°)走低通
        f.onImuTilt(Tilt(rollDeg = 6f, pitchDeg = 5f), 20L)
        val afterDrop = f.state().rollDeg
        assertTrue("low-pass should lag, got $afterDrop", afterDrop > 6.5f && afterDrop < 10f)
        // 持續餵 3 秒(50Hz)→ 收斂到 6 附近
        var t = 20L
        repeat(150) { t += 20; f.onImuTilt(Tilt(6f, 5f), t) }
        assertEquals(6f, f.state().rollDeg, 1.0f)
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
