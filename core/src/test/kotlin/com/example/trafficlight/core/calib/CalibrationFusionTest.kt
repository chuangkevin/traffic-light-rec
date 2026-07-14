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
        // 突然 +20°(手機被重新擺放)→ 通知重置緩衝、IMU 立即跟上
        val reset = f.onImuTilt(Tilt(20f, 0f), t + 20)
        assertTrue(reset)
        assertEquals(0, f.state().sampleCount)
        assertEquals(20f, f.state().rollDeg, 0.5f)
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

    private fun plausibleVision(): FloatArray {
        val vision = FloatArray(1600)
        vision[87] = 5f
        vision[87 + 6] = -3f
        vision[99 + 1] = Math.toRadians(3.0).toFloat()
        vision[99 + 4] = -3f
        vision[99 + 5] = -3f
        vision[105 + 2] = 1.4f
        vision[105 + 8] = -3f
        return vision
    }

    @Test fun stationaryPhoneDoesNotCalibrate() {
        val f = CalibrationFusion()
        // 未提供速度(=0)→ 即使模型輸出可信也不累積
        repeat(25) { f.onModelOutputs(plausibleVision()) }
        assertEquals(0, f.state().sampleCount)
        assertFalse(f.state().valid)
        assertFalse(f.state().movingFastEnough)
    }

    @Test fun slowSpeedDoesNotCalibrate() {
        val f = CalibrationFusion()
        f.onSpeed(15f / 3.6f) // 15 km/h < 20 門檻
        repeat(25) { f.onModelOutputs(plausibleVision()) }
        assertEquals(0, f.state().sampleCount)
    }

    @Test fun drivingSpeedEnablesCalibration() {
        val f = CalibrationFusion()
        f.onSpeed(30f / 3.6f) // 30 km/h
        repeat(25) { f.onModelOutputs(plausibleVision()) }
        assertTrue(f.state().valid)
        assertTrue(f.state().movingFastEnough)
    }

    @Test fun forceCalibrateIsImmediatelyValid() {
        val f = CalibrationFusion()
        f.onImuTilt(Tilt(rollDeg = 2f, pitchDeg = -8f), 0L)
        f.forceCalibrate()
        assertTrue(f.state().valid)
        assertEquals(-8f, f.state().pitchDeg, 0.1f)   // 直接採用 IMU pitch
        assertEquals(2f, f.state().rollDeg, 0.1f)
    }

    @Test fun forceCalibrateSurvivesSuddenTilt() {
        val f = CalibrationFusion()
        var t = 0L
        repeat(50) { t += 20; f.onImuTilt(Tilt(0f, -5f), t) }
        f.forceCalibrate()
        assertTrue(f.state().valid)
        assertTrue(f.onImuTilt(Tilt(20f, -5f), t + 20))  // 突變 → 重置緩衝
        assertTrue(f.state().valid)                       // 但校正不作廢(IMU 即時跟隨)
        assertEquals(20f, f.state().rollDeg, 0.5f)
    }

    @Test fun modelOutputsRefineYawAndActivateWarp() {
        val f = CalibrationFusion()
        f.onSpeed(10f) // 36 km/h,超過校正速度門檻
        var activated = false
        repeat(25) { if (f.onModelOutputs(plausibleVision())) activated = true }
        assertTrue(activated)          // 累積 20 樣本後 warp 啟用一次
        assertTrue(f.state().valid)
    }

    // pitch 永遠即時跟隨 IMU,不因校正 valid 而凍結或被模型覆蓋
    @Test fun pitchAlwaysTracksImuEvenWhenValid() {
        val f = CalibrationFusion()
        var t = 0L
        repeat(150) { t += 20; f.onImuTilt(Tilt(0f, -12f), t) }
        f.forceCalibrate()
        assertTrue(f.state().valid)
        assertEquals(-12f, f.state().pitchDeg, 1f)
        // 手機姿態緩慢改變(每步 <6° 不觸發重置)→ pitch 必須跟上
        repeat(300) { t += 20; f.onImuTilt(Tilt(0f, -4f), t) }
        assertEquals(-4f, f.state().pitchDeg, 1f)
        assertTrue(f.state().valid)   // 校正狀態不受影響
    }
}
