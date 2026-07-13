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
        val runner2 = FakeRunner()
        val p2 = DrivingPipeline(runner2)
        p2.processFrame(frame(), 0, 72f)
        assertTrue(p2.bufferedFrameReady)
        p2.processFrame(frame(480, 640), 90, 72f)         // 直向
        // 切換後第一幀:stacked 的前半 == 後半(prior=current,因為緩衝被清)
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
