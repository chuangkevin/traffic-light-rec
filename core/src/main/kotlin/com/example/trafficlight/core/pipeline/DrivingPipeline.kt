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
    /** stackedImg / stackedBigImg:2 幀 × 6ch × 128×256 = 393216 bytes,uint8 */
    fun runVision(stackedImg: ByteArray, stackedBigImg: ByteArray): FloatArray
    fun runPolicy(featuresBuffer: FloatArray, desire: FloatArray, trafficConvention: FloatArray): FloatArray
}

data class PipelineResult(val plan: DrivingPlan, val calibration: CalibrationState)

/**
 * 端到端 openpilot pipeline:轉正 → warp(含 roll 修正)→ YUV 打包 →
 * vision → 特徵緩衝 → policy → plan 解析。
 * app 與 desktop-harness 共用;方向切換與 IMU 突變時自動重置時序緩衝。
 */
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

    /** GPS 車速(m/s),用於校正速度閘門。 */
    fun onSpeed(mps: Float) = calibration.onSpeed(mps)

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
            // warp 剛啟用:輸入幾何改變,清特徵緩衝(保留本幀作為新起點)
            for (f in featureBuffer) f.fill(0f)
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
