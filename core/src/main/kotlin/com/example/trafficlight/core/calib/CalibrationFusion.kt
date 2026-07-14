package com.example.trafficlight.core.calib

import kotlin.math.abs
import kotlin.math.exp

data class CalibrationState(
    val rollDeg: Float,
    val pitchDeg: Float,
    val yawDeg: Float,
    val heightM: Float,
    val valid: Boolean,
    val sampleCount: Int,
    val speedMps: Float = 0f,
    val movingFastEnough: Boolean = false
)

/**
 * 融合兩個校正來源:
 * - IMU(重力):roll 即時修正 + pitch 冷啟動初始值,低通濾波、突變偵測
 * - 模型 pose 輸出:pitch/yaw/height 慢速細化(移植自原 InferenceEngine.updateAutoCalibration)
 */
class CalibrationFusion(private val initialPitchDeg: Float = 5.5f) {

    private var imuRollDeg = 0f
    private var imuPitchDeg = initialPitchDeg
    private var imuInitialized = false
    private var lastImuTimestampMs = 0L

    private var speedMps = 0f
    private var modelPitchDeg = initialPitchDeg
    private var modelYawDeg = 0f
    private var heightM = 1.35f
    private var sampleCount = 0
    private var warpWasActive = false
    var valid = false
        private set

    companion object {
        const val MIN_SAMPLES = 20
        /** 模型自校正只在車速達此門檻時累積(靜止時模型 pose 輸出無意義) */
        const val MIN_CALIB_SPEED_KMH = 20f
        const val MIN_CALIB_SPEED_MPS = MIN_CALIB_SPEED_KMH / 3.6f
        private const val IMU_TAU_S = 0.5f            // 低通時間常數
        private const val SUDDEN_CHANGE_DEG = 6f       // 觀測值 vs 濾波值差異閾值
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
            // 濾波後仍偏離 → 真突變(非單次雜訊):快速跟上並要求重置
            imuRollDeg = tilt.rollDeg
            imuPitchDeg = tilt.pitchDeg
            reset()
            return true
        }
        return false
    }

    /** GPS 車速(m/s)。低於門檻時模型自校正不累積。 */
    fun onSpeed(mps: Float) {
        speedMps = mps
    }

    /** 移植自 InferenceEngine.updateAutoCalibration;回傳 true = warp 剛轉為啟用。 */
    fun onModelOutputs(visionData: FloatArray): Boolean {
        if (speedMps < MIN_CALIB_SPEED_MPS) return false
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
        return CalibrationState(imuRollDeg, pitch, modelYawDeg, heightM, valid, sampleCount,
            speedMps, speedMps >= MIN_CALIB_SPEED_MPS)
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
