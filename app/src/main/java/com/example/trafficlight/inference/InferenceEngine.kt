package com.example.trafficlight.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

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

enum class DrivingAction {
    STOP,
    GO,
    HOLD
}

data class PlanPoint(
    val x: Float,
    val y: Float
)

data class CameraCalibrationEstimate(
    val pitchDeg: Float = 5.5f,
    val yawDeg: Float = 0f,
    val heightM: Float = 1.35f,
    val valid: Boolean = false,
    val sampleCount: Int = 0
)

class InferenceEngine(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var visionSession: OrtSession? = null
    private var policySession: OrtSession? = null

    private val modelWidth = 512
    private val modelHeight = 256
    private val packedFrameSize = 6 * 128 * 256
    private val featureBuffer = Array(POLICY_FRAMES) { FloatArray(FEATURE_LEN) }
    private var previousFrame: ByteArray? = null
    private var previousBigFrame: ByteArray? = null
    private var autoPitchDeg = 5.5f
    private var autoYawDeg = 0f
    private var autoHeightM = 1.35f
    private var calibrationValid = false
    private var calibrationSampleCount = 0
    private var warpWasActive = false

    companion object {
        private const val VISION_MODEL = "models/openpilot_driving_vision.onnx"
        private const val POLICY_MODEL = "models/openpilot_driving_policy.onnx"
        private const val POLICY_FRAMES = 25
        private const val FEATURE_LEN = 512
        private const val VISION_POSE_START = 87
        private const val VISION_WIDE_FROM_DEVICE_EULER_START = 99
        private const val VISION_ROAD_TRANSFORM_START = 105
        private const val VISION_HIDDEN_STATE_START = 1064
        private const val PLAN_VALUES = 33 * 15
        private const val PLAN_WIDTH = 15
        private const val PLAN_ACCELERATION_X = 6
        private const val PLAN_VELOCITY_X = 3
        private const val MIN_STABLE_DELAY_S = 0.3f
        private const val MODEL_ACTION_T_S = 0.075f
        private const val STOPPING_VELOCITY_MPS = 0.3f
        private const val GO_ACCELERATION_MPS2 = 0.45f
        private const val GO_VELOCITY_DELTA_MPS = 0.35f
        private const val CALIBRATION_MIN_SAMPLES = 20
        private const val SOURCE_CAMERA_FOV_DEG = 72f
        private const val MEDMODEL_FL = 910f
        private const val MEDMODEL_CX = 256f
        private const val MEDMODEL_CY = 47.6f
        private const val SBIGMODEL_FL = 455f
        private const val SBIGMODEL_CX = 256f
        private const val SBIGMODEL_CY = 151.8f
    }

    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d("InferenceEngine", "初始化 openpilot driving models...")
            ortEnvironment = OrtEnvironment.getEnvironment()
            val env = ortEnvironment ?: return@withContext false

            val visionBytes = context.assets.open(VISION_MODEL).readBytes()
            val policyBytes = context.assets.open(POLICY_MODEL).readBytes()
            Log.d("InferenceEngine", "vision model: ${visionBytes.size} bytes")
            Log.d("InferenceEngine", "policy model: ${policyBytes.size} bytes")

            visionSession = env.createSession(visionBytes)
            policySession = env.createSession(policyBytes)
            Log.d("InferenceEngine", "openpilot models ready")
            true
        } catch (e: Exception) {
            Log.e("InferenceEngine", "openpilot model init failed: ${e.message}", e)
            false
        }
    }

    suspend fun analyzeDrivingPlan(bitmap: Bitmap): DrivingPlanResult = withContext(Dispatchers.Default) {
        val env = ortEnvironment ?: return@withContext emptyPlanResult()
        val vision = visionSession ?: return@withContext emptyPlanResult()
        val policy = policySession ?: return@withContext emptyPlanResult()

        try {
            val currentFrame = packOpenpilotFrame(bitmap, bigModelFrame = false)
            val priorFrame = previousFrame ?: currentFrame
            previousFrame = currentFrame
            val currentBigFrame = packOpenpilotFrame(bitmap, bigModelFrame = true)
            val priorBigFrame = previousBigFrame ?: currentBigFrame
            previousBigFrame = currentBigFrame

            val stacked = ByteBuffer
                .allocateDirect(packedFrameSize * 2)
                .order(ByteOrder.nativeOrder())
            stacked.put(priorFrame)
            stacked.put(currentFrame)
            stacked.rewind()
            val bigStacked = ByteBuffer
                .allocateDirect(packedFrameSize * 2)
                .order(ByteOrder.nativeOrder())
            bigStacked.put(priorBigFrame)
            bigStacked.put(currentBigFrame)
            bigStacked.rewind()

            val imgTensor = OnnxTensor.createTensor(
                env,
                stacked,
                longArrayOf(1, 12, 128, 256),
                OnnxJavaType.UINT8
            )
            val bigImgTensor = OnnxTensor.createTensor(
                env,
                bigStacked,
                longArrayOf(1, 12, 128, 256),
                OnnxJavaType.UINT8
            )

            val visionOutputs = vision.run(mapOf("img" to imgTensor, "big_img" to bigImgTensor))
            val visionTensor = visionOutputs.get(0) as OnnxTensor
            val visionData = readFloatOutput(visionTensor)
            if (updateAutoCalibration(visionData)) {
                resetTemporalBuffers()
            }

            shiftFeatureBuffer(visionData.copyOfRange(VISION_HIDDEN_STATE_START, VISION_HIDDEN_STATE_START + FEATURE_LEN))

            val desireTensor = createZeroHalfTensor(env, longArrayOf(1, POLICY_FRAMES.toLong(), 8))
            val trafficConventionTensor = createTrafficConventionTensor(env)
            val featuresTensor = createFeaturesTensor(env)

            val policyOutputs = policy.run(mapOf(
                "desire_pulse" to desireTensor,
                "traffic_convention" to trafficConventionTensor,
                "features_buffer" to featuresTensor
            ))
            val policyTensor = policyOutputs.get(0) as OnnxTensor
            val policyData = readFloatOutput(policyTensor)
            val result = parseDrivingPlan(policyData)

            policyTensor.close()
            policyOutputs.close()
            desireTensor.close()
            trafficConventionTensor.close()
            featuresTensor.close()
            visionTensor.close()
            visionOutputs.close()
            imgTensor.close()
            bigImgTensor.close()

            result
        } catch (e: Exception) {
            Log.e("InferenceEngine", "openpilot inference failed: ${e.message}", e)
            emptyPlanResult()
        }
    }

    private fun parseDrivingPlan(policyData: FloatArray): DrivingPlanResult {
        if (policyData.size < PLAN_VALUES) return emptyPlanResult()

        val nearVelocity = policyData[PLAN_VELOCITY_X]
        val futureVelocity = policyData[16 * PLAN_WIDTH + PLAN_VELOCITY_X]
        val path = (0 until 33).map { i ->
            PlanPoint(
                x = policyData[i * PLAN_WIDTH],
                y = policyData[i * PLAN_WIDTH + 1]
            )
        }.filter { it.x.isFinite() && it.y.isFinite() && it.x >= 0f }
        val accelerationNow = policyData[PLAN_ACCELERATION_X]
        val desiredAcceleration = getOpenpilotDesiredAcceleration(policyData, nearVelocity, accelerationNow)
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

        return DrivingPlanResult(
            shouldStop,
            shouldGo,
            confidence,
            nearVelocity,
            futureVelocity,
            desiredAcceleration,
            action,
            path,
            CameraCalibrationEstimate(autoPitchDeg, autoYawDeg, autoHeightM, calibrationValid, calibrationSampleCount)
        )
    }

    private fun emptyPlanResult(): DrivingPlanResult {
        return DrivingPlanResult(
            shouldStop = false,
            shouldGo = false,
            confidence = 0f,
            nearVelocity = 0f,
            futureVelocity = 0f,
            desiredAcceleration = 0f,
            action = DrivingAction.HOLD,
            calibration = CameraCalibrationEstimate(autoPitchDeg, autoYawDeg, autoHeightM, calibrationValid, calibrationSampleCount)
        )
    }

    private fun resetTemporalBuffers() {
        for (frame in featureBuffer) frame.fill(0f)
        previousFrame = null
        previousBigFrame = null
    }

    private fun updateAutoCalibration(visionData: FloatArray): Boolean {
        if (visionData.size <= VISION_HIDDEN_STATE_START) return false
        val wasValid = calibrationValid

        val poseTransX = visionData[VISION_POSE_START]
        val poseStdX = safeExp(visionData[VISION_POSE_START + 6])
        val roadHeight = kotlin.math.abs(visionData[VISION_ROAD_TRANSFORM_START + 2])
        val roadHeightStd = safeExp(visionData[VISION_ROAD_TRANSFORM_START + 8])
        val wideRoll = visionData[VISION_WIDE_FROM_DEVICE_EULER_START]
        val widePitch = visionData[VISION_WIDE_FROM_DEVICE_EULER_START + 1]
        val wideYaw = visionData[VISION_WIDE_FROM_DEVICE_EULER_START + 2]
        val widePitchStd = safeExp(visionData[VISION_WIDE_FROM_DEVICE_EULER_START + 4])
        val wideYawStd = safeExp(visionData[VISION_WIDE_FROM_DEVICE_EULER_START + 5])

        val poseReliable = poseTransX.isFinite() && kotlin.math.abs(poseTransX) > 0.05f && poseStdX < 2.5f
        val eulerReliable = poseReliable &&
            wideRoll.isFinite() && widePitch.isFinite() && wideYaw.isFinite() &&
            widePitchStd.isFinite() && wideYawStd.isFinite() &&
            widePitchStd < 0.20f && wideYawStd < 0.20f
        val observedPitchDeg = if (eulerReliable) Math.toDegrees(widePitch.toDouble()).toFloat() else null
        val observedYawDeg = if (eulerReliable) Math.toDegrees(wideYaw.toDouble()).toFloat() else null
        val observedHeight = if (roadHeight.isFinite() && roadHeightStd.isFinite() && roadHeightStd < 0.60f && roadHeight in 0.7f..2.2f) {
            roadHeight
        } else null

        var updated = false
        observedPitchDeg?.let {
            if (it in -12f..12f) {
                autoPitchDeg = lowPass(autoPitchDeg, it, 0.015f)
                updated = true
            }
        }
        observedYawDeg?.let {
            if (it in -12f..12f) {
                autoYawDeg = lowPass(autoYawDeg, it, 0.015f)
                updated = true
            }
        }
        observedHeight?.let {
            autoHeightM = lowPass(autoHeightM, it, 0.01f)
            updated = true
        }
        if (updated) calibrationSampleCount += 1
        calibrationValid = calibrationSampleCount >= CALIBRATION_MIN_SAMPLES
        val warpActivated = calibrationValid && !wasValid && !warpWasActive
        if (calibrationValid) warpWasActive = true
        return warpActivated
    }

    private fun safeExp(value: Float): Float {
        return if (value.isFinite()) kotlin.math.exp(value.coerceAtMost(11f)) else Float.POSITIVE_INFINITY
    }

    private fun lowPass(previous: Float, observed: Float, alpha: Float): Float {
        return previous + (observed - previous) * alpha
    }

    private fun getOpenpilotDesiredAcceleration(policyData: FloatArray, vNow: Float, aNow: Float): Float {
        val stableTargetVelocity = interpolatePlanVelocity(policyData, MIN_STABLE_DELAY_S)
        val vTarget = vNow + (MODEL_ACTION_T_S / MIN_STABLE_DELAY_S) * (stableTargetVelocity - vNow)
        return 2f * (vTarget - vNow) / MODEL_ACTION_T_S - aNow
    }

    private fun interpolatePlanVelocity(policyData: FloatArray, targetTimeS: Float): Float {
        var previousTime = 0f
        var previousVelocity = policyData[PLAN_VELOCITY_X]
        for (i in 1 until 33) {
            val time = modelTimeIndex(i)
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

    private fun modelTimeIndex(index: Int): Float {
        val normalized = index / 32f
        return 10f * normalized * normalized
    }

    private fun shiftFeatureBuffer(hiddenState: FloatArray) {
        for (i in 0 until POLICY_FRAMES - 1) {
            System.arraycopy(featureBuffer[i + 1], 0, featureBuffer[i], 0, FEATURE_LEN)
        }
        System.arraycopy(hiddenState, 0, featureBuffer[POLICY_FRAMES - 1], 0, FEATURE_LEN)
    }

    private fun createFeaturesTensor(env: OrtEnvironment): OnnxTensor {
        val buffer = FloatBuffer.allocate(POLICY_FRAMES * FEATURE_LEN)
        for (frame in featureBuffer) {
            for (value in frame) buffer.put(value)
        }
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, POLICY_FRAMES.toLong(), FEATURE_LEN.toLong()))
    }

    private fun createTrafficConventionTensor(env: OrtEnvironment): OnnxTensor {
        val buffer = FloatBuffer.allocate(2)
        buffer.put(1f)
        buffer.put(0f)
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, 2))
    }

    private fun createZeroHalfTensor(env: OrtEnvironment, shape: LongArray): OnnxTensor {
        val elements = shape.fold(1L) { acc, item -> acc * item }.toInt()
        val buffer = FloatBuffer.allocate(elements)
        repeat(elements) { buffer.put(0f) }
        buffer.rewind()
        return OnnxTensor.createTensor(env, buffer, shape)
    }

    private fun readFloatOutput(tensor: OnnxTensor): FloatArray {
        val buffer: FloatBuffer = tensor.floatBuffer
        buffer.rewind()
        val data = FloatArray(buffer.remaining())
        buffer.get(data)
        return data
    }

    private fun packOpenpilotFrame(bitmap: Bitmap, bigModelFrame: Boolean): ByteArray {
        val scaled = warpBitmapToModelFrame(bitmap, bigModelFrame)
        val pixels = IntArray(modelWidth * modelHeight)
        scaled.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight)
        scaled.recycle()

        val yPlane = IntArray(modelWidth * modelHeight)
        val uPlane = IntArray((modelWidth / 2) * (modelHeight / 2))
        val vPlane = IntArray((modelWidth / 2) * (modelHeight / 2))
        val packed = ByteArray(packedFrameSize)

        for (blockY in 0 until modelHeight step 2) {
            for (blockX in 0 until modelWidth step 2) {
                var uSum = 0
                var vSum = 0
                for (dy in 0..1) {
                    for (dx in 0..1) {
                        val x = blockX + dx
                        val y = blockY + dy
                        val pixel = pixels[y * modelWidth + x]
                        val r = (pixel shr 16) and 0xFF
                        val g = (pixel shr 8) and 0xFF
                        val b = pixel and 0xFF
                        val yy = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
                        val uu = (-0.169f * r - 0.331f * g + 0.5f * b + 128f).toInt().coerceIn(0, 255)
                        val vv = (0.5f * r - 0.419f * g - 0.081f * b + 128f).toInt().coerceIn(0, 255)
                        yPlane[y * modelWidth + x] = yy
                        uSum += uu
                        vSum += vv
                    }
                }
                val uvIndex = (blockY / 2) * (modelWidth / 2) + (blockX / 2)
                uPlane[uvIndex] = uSum / 4
                vPlane[uvIndex] = vSum / 4
            }
        }

        val halfW = modelWidth / 2
        val halfH = modelHeight / 2
        for (y in 0 until halfH) {
            for (x in 0 until halfW) {
                val base = y * halfW + x
                // Match openpilot frames_to_tensor: Y top-left, bottom-left, top-right, bottom-right, then U, V.
                packed[base] = yPlane[(y * 2) * modelWidth + x * 2].toByte()
                packed[halfW * halfH + base] = yPlane[(y * 2 + 1) * modelWidth + x * 2].toByte()
                packed[2 * halfW * halfH + base] = yPlane[(y * 2) * modelWidth + x * 2 + 1].toByte()
                packed[3 * halfW * halfH + base] = yPlane[(y * 2 + 1) * modelWidth + x * 2 + 1].toByte()
                packed[4 * halfW * halfH + base] = uPlane[base].toByte()
                packed[5 * halfW * halfH + base] = vPlane[base].toByte()
            }
        }

        return packed
    }

    private fun warpBitmapToModelFrame(bitmap: Bitmap, bigModelFrame: Boolean): Bitmap {
        if (!calibrationValid) return centerCropScale(bitmap, modelWidth, modelHeight)

        val sourceFromModel = sourceFromModelFrameMatrix(bitmap.width.toFloat(), bitmap.height.toFloat(), bigModelFrame)
        val modelFromSource = invert3x3(sourceFromModel) ?: return centerCropScale(bitmap, modelWidth, modelHeight)
        val warped = Bitmap.createBitmap(modelWidth, modelHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(warped)
        canvas.drawColor(Color.BLACK)
        val matrix = Matrix().apply {
            setValues(floatArrayOf(
                modelFromSource[0][0], modelFromSource[0][1], modelFromSource[0][2],
                modelFromSource[1][0], modelFromSource[1][1], modelFromSource[1][2],
                modelFromSource[2][0], modelFromSource[2][1], modelFromSource[2][2]
            ))
        }
        canvas.drawBitmap(bitmap, matrix, null)
        return warped
    }

    private fun sourceFromModelFrameMatrix(sourceWidth: Float, sourceHeight: Float, bigModelFrame: Boolean): Array<FloatArray> {
        val sourceFl = sourceWidth / (2f * kotlin.math.tan(Math.toRadians((SOURCE_CAMERA_FOV_DEG / 2f).toDouble())).toFloat())
        val sourceIntrinsics = arrayOf(
            floatArrayOf(sourceFl, 0f, sourceWidth / 2f),
            floatArrayOf(0f, sourceFl, sourceHeight / 2f),
            floatArrayOf(0f, 0f, 1f)
        )
        val viewFromDevice = arrayOf(
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(0f, 0f, 1f),
            floatArrayOf(1f, 0f, 0f)
        )
        val deviceFromCalib = rotationFromEuler(
            roll = 0f,
            pitch = Math.toRadians(autoPitchDeg.toDouble()).toFloat(),
            yaw = Math.toRadians(autoYawDeg.toDouble()).toFloat()
        )
        val cameraFromCalib = multiply3x3(multiply3x3(sourceIntrinsics, viewFromDevice), deviceFromCalib)
        return multiply3x3(cameraFromCalib, calibFromModelFrame(bigModelFrame))
    }

    private fun calibFromModelFrame(bigModelFrame: Boolean): Array<FloatArray> {
        val modelFl = if (bigModelFrame) SBIGMODEL_FL else MEDMODEL_FL
        val modelCx = if (bigModelFrame) SBIGMODEL_CX else MEDMODEL_CX
        val modelCy = if (bigModelFrame) SBIGMODEL_CY else MEDMODEL_CY
        val medIntrinsics = arrayOf(
            floatArrayOf(modelFl, 0f, modelCx),
            floatArrayOf(0f, modelFl, modelCy),
            floatArrayOf(0f, 0f, 1f)
        )
        val viewFromDevice = arrayOf(
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(0f, 0f, 1f),
            floatArrayOf(1f, 0f, 0f)
        )
        return invert3x3(multiply3x3(medIntrinsics, viewFromDevice)) ?: identity3x3()
    }

    private fun rotationFromEuler(roll: Float, pitch: Float, yaw: Float): Array<FloatArray> {
        val cr = kotlin.math.cos(roll)
        val sr = kotlin.math.sin(roll)
        val cp = kotlin.math.cos(pitch)
        val sp = kotlin.math.sin(pitch)
        val cy = kotlin.math.cos(yaw)
        val sy = kotlin.math.sin(yaw)
        val rollMatrix = arrayOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(0f, cr, -sr),
            floatArrayOf(0f, sr, cr)
        )
        val pitchMatrix = arrayOf(
            floatArrayOf(cp, 0f, sp),
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(-sp, 0f, cp)
        )
        val yawMatrix = arrayOf(
            floatArrayOf(cy, -sy, 0f),
            floatArrayOf(sy, cy, 0f),
            floatArrayOf(0f, 0f, 1f)
        )
        return multiply3x3(yawMatrix, multiply3x3(pitchMatrix, rollMatrix))
    }

    private fun multiply3x3(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
        val result = Array(3) { FloatArray(3) }
        for (row in 0..2) {
            for (col in 0..2) {
                result[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            }
        }
        return result
    }

    private fun invert3x3(m: Array<FloatArray>): Array<FloatArray>? {
        val det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        if (!det.isFinite() || kotlin.math.abs(det) < 1e-6f) return null
        val invDet = 1f / det
        return arrayOf(
            floatArrayOf(
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet
            ),
            floatArrayOf(
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet
            ),
            floatArrayOf(
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet
            )
        )
    }

    private fun identity3x3(): Array<FloatArray> {
        return arrayOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(0f, 0f, 1f)
        )
    }

    private fun centerCropScale(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val sourceRatio = bitmap.width.toFloat() / bitmap.height
        val targetRatio = targetWidth.toFloat() / targetHeight
        val cropWidth: Int
        val cropHeight: Int
        if (sourceRatio > targetRatio) {
            cropHeight = bitmap.height
            cropWidth = (cropHeight * targetRatio).toInt()
        } else {
            cropWidth = bitmap.width
            cropHeight = (cropWidth / targetRatio).toInt()
        }
        val cropX = ((bitmap.width - cropWidth) / 2).coerceAtLeast(0)
        val cropY = ((bitmap.height - cropHeight) / 2).coerceAtLeast(0)
        val cropped = Bitmap.createBitmap(bitmap, cropX, cropY, cropWidth, cropHeight)
        val scaled = Bitmap.createScaledBitmap(cropped, targetWidth, targetHeight, false)
        cropped.recycle()
        return scaled
    }

    fun release() {
        visionSession?.close()
        policySession?.close()
        ortEnvironment?.close()
    }
}
