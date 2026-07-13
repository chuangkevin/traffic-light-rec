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

/**
 * ONNX session 的 Android 殼;推理與前處理邏輯全部在 :core 的 DrivingPipeline。
 */
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
            try {
                vision.run(mapOf("img" to img, "big_img" to bigImg)).use { out ->
                    return readFloats(out.get(0) as OnnxTensor)
                }
            } finally {
                img.close()
                bigImg.close()
            }
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
            try {
                policy.run(mapOf(
                    "desire_pulse" to desireT,
                    "traffic_convention" to tcT,
                    "features_buffer" to featT
                )).use { out ->
                    return readFloats(out.get(0) as OnnxTensor)
                }
            } finally {
                desireT.close()
                tcT.close()
                featT.close()
            }
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
            val result = p.processFrame(
                IntImage(bitmap.width, bitmap.height, pixels), rotationDegrees, horizontalFovDeg)
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
