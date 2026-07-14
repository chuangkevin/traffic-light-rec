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
    val rollDeg: Float = 0f,
    val speedKmh: Float = 0f,
    val movingFastEnough: Boolean = false
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
        private const val USE_NNAPI = false
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
            val visionBytes = context.assets.open(VISION_MODEL).readBytes()
            val policyBytes = context.assets.open(POLICY_MODEL).readBytes()

            // 優先 NNAPI(SD 8+ Gen1 硬體加速),失敗退回多執行緒 CPU
            visionSession = createSessionPreferNnapi(env, visionBytes, "vision")
            policySession = createSessionPreferNnapi(env, policyBytes, "policy")
            pipeline = DrivingPipeline(OrtModelRunner())
            Log.d("InferenceEngine", "openpilot models ready (core pipeline)")
            true
        } catch (e: Exception) {
            Log.e("InferenceEngine", "openpilot model init failed: ${e.message}", e)
            false
        }
    }

    private fun createSessionPreferNnapi(env: OrtEnvironment, bytes: ByteArray, tag: String): OrtSession {
        // 實測:NNAPI 對 openpilot 模型反而更慢(436ms/幀 vs CPU ~100ms)——
        // 模型含 NNAPI 不支援的算子,圖被切碎跨界搬運。固定走多執行緒 CPU。
        if (USE_NNAPI) {
            try {
                val opts = OrtSession.SessionOptions()
                opts.addNnapi()
                val s = env.createSession(bytes, opts)
                Log.i("InferenceEngine", "$tag: NNAPI enabled")
                return s
            } catch (e: Exception) {
                Log.w("InferenceEngine", "$tag: NNAPI unavailable (${e.message}), CPU fallback")
            }
        }
        try {
            val opts = OrtSession.SessionOptions()
            opts.addXnnpack(mapOf("intra_op_num_threads" to "4"))
            val s = env.createSession(bytes, opts)
            Log.i("InferenceEngine", "$tag: XNNPACK x4")
            return s
        } catch (e: Exception) {
            Log.w("InferenceEngine", "$tag: XNNPACK unavailable (${e.message})")
        }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(6)
        Log.i("InferenceEngine", "$tag: CPU x6 threads")
        return env.createSession(bytes, opts)
    }

    fun onImuTilt(tilt: Tilt, timestampMs: Long) {
        pipeline?.onImuTilt(tilt, timestampMs)
    }

    fun onSpeed(speedMps: Float) {
        pipeline?.onSpeed(speedMps)
    }

    fun forceCalibrate() {
        pipeline?.forceCalibrate()
    }

    private var inferCount = 0
    private var inferTotalMs = 0L

    suspend fun analyzeDrivingPlan(
        frame: IntImage,
        rotationDegrees: Int,
        horizontalFovDeg: Float
    ): DrivingPlanResult = withContext(Dispatchers.Default) {
        val p = pipeline ?: return@withContext DrivingPlanResult(
            false, false, 0f, 0f, 0f, 0f, DrivingAction.HOLD)
        try {
            val t0 = android.os.SystemClock.elapsedRealtime()
            val result = p.processFrame(frame, rotationDegrees, horizontalFovDeg)
            val dt = android.os.SystemClock.elapsedRealtime() - t0
            inferTotalMs += dt
            if (++inferCount % 50 == 0) {
                Log.i("InferenceEngine", "pipeline avg ${inferTotalMs / 50}ms/frame (last=${dt}ms)")
                inferTotalMs = 0
            }
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
                    cal.pitchDeg, cal.yawDeg, cal.heightM, cal.valid, cal.sampleCount, cal.rollDeg,
                    cal.speedMps * 3.6f, cal.movingFastEnough)
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
