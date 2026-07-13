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

/** 桌面版 ONNX Runtime 實作,與 app 端 OrtModelRunner 行為一致。 */
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
                    policy.run(mapOf(
                        "desire_pulse" to d,
                        "traffic_convention" to tc,
                        "features_buffer" to f
                    )).use { out ->
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
