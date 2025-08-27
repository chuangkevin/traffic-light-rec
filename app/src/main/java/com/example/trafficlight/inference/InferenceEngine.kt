package com.example.trafficlight.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

data class DetectionResult(
    val bbox: RectF,
    val confidence: Float,
    val classId: Int
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

class InferenceEngine(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var detectorSession: OrtSession? = null
    private var classifierSession: OrtSession? = null
    
    private val detectorInputSize = 640
    private val classifierInputSize = 64
    
    companion object {
        private const val DETECTOR_MODEL = "detector.onnx"
        private const val CLASSIFIER_MODEL = "classifier.onnx"
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            val detectorModelBytes = context.assets.open(DETECTOR_MODEL).readBytes()
            val classifierModelBytes = context.assets.open(CLASSIFIER_MODEL).readBytes()
            
            detectorSession = ortEnvironment?.createSession(detectorModelBytes)
            classifierSession = ortEnvironment?.createSession(classifierModelBytes)
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    suspend fun detectTrafficLights(bitmap: Bitmap): List<DetectionResult> = withContext(Dispatchers.Default) {
        val session = detectorSession ?: return@withContext emptyList()
        val env = ortEnvironment ?: return@withContext emptyList()
        
        try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, detectorInputSize, detectorInputSize, false)
            val inputTensor = createDetectorInputTensor(env, resizedBitmap)
            
            val outputs = session.run(mapOf("images" to inputTensor))
            val outputTensor = outputs.get(0) as OnnxTensor
            
            val detections = parseDetectorOutput(outputTensor, bitmap.width, bitmap.height)
            
            inputTensor.close()
            outputTensor.close()
            outputs.close()
            
            detections
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }
    
    suspend fun classifyTrafficLight(bitmap: Bitmap, roi: RectF): ClassificationResult = withContext(Dispatchers.Default) {
        val session = classifierSession ?: return@withContext ClassificationResult(
            ClassificationResult.UNKNOWN, 0f, floatArrayOf()
        )
        val env = ortEnvironment ?: return@withContext ClassificationResult(
            ClassificationResult.UNKNOWN, 0f, floatArrayOf()
        )
        
        try {
            val roiBitmap = cropRoi(bitmap, roi)
            val resizedBitmap = Bitmap.createScaledBitmap(roiBitmap, classifierInputSize, classifierInputSize, false)
            val inputTensor = createClassifierInputTensor(env, resizedBitmap)
            
            val outputs = session.run(mapOf("input" to inputTensor))
            val outputTensor = outputs.get(0) as OnnxTensor
            
            val result = parseClassifierOutput(outputTensor)
            
            inputTensor.close()
            outputTensor.close()
            outputs.close()
            
            result
        } catch (e: Exception) {
            e.printStackTrace()
            ClassificationResult(ClassificationResult.UNKNOWN, 0f, floatArrayOf())
        }
    }
    
    private fun createDetectorInputTensor(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val floatBuffer = FloatBuffer.allocate(3 * bitmap.width * bitmap.height)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            floatBuffer.put(r)
            floatBuffer.put(g)
            floatBuffer.put(b)
        }
        
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong()))
    }
    
    private fun createClassifierInputTensor(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val floatBuffer = FloatBuffer.allocate(3 * bitmap.width * bitmap.height)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            floatBuffer.put(r)
            floatBuffer.put(g)
            floatBuffer.put(b)
        }
        
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong()))
    }
    
    private fun parseDetectorOutput(outputTensor: OnnxTensor, originalWidth: Int, originalHeight: Int): List<DetectionResult> {
        val output = outputTensor.floatBuffer.array()
        val detections = mutableListOf<DetectionResult>()
        
        val numDetections = output.size / 6
        val scaleX = originalWidth.toFloat() / detectorInputSize
        val scaleY = originalHeight.toFloat() / detectorInputSize
        
        for (i in 0 until numDetections) {
            val confidence = output[i * 6 + 4]
            if (confidence >= CONFIDENCE_THRESHOLD) {
                val x1 = output[i * 6] * scaleX
                val y1 = output[i * 6 + 1] * scaleY
                val x2 = output[i * 6 + 2] * scaleX
                val y2 = output[i * 6 + 3] * scaleY
                val classId = output[i * 6 + 5].toInt()
                
                detections.add(DetectionResult(
                    RectF(x1, y1, x2, y2),
                    confidence,
                    classId
                ))
            }
        }
        
        return applyNMS(detections)
    }
    
    private fun parseClassifierOutput(outputTensor: OnnxTensor): ClassificationResult {
        val probabilities = outputTensor.floatBuffer.array()
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: ClassificationResult.UNKNOWN
        val confidence = probabilities[maxIndex]
        
        return ClassificationResult(maxIndex, confidence, probabilities)
    }
    
    private fun applyNMS(detections: List<DetectionResult>): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()
        
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<DetectionResult>()
        
        for (detection in sortedDetections) {
            var shouldSelect = true
            for (selected in selectedDetections) {
                if (calculateIoU(detection.bbox, selected.bbox) > IOU_THRESHOLD) {
                    shouldSelect = false
                    break
                }
            }
            if (shouldSelect) {
                selectedDetections.add(detection)
            }
        }
        
        return selectedDetections
    }
    
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionArea = maxOf(0f, minOf(box1.right, box2.right) - maxOf(box1.left, box2.left)) *
                maxOf(0f, minOf(box1.bottom, box2.bottom) - maxOf(box1.top, box2.top))
        
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    private fun cropRoi(bitmap: Bitmap, roi: RectF): Bitmap {
        val x = maxOf(0, roi.left.toInt())
        val y = maxOf(0, roi.top.toInt())
        val width = minOf(bitmap.width - x, roi.width().toInt())
        val height = minOf(bitmap.height - y, roi.height().toInt())
        
        return if (width > 0 && height > 0) {
            Bitmap.createBitmap(bitmap, x, y, width, height)
        } else {
            Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        }
    }
    
    fun release() {
        detectorSession?.close()
        classifierSession?.close()
        ortEnvironment?.close()
    }
}