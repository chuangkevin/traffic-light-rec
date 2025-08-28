package com.example.trafficlight.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

data class DetectionResult(
    val bbox: RectF,
    val confidence: Float,
    val classId: Int,
    val label: String = getClassLabel(classId)
) {
    companion object {
        fun getClassLabel(classId: Int): String {
            // COCO dataset class labels (commonly used in YOLO models)
            val labels = arrayOf(
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
            )
            return if (classId in 0 until labels.size) labels[classId] else "unknown_$classId"
        }
    }
}

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
        private const val DETECTOR_MODEL = "models/detector.onnx"
        private const val CLASSIFIER_MODEL = "models/classifier.onnx"
        private const val CONFIDENCE_THRESHOLD = 0.05f  // 更低的閾值以顯示更多檢測結果
        private const val IOU_THRESHOLD = 0.45f
    }
    
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d("InferenceEngine", "開始初始化 AI 模型...")
            ortEnvironment = OrtEnvironment.getEnvironment()
            Log.d("InferenceEngine", "ONNX 環境建立成功")
            
            Log.d("InferenceEngine", "載入檢測模型: $DETECTOR_MODEL")
            val detectorModelBytes = context.assets.open(DETECTOR_MODEL).readBytes()
            Log.d("InferenceEngine", "檢測模型載入成功，大小: ${detectorModelBytes.size} bytes")
            
            Log.d("InferenceEngine", "載入分類模型: $CLASSIFIER_MODEL")
            val classifierModelBytes = context.assets.open(CLASSIFIER_MODEL).readBytes()
            Log.d("InferenceEngine", "分類模型載入成功，大小: ${classifierModelBytes.size} bytes")
            
            detectorSession = ortEnvironment?.createSession(detectorModelBytes)
            classifierSession = ortEnvironment?.createSession(classifierModelBytes)
            
            Log.d("InferenceEngine", "AI 模型初始化完成")
            true
        } catch (e: Exception) {
            Log.e("InferenceEngine", "AI 模型載入失敗: ${e.message}", e)
            e.printStackTrace()
            false
        }
    }
    
    suspend fun detectTrafficLights(bitmap: Bitmap): List<DetectionResult> = withContext(Dispatchers.Default) {
        val session = detectorSession ?: return@withContext emptyList()
        val env = ortEnvironment ?: return@withContext emptyList()
        
        try {
            Log.d("InferenceEngine", "開始交通燈檢測，原始尺寸: ${bitmap.width}x${bitmap.height}")
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, detectorInputSize, detectorInputSize, false)
            val inputTensor = createDetectorInputTensor(env, resizedBitmap)
            
            val outputs = session.run(mapOf("images" to inputTensor))
            val outputTensor = outputs.get(0) as OnnxTensor
            
            val detections = parseDetectorOutput(outputTensor, bitmap.width, bitmap.height)
            Log.d("InferenceEngine", "檢測到 ${detections.size} 個物件")
            
            inputTensor.close()
            outputTensor.close()
            outputs.close()
            
            detections
        } catch (e: Exception) {
            Log.e("InferenceEngine", "物件檢測失敗: ${e.message}", e)
            e.printStackTrace()
            emptyList()
        }
    }

    // 專門用於取得交通燈檢測結果的函數
    suspend fun detectTrafficLightsOnly(bitmap: Bitmap): List<DetectionResult> = withContext(Dispatchers.Default) {
        val allDetections = detectTrafficLights(bitmap)
        // 過濾出交通燈 (classId = 9 in COCO dataset)
        return@withContext allDetections.filter { it.classId == 9 }
    }
    
    suspend fun classifyTrafficLight(bitmap: Bitmap, roi: RectF): ClassificationResult = withContext(Dispatchers.Default) {
        val session = classifierSession ?: return@withContext ClassificationResult(
            ClassificationResult.UNKNOWN, 0f, floatArrayOf()
        )
        val env = ortEnvironment ?: return@withContext ClassificationResult(
            ClassificationResult.UNKNOWN, 0f, floatArrayOf()
        )
        
        try {
            Log.d("InferenceEngine", "開始交通燈分類，ROI: ${roi}")
            val roiBitmap = cropRoi(bitmap, roi)
            val resizedBitmap = Bitmap.createScaledBitmap(roiBitmap, classifierInputSize, classifierInputSize, false)
            val inputTensor = createClassifierInputTensor(env, resizedBitmap)
            
            val outputs = session.run(mapOf("input" to inputTensor))
            val outputTensor = outputs.get(0) as OnnxTensor
            
            val result = parseClassifierOutput(outputTensor)
            Log.d("InferenceEngine", "分類結果: 類別=${result.classId}, 信心度=${result.confidence}")
            
            inputTensor.close()
            outputTensor.close()
            outputs.close()
            
            result
        } catch (e: Exception) {
            Log.e("InferenceEngine", "交通燈分類失敗: ${e.message}", e)
            e.printStackTrace()
            ClassificationResult(ClassificationResult.UNKNOWN, 0f, floatArrayOf())
        }
    }
    
    private fun createDetectorInputTensor(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val floatBuffer = FloatBuffer.allocate(3 * bitmap.width * bitmap.height)
        
        // YOLOv8 期望的格式: CHW (Channel-Height-Width), RGB順序, 標準化到[0,1]
        // R channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        }
        // G channel  
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        }
        // B channel
        for (pixel in pixels) {
            floatBuffer.put((pixel and 0xFF) / 255.0f)
        }
        
        floatBuffer.rewind() // 重設 buffer 位置到開頭
        Log.d("InferenceEngine", "檢測器輸入張量形狀: [1, 3, ${bitmap.height}, ${bitmap.width}]")
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong()))
    }
    
    private fun createClassifierInputTensor(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val floatBuffer = FloatBuffer.allocate(3 * bitmap.width * bitmap.height)
        
        // MobileNetV3 期望的格式: CHW (Channel-Height-Width), RGB順序, 標準化到[0,1]
        // R channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        }
        // G channel  
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        }
        // B channel
        for (pixel in pixels) {
            floatBuffer.put((pixel and 0xFF) / 255.0f)
        }
        
        floatBuffer.rewind() // 重設 buffer 位置到開頭
        Log.d("InferenceEngine", "分類器輸入張量形狀: [1, 3, ${bitmap.height}, ${bitmap.width}]")
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong()))
    }
    
    private fun parseDetectorOutput(outputTensor: OnnxTensor, originalWidth: Int, originalHeight: Int): List<DetectionResult> {
        val output = outputTensor.floatBuffer.array()
        val detections = mutableListOf<DetectionResult>()
        
        Log.d("InferenceEngine", "檢測器輸出張量大小: ${output.size}")
        
        // YOLOv8 輸出格式通常是 [1, 84, 8400] 或類似
        // 需要根據實際模型輸出調整解析方式
        val tensorShape = outputTensor.info.shape
        Log.d("InferenceEngine", "檢測器輸出形狀: ${tensorShape.contentToString()}")
        
        if (tensorShape.size >= 3) {
            val numDetections = tensorShape[2].toInt() // 通常是8400
            val numFeatures = tensorShape[1].toInt()   // 通常是84 (4 bbox + 1 conf + 80 classes)
            
            Log.d("InferenceEngine", "解析 $numDetections 個候選檢測，每個有 $numFeatures 個特徵")
            
            val scaleX = originalWidth.toFloat() / detectorInputSize
            val scaleY = originalHeight.toFloat() / detectorInputSize
            
            for (i in 0 until numDetections) {
                // YOLOv8 格式: [x_center, y_center, width, height, confidence, class_scores...]
                val x_center = output[i] * scaleX
                val y_center = output[numDetections + i] * scaleY
                val width = output[2 * numDetections + i] * scaleX
                val height = output[3 * numDetections + i] * scaleY
                val confidence = output[4 * numDetections + i]
                
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    // 找到最高分數的類別（從第5個特徵開始是類別分數）
                    var maxClassScore = 0f
                    var bestClassId = 0
                    for (c in 0 until (numFeatures - 5)) {
                        val classScore = output[(5 + c) * numDetections + i]
                        if (classScore > maxClassScore) {
                            maxClassScore = classScore
                            bestClassId = c
                        }
                    }
                    
                    // 總信心度 = 物件信心度 * 類別信心度
                    val totalConfidence = confidence * maxClassScore
                    
                    if (totalConfidence >= CONFIDENCE_THRESHOLD) {
                        val x1 = x_center - width / 2
                        val y1 = y_center - height / 2
                        val x2 = x_center + width / 2
                        val y2 = y_center + height / 2
                        
                        detections.add(DetectionResult(
                            RectF(x1, y1, x2, y2),
                            totalConfidence,
                            bestClassId,
                            DetectionResult.getClassLabel(bestClassId)
                        ))
                        
                        Log.d("InferenceEngine", "檢測到物件: ${DetectionResult.getClassLabel(bestClassId)}, 信心度=$totalConfidence, bbox=($x1,$y1,$x2,$y2)")
                    }
                }
            }
        } else {
            Log.e("InferenceEngine", "未知的檢測器輸出格式")
        }
        
        Log.d("InferenceEngine", "初步檢測到 ${detections.size} 個候選物件")
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