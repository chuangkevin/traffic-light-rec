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
        private const val DETECTOR_MODEL = "models/yolo_v5_f32.onnx"
        private const val CLASSIFIER_MODEL = "models/classifier.onnx"
        private const val CONFIDENCE_THRESHOLD = 0.2f  // 合理的閾值以避免過多噪音
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
            
            // 保持長寬比的縮放到 640x640，多餘部分填充黑色
            val resizedBitmap = resizeBitmapWithPadding(bitmap, detectorInputSize, detectorInputSize)
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
        
        // 標準 YOLOv5s 期望 RGB [0,1] 正規化
        val pixelCount = bitmap.width * bitmap.height
        
        // R channel 
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            floatBuffer.put(r)
        }
        // G channel   
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            floatBuffer.put(g)
        }
        // B channel
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            val b = (pixel and 0xFF) / 255.0f
            floatBuffer.put(b)
        }
        
        Log.d("InferenceEngine", "輸入預處理完成: ${pixelCount} 像素, RGB [0,1] 正規化")
        
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
        
        if (tensorShape.size == 5) {
            // YOLOv5 多尺度輸出格式: [1, 3, 80, 80, 85]
            val batchSize = tensorShape[0].toInt()     // 1
            val numAnchors = tensorShape[1].toInt()    // 3
            val gridH = tensorShape[2].toInt()         // 80
            val gridW = tensorShape[3].toInt()         // 80  
            val numFeatures = tensorShape[4].toInt()   // 85
            
            Log.d("InferenceEngine", "YOLOv5 多尺度格式: [$batchSize, $numAnchors, $gridH, $gridW, $numFeatures]")
            
            val totalDetections = numAnchors * gridH * gridW
            Log.d("InferenceEngine", "總檢測數量: $totalDetections, 特徵數: $numFeatures")
            
            return parseYoloV5MultiScaleOutput(output, numAnchors, gridH, gridW, numFeatures, originalWidth, originalHeight)
        } else if (tensorShape.size >= 3) {
            // 標準3維輸出格式
            val batchSize = tensorShape[0].toInt()
            val numDetections = tensorShape[1].toInt()
            val numFeatures = tensorShape[2].toInt()
            
            Log.d("InferenceEngine", "標準 YOLO 格式: [$batchSize, $numDetections, $numFeatures]")
            Log.d("InferenceEngine", "解析 $numDetections 個候選檢測，每個有 $numFeatures 個特徵")
            
            val scaleX = originalWidth.toFloat() / detectorInputSize
            val scaleY = originalHeight.toFloat() / detectorInputSize
            
            // YOLOv5 格式: [x_center, y_center, width, height, objectness, class0_score, class1_score, ...]
            // 先檢查前幾個檢測結果作為調試
            val debugSamples = minOf(3, numDetections)
            for (i in 0 until debugSamples) {
                val baseIndex = i * numFeatures
                val x_center_raw = output[baseIndex]
                val y_center_raw = output[baseIndex + 1]
                val width_raw = output[baseIndex + 2]
                val height_raw = output[baseIndex + 3]
                val objectness = output[baseIndex + 4]
                
                Log.d("InferenceEngine", "樣本 $i: 正規化座標=($x_center_raw, $y_center_raw, $width_raw, $height_raw), objectness=$objectness")
                
                // 檢查前幾個類別分數，特別關注交通燈 (class 9)
                var maxClassScore = 0f
                var bestClassId = 0
                val trafficLightScore = if (numFeatures > 14) output[baseIndex + 5 + 9] else 0f // class 9 = traffic light
                
                for (c in 0 until minOf(10, numFeatures - 5)) {
                    val classScore = output[baseIndex + 5 + c]
                    if (classScore > maxClassScore) {
                        maxClassScore = classScore
                        bestClassId = c
                    }
                }
                Log.d("InferenceEngine", "樣本 $i: 最高類別 $bestClassId (${DetectionResult.getClassLabel(bestClassId)}): $maxClassScore")
                Log.d("InferenceEngine", "樣本 $i: 交通燈分數 (class 9): $trafficLightScore")
            }
            
            // 正常處理所有檢測
            for (i in 0 until numDetections) {
                val baseIndex = i * numFeatures
                
                // YOLOv5 格式解析 - 輸出是正規化座標 (0-1)
                val x_center_norm = output[baseIndex]
                val y_center_norm = output[baseIndex + 1] 
                val width_norm = output[baseIndex + 2]
                val height_norm = output[baseIndex + 3]
                
                // 標準 YOLOv5s 輸出正規化座標 [0,1]
                val x_center_pixels = x_center_norm * detectorInputSize
                val y_center_pixels = y_center_norm * detectorInputSize
                val width_pixels = width_norm * detectorInputSize
                val height_pixels = height_norm * detectorInputSize
                
                // 縮放到原始圖片尺寸
                val x_center = x_center_pixels * scaleX
                val y_center = y_center_pixels * scaleY
                val width = width_pixels * scaleX
                val height = height_pixels * scaleY
                val objectness = output[baseIndex + 4]
                
                // 只處理 objectness 分數夠高的檢測
                if (objectness >= 0.05f) {
                    // 找到最高分數的類別（從第5個特徵開始是類別分數）
                    var maxClassScore = 0f
                    var bestClassId = 0
                    for (c in 0 until (numFeatures - 5)) {
                        val classScore = output[baseIndex + 5 + c]
                        if (classScore > maxClassScore) {
                            maxClassScore = classScore
                            bestClassId = c
                        }
                    }
                    
                    // YOLOv5 總信心度 = objectness * 最高類別分數
                    val totalConfidence = objectness * maxClassScore
                    
                    // 使用合理的閾值
                    val testThreshold = 0.25f
                    if (totalConfidence >= testThreshold) {
                        // 計算邊界框座標並確保合理範圍
                        val x1 = kotlin.math.max(0f, x_center - width / 2)
                        val y1 = kotlin.math.max(0f, y_center - height / 2)
                        val x2 = kotlin.math.min(originalWidth.toFloat(), x_center + width / 2)
                        val y2 = kotlin.math.min(originalHeight.toFloat(), y_center + height / 2)
                        
                        // 確保 x2 > x1 和 y2 > y1
                        if (x2 > x1 && y2 > y1) {
                            val label = DetectionResult.getClassLabel(bestClassId)
                            detections.add(DetectionResult(
                                RectF(x1, y1, x2, y2),
                                totalConfidence,
                                bestClassId,
                                label
                            ))
                            
                            Log.d("InferenceEngine", "檢測到物件: ${DetectionResult.getClassLabel(bestClassId)}, 信心度=$totalConfidence (obj=$objectness, cls=$maxClassScore)")
                            Log.d("InferenceEngine", "正規化座標: center=($x_center_norm,$y_center_norm) size=($width_norm,$height_norm)")
                            Log.d("InferenceEngine", "640px座標: center=($x_center_pixels,$y_center_pixels) size=($width_pixels,$height_pixels)")
                            Log.d("InferenceEngine", "原始圖片座標: center=($x_center,$y_center) size=($width,$height)")
                            Log.d("InferenceEngine", "最終 bbox: ($x1,$y1,$x2,$y2)")
                        } else {
                            Log.d("InferenceEngine", "跳過無效座標: center=($x_center,$y_center) size=($width,$height) -> bbox=($x1,$y1,$x2,$y2)")
                        }
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
    
    private fun parseYoloV5MultiScaleOutput(
        output: FloatArray,
        numAnchors: Int,
        gridH: Int, 
        gridW: Int,
        numFeatures: Int,
        originalWidth: Int,
        originalHeight: Int
    ): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()
        
        val scaleX = originalWidth.toFloat() / detectorInputSize
        val scaleY = originalHeight.toFloat() / detectorInputSize
        
        // YOLOv5 anchor boxes (針對 640x640 輸入)
        val anchors = arrayOf(
            floatArrayOf(10f, 13f, 16f, 30f, 33f, 23f),      // P3/8
            floatArrayOf(30f, 61f, 62f, 45f, 59f, 119f),     // P4/16  
            floatArrayOf(116f, 90f, 156f, 198f, 373f, 326f)  // P5/32
        )
        
        val stride = 8 // 針對 80x80 grid 的 stride
        val anchorIndex = 0 // 使用 P3/8 的 anchors
        
        for (anchor in 0 until numAnchors) {
            for (y in 0 until gridH) {
                for (x in 0 until gridW) {
                    val baseIndex = (anchor * gridH * gridW + y * gridW + x) * numFeatures
                    
                    if (baseIndex + numFeatures <= output.size) {
                        val rawX = output[baseIndex]
                        val rawY = output[baseIndex + 1] 
                        val rawW = output[baseIndex + 2]
                        val rawH = output[baseIndex + 3]
                        val objectness = sigmoid(output[baseIndex + 4])
                        
                        // 簡化的座標解碼 - 先測試這個模型是否已經正規化
                        val centerX = rawX * detectorInputSize // 直接使用原始輸出
                        val centerY = rawY * detectorInputSize
                        val width = rawW * detectorInputSize
                        val height = rawH * detectorInputSize
                        
                        if (objectness >= 0.01f) {
                            // 找到最高分數的類別
                            var maxClassScore = 0f
                            var bestClassId = 0
                            for (c in 0 until (numFeatures - 5)) {
                                val classScore = output[baseIndex + 5 + c] // 不使用 sigmoid
                                if (classScore > maxClassScore) {
                                    maxClassScore = classScore
                                    bestClassId = c
                                }
                            }
                            
                            val totalConfidence = objectness * maxClassScore
                            if (totalConfidence >= 0.01f) {
                                // 縮放到原始圖片尺寸
                                val scaledCenterX = centerX * scaleX
                                val scaledCenterY = centerY * scaleY
                                val scaledWidth = width * scaleX
                                val scaledHeight = height * scaleY
                                
                                val x1 = kotlin.math.max(0f, scaledCenterX - scaledWidth / 2)
                                val y1 = kotlin.math.max(0f, scaledCenterY - scaledHeight / 2)
                                val x2 = kotlin.math.min(originalWidth.toFloat(), scaledCenterX + scaledWidth / 2)
                                val y2 = kotlin.math.min(originalHeight.toFloat(), scaledCenterY + scaledHeight / 2)
                                
                                if (x2 > x1 && y2 > y1) {
                                    detections.add(DetectionResult(
                                        android.graphics.RectF(x1, y1, x2, y2),
                                        totalConfidence,
                                        bestClassId,
                                        DetectionResult.getClassLabel(bestClassId)
                                    ))
                                    
                                    Log.d("InferenceEngine", "=== 檢測到物件: ${DetectionResult.getClassLabel(bestClassId)} ===")
                                    Log.d("InferenceEngine", "信心度: $totalConfidence (obj=$objectness, cls=$maxClassScore)")
                                    Log.d("InferenceEngine", "Grid位置: ($x,$y), Anchor: $anchor")
                                    Log.d("InferenceEngine", "原始輸出: ($rawX,$rawY,$rawW,$rawH)")
                                    Log.d("InferenceEngine", "640px座標: center=($centerX,$centerY) size=($width,$height)")
                                    Log.d("InferenceEngine", "縮放因子: scaleX=$scaleX, scaleY=$scaleY")
                                    Log.d("InferenceEngine", "縮放後座標: center=($scaledCenterX,$scaledCenterY) size=($scaledWidth,$scaledHeight)")
                                    Log.d("InferenceEngine", "最終 bbox: ($x1,$y1,$x2,$y2)")
                                    Log.d("InferenceEngine", "=================")
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Log.d("InferenceEngine", "初步檢測到 ${detections.size} 個候選物件")
        return applyNMS(detections)
    }
    
    private fun sigmoid(x: Float): Float {
        return 1f / (1f + kotlin.math.exp(-x))
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
    
    private fun resizeBitmapWithPadding(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val scale = minOf(targetWidth.toFloat() / bitmap.width, targetHeight.toFloat() / bitmap.height)
        val scaledWidth = (bitmap.width * scale).toInt()
        val scaledHeight = (bitmap.height * scale).toInt()
        
        // 創建目標尺寸的黑色背景
        val paddedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(paddedBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)
        
        // 縮放原圖並居中放置
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, false)
        val x = (targetWidth - scaledWidth) / 2
        val y = (targetHeight - scaledHeight) / 2
        canvas.drawBitmap(scaledBitmap, x.toFloat(), y.toFloat(), null)
        
        scaledBitmap.recycle()
        
        Log.d("InferenceEngine", "縮放: ${bitmap.width}x${bitmap.height} -> ${scaledWidth}x${scaledHeight}, 填充到 ${targetWidth}x${targetHeight}")
        return paddedBitmap
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