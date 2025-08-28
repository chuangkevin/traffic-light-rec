package com.example.trafficlight.analyzer

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.trafficlight.inference.InferenceEngine
import com.example.trafficlight.logic.RoiSelector
import com.example.trafficlight.logic.StateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class FrameAnalyzer(
    private val inferenceEngine: InferenceEngine,
    private val stateMachine: StateMachine,
    private val roiSelector: RoiSelector,
    private val onResultCallback: (AnalysisResult) -> Unit,
    private val onDebugCallback: (String) -> Unit = {}
) : ImageAnalysis.Analyzer {

    private var frameCounter = 0
    private var lastDetectionTime = 0L
    private var lastClassificationTime = 0L
    private var fpsCounter = FpsCounter()
    private var allDetections = emptyList<com.example.trafficlight.inference.DetectionResult>()
    
    private val detectionInterval = 3
    private val classificationInterval = 1
    private val analysisScope = CoroutineScope(Dispatchers.Default)

    data class AnalysisResult(
        val currentState: String,
        val confidence: Float,
        val fps: Int,
        val roiInfo: String,
        val debugInfo: String,
        val allDetections: List<com.example.trafficlight.inference.DetectionResult> = emptyList()
    )

    override fun analyze(image: ImageProxy) {
        frameCounter++
        fpsCounter.tick()
        
        val currentTime = System.currentTimeMillis()
        val shouldRunDetection = frameCounter % detectionInterval == 0
        val shouldRunClassification = frameCounter % classificationInterval == 0
        
        analysisScope.launch {
            try {
                val bitmap = convertImageProxyToBitmap(image)
                onDebugCallback("Frame #$frameCounter (${bitmap.width}x${bitmap.height})")
                
                if (shouldRunDetection) {
                    onDebugCallback("🔍 執行物件檢測...")
                    runDetection(bitmap, currentTime)
                }
                
                if (shouldRunClassification) {
                    onDebugCallback("🎯 執行狀態分類...")
                    runClassification(bitmap, currentTime)
                }
                
                val result = createAnalysisResult()
                withContext(Dispatchers.Main) {
                    onResultCallback(result)
                }
                
            } catch (e: Exception) {
                onDebugCallback("❌ 分析錯誤: ${e.message}")
                e.printStackTrace()
            } finally {
                image.close()
            }
        }
    }
    
    private suspend fun runDetection(bitmap: Bitmap, currentTime: Long) {
        allDetections = inferenceEngine.detectTrafficLights(bitmap)
        onDebugCallback("檢測到 ${allDetections.size} 個物件")
        
        // 過濾出交通燈進行 ROI 選擇
        val trafficLightDetections = allDetections.filter { it.classId == 9 } // traffic light class
        onDebugCallback("其中 ${trafficLightDetections.size} 個是交通燈")
        
        val selectedRoi = roiSelector.selectBestRoi(trafficLightDetections, bitmap.width, bitmap.height)
        if (selectedRoi != null) {
            onDebugCallback("✅ 選中 ROI: ${selectedRoi.width().toInt()}x${selectedRoi.height().toInt()}")
        } else {
            onDebugCallback("⚠️ 未找到合適的 ROI")
        }
        
        lastDetectionTime = currentTime
    }
    
    private suspend fun runClassification(bitmap: Bitmap, currentTime: Long) {
        val currentRoi = roiSelector.getCurrentRoi()
        
        if (currentRoi != null && roiSelector.isRoiStable()) {
            onDebugCallback("🎯 分類 ROI: ${currentRoi.width().toInt()}x${currentRoi.height().toInt()}")
            val expandedRoi = roiSelector.expandRoi(currentRoi, 1.1f)
            val clampedRoi = roiSelector.cropRoiToImageBounds(expandedRoi, bitmap.width, bitmap.height)
            
            val classificationResult = inferenceEngine.classifyTrafficLight(bitmap, clampedRoi)
            stateMachine.processClassification(classificationResult)
            
            val stateNames = arrayOf("紅燈", "黃燈", "綠燈", "關閉", "未知")
            val stateName = stateNames.getOrNull(classificationResult.classId) ?: "未知"
            onDebugCallback("分類結果: $stateName (信心度: ${(classificationResult.confidence * 100).toInt()}%)")
        }
        
        lastClassificationTime = currentTime
    }
    
    private fun convertImageProxyToBitmap(image: ImageProxy): Bitmap {
        return when (image.format) {
            ImageFormat.YUV_420_888 -> convertYuv420ToBitmap(image)
            else -> throw UnsupportedOperationException("Unsupported image format: ${image.format}")
        }
    }
    
    private fun convertYuv420ToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        yBuffer.get(nv21, 0, ySize)
        val uvBuffer = ByteBuffer.allocate(uSize + vSize)
        
        val pixelStride = image.planes[1].pixelStride
        if (pixelStride == 1) {
            uBuffer.get(nv21, ySize, uSize)
            vBuffer.get(nv21, ySize + uSize, vSize)
        } else {
            val uvBytes = ByteArray(uSize + vSize)
            uBuffer.get(uvBytes, 0, uSize)
            vBuffer.get(uvBytes, uSize, vSize)
            
            var uvIndex = 0
            for (i in 0 until uSize + vSize step pixelStride) {
                nv21[ySize + uvIndex] = uvBytes[i]
                uvIndex++
            }
        }
        
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val imageBytes = out.toByteArray()
        
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
    
    private fun createAnalysisResult(): AnalysisResult {
        val currentState = stateMachine.getCurrentStateString()
        val confidence = stateMachine.getStateConfidence()
        val fps = fpsCounter.getFps()
        val roiInfo = createRoiInfo()
        val debugInfo = createDebugInfo()
        
        return AnalysisResult(currentState, confidence, fps, roiInfo, debugInfo, allDetections)
    }
    
    private fun createRoiInfo(): String {
        val currentRoi = roiSelector.getCurrentRoi()
        return if (currentRoi != null) {
            val stability = (roiSelector.getRoiStability() * 100).toInt()
            "ROI: ${currentRoi.width().toInt()}x${currentRoi.height().toInt()} (${stability}%)"
        } else {
            "ROI: None"
        }
    }
    
    private fun createDebugInfo(): String {
        val currentTime = System.currentTimeMillis()
        val detectionAge = if (lastDetectionTime > 0) (currentTime - lastDetectionTime) else -1
        val classificationAge = if (lastClassificationTime > 0) (currentTime - lastClassificationTime) else -1
        val votingInfo = stateMachine.getVotingWindowInfo()
        
        return "Det:${detectionAge}ms Cls:${classificationAge}ms Votes:$votingInfo"
    }
    
    private class FpsCounter {
        private val timestamps = mutableListOf<Long>()
        private val windowSize = 30
        
        fun tick() {
            val currentTime = System.currentTimeMillis()
            timestamps.add(currentTime)
            
            if (timestamps.size > windowSize) {
                timestamps.removeAt(0)
            }
        }
        
        fun getFps(): Int {
            if (timestamps.size < 2) return 0
            
            val timeSpan = timestamps.last() - timestamps.first()
            return if (timeSpan > 0) {
                ((timestamps.size - 1) * 1000 / timeSpan).toInt()
            } else {
                0
            }
        }
    }
}