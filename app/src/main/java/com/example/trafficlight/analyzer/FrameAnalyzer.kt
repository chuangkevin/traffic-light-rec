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

    // viewWidth and viewHeight are no longer needed here for transformation,
    // but might be used by RoiSelector, so we keep them for now.
    private var viewWidth: Int = 0
    private var viewHeight: Int = 0

    fun setViewDimensions(width: Int, height: Int) {
        if (width > 0 && height > 0) {
            this.viewWidth = width
            this.viewHeight = height
            onDebugCallback("View 尺寸更新: ${width}x${height}")
        }
    }

    // AnalysisResult now includes all necessary data for the UI layer to draw correctly.
    data class AnalysisResult(
        val detections: List<com.example.trafficlight.inference.DetectionResult>,
        val imageWidth: Int,
        val imageHeight: Int,
        val imageRotation: Int,
        val currentState: String,
        val confidence: Float,
        val fps: Int,
        val roiInfo: String,
        val debugInfo: String
    )

    override fun analyze(image: ImageProxy) {
        frameCounter++
        fpsCounter.tick()
        
        val currentTime = System.currentTimeMillis()
        val shouldRunDetection = frameCounter % detectionInterval == 0
        val shouldRunClassification = frameCounter % classificationInterval == 0
        
        analysisScope.launch {
            try {
                val originalBitmap = convertImageProxyToBitmap(image)
                // Create a mutable copy to avoid issues with recycled bitmaps from the camera proxy
                val bitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                originalBitmap.recycle()

                val imageRotation = image.imageInfo.rotationDegrees
                
                if (shouldRunDetection) {
                    runDetection(bitmap, currentTime)
                }
                
                if (shouldRunClassification) {
                    runClassification(bitmap, currentTime)
                }
                
                val result = createAnalysisResult(bitmap.width, bitmap.height, imageRotation)
                withContext(Dispatchers.Main) {
                    onResultCallback(result)
                }
                
            } catch (t: Throwable) {
                onDebugCallback("❌ 分析時發生嚴重錯誤: ${t.message}")
                if (t is OutOfMemoryError) {
                    onDebugCallback("!! 記憶體不足，請檢查影片解析度是否過高 !!")
                }
                t.printStackTrace()
            } finally {
                image.close()
            }
        }
    }
    
    private suspend fun runDetection(bitmap: Bitmap, currentTime: Long) {
        // This function now only performs detection and does NOT transform coordinates.
        allDetections = inferenceEngine.detectTrafficLights(bitmap)
        onDebugCallback("檢測到 ${allDetections.size} 個物件")
        
        // Note: RoiSelector might need adjustment if it relies on transformed coordinates.
        // For now, we pass the raw detections.
        val trafficLightDetections = allDetections.filter { it.classId == 9 }
        roiSelector.selectBestRoi(trafficLightDetections, viewWidth, viewHeight)
        
        lastDetectionTime = currentTime
    }
    
    private suspend fun runClassification(bitmap: Bitmap, currentTime: Long) {
        val currentRoi = roiSelector.getCurrentRoi()
        
        if (currentRoi != null && roiSelector.isRoiStable()) {
            val expandedRoi = roiSelector.expandRoi(currentRoi, 1.1f)
            val clampedRoi = roiSelector.cropRoiToImageBounds(expandedRoi, bitmap.width, bitmap.height)
            
            val classificationResult = inferenceEngine.classifyTrafficLight(bitmap, clampedRoi)
            stateMachine.processClassification(classificationResult)
        }
        
        lastClassificationTime = currentTime
    }
    
    private fun convertImageProxyToBitmap(image: ImageProxy): Bitmap {
        // This is a more direct and efficient method to convert YUV_420_888 to a Bitmap.
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, yuvImage.width, yuvImage.height), 90, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
    
    private fun createAnalysisResult(imageWidth: Int, imageHeight: Int, imageRotation: Int): AnalysisResult {
        val currentState = stateMachine.getCurrentStateString()
        val confidence = stateMachine.getStateConfidence()
        val fps = fpsCounter.getFps()
        val roiInfo = createRoiInfo()
        val debugInfo = createDebugInfo(imageRotation)
        
        return AnalysisResult(
            detections = allDetections,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            imageRotation = imageRotation,
            currentState = currentState,
            confidence = confidence,
            fps = fps,
            roiInfo = roiInfo,
            debugInfo = debugInfo
        )
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
    
    private fun createDebugInfo(rotation: Int): String {
        val currentTime = System.currentTimeMillis()
        val detectionAge = if (lastDetectionTime > 0) (currentTime - lastDetectionTime) else -1
        val classificationAge = if (lastClassificationTime > 0) (currentTime - lastClassificationTime) else -1
        val votingInfo = stateMachine.getVotingWindowInfo()
        
        return "Rot:${rotation} Det:${detectionAge}ms Cls:${classificationAge}ms Votes:$votingInfo"
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
