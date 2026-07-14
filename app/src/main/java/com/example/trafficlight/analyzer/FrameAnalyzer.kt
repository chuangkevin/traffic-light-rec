package com.example.trafficlight.analyzer

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.trafficlight.inference.ClassificationResult
import com.example.trafficlight.inference.CameraCalibrationEstimate
import com.example.trafficlight.inference.DrivingAction
import com.example.trafficlight.inference.InferenceEngine
import com.example.trafficlight.inference.PlanPoint
import com.example.trafficlight.logic.RoiSelector
import com.example.trafficlight.logic.StateMachine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

class FrameAnalyzer(
    private val inferenceEngine: InferenceEngine,
    private val stateMachine: StateMachine,
    private val roiSelector: RoiSelector,
    private val onResultCallback: (AnalysisResult) -> Unit,
    private val onDebugCallback: (String) -> Unit = {},
    private val horizontalFovProvider: () -> Float = { 72f },
    private val onRotationChanged: (Int) -> Unit = {},
    private val dumpDirProvider: () -> java.io.File? = { null }
) : ImageAnalysis.Analyzer {

    /** 設為 true 後,下一個推理幀會把畫面+校正值+路徑存到 dumpDir(遠端偵錯用)。 */
    @Volatile
    var dumpRequested = false

    private var frameCounter = 0
    private var lastDetectionTime = 0L
    private var lastClassificationTime = 0L
    private var fpsCounter = FpsCounter()
    private var allDetections = emptyList<com.example.trafficlight.inference.DetectionResult>()
    private var planInfo = "Plan: warming up"
    private var pathPoints = emptyList<PlanPoint>()
    private var shouldStop = false
    private var shouldGo = false
    private var cameraCalibration = CameraCalibrationEstimate()
    
    // 每個非忙碌幀都推理:openpilot 模型期望 ~50ms 間隔的連續幀對,
    // 間隔越接近訓練分布,速度/距離估計越準(舊值 4 → 間隔 >133ms)
    private val detectionInterval = 1
    private val analysisScope = CoroutineScope(Dispatchers.Default)
    private val isProcessing = AtomicBoolean(false)

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
        val horizontalFovDeg: Float,
        val currentState: String,
        val confidence: Float,
        val fps: Int,
        val roiInfo: String,
        val debugInfo: String,
        val pathPoints: List<PlanPoint>,
        val shouldStop: Boolean,
        val shouldGo: Boolean,
        val cameraCalibration: CameraCalibrationEstimate
    )

    override fun analyze(image: ImageProxy) {
        if (!isProcessing.compareAndSet(false, true)) {
            image.close()
            return
        }

        frameCounter++
        fpsCounter.tick()
        
        val currentTime = System.currentTimeMillis()
        val shouldRunDetection = frameCounter % detectionInterval == 0
        
        analysisScope.launch {
            var bitmap: Bitmap? = null
            try {
                val originalBitmap = convertImageProxyToBitmap(image)
                // Create a mutable copy to avoid issues with recycled bitmaps from the camera proxy
                bitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                originalBitmap.recycle()

                val imageRotation = image.imageInfo.rotationDegrees
                onRotationChanged(imageRotation)

                if (shouldRunDetection) {
                    runDetection(bitmap, imageRotation, currentTime)
                }

                // core pipeline 已把幀轉正,overlay 座標以轉正後的尺寸為基準
                val uprightW = if (imageRotation == 90 || imageRotation == 270) bitmap.height else bitmap.width
                val uprightH = if (imageRotation == 90 || imageRotation == 270) bitmap.width else bitmap.height
                val result = createAnalysisResult(uprightW, uprightH, 0)
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
                bitmap?.recycle()
                isProcessing.set(false)
                image.close()
            }
        }
    }
    
    private fun dumpDebugSnapshot(bitmap: Bitmap, rotationDegrees: Int, fov: Float, plan: com.example.trafficlight.inference.DrivingPlanResult) {
        try {
            val root = dumpDirProvider() ?: return
            val dir = java.io.File(root, "dump-${System.currentTimeMillis()}")
            dir.mkdirs()
            // 轉正後的幀(與 pipeline 輸入一致)
            val matrix = android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            java.io.FileOutputStream(java.io.File(dir, "frame.png")).use {
                upright.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            if (upright !== bitmap) upright.recycle()
            val cal = plan.calibration
            val meta = StringBuilder()
            meta.append("fovDeg=$fov\n")
            meta.append("rollDeg=${cal.rollDeg}\n")
            meta.append("pitchDeg=${cal.pitchDeg}\n")
            meta.append("yawDeg=${cal.yawDeg}\n")
            meta.append("heightM=${cal.heightM}\n")
            meta.append("valid=${cal.valid}\n")
            meta.append("sampleCount=${cal.sampleCount}\n")
            meta.append("action=${plan.action}\n")
            plan.path.forEach { meta.append("path=${it.x},${it.y}\n") }
            java.io.File(dir, "meta.txt").writeText(meta.toString())
            onDebugCallback("偵錯快照已存:${dir.name}")
        } catch (e: Exception) {
            onDebugCallback("偵錯快照失敗:${e.message}")
        }
    }

    private suspend fun runDetection(bitmap: Bitmap, rotationDegrees: Int, currentTime: Long) {
        allDetections = emptyList()
        val plan = inferenceEngine.analyzeDrivingPlan(bitmap, rotationDegrees, horizontalFovProvider())
        if (dumpRequested) {
            dumpRequested = false
            dumpDebugSnapshot(bitmap, rotationDegrees, horizontalFovProvider(), plan)
        }
        shouldStop = plan.shouldStop
        shouldGo = plan.shouldGo
        val classId = when (plan.action) {
            DrivingAction.STOP -> ClassificationResult.RED
            DrivingAction.GO -> ClassificationResult.GREEN
            DrivingAction.HOLD -> ClassificationResult.UNKNOWN
        }
        stateMachine.processClassification(ClassificationResult(classId, plan.confidence, floatArrayOf()))
        pathPoints = plan.path
        cameraCalibration = plan.calibration
        planInfo = "Plan: ${plan.action} near=${String.format("%.1f", plan.nearVelocity)}m/s future=${String.format("%.1f", plan.futureVelocity)}m/s " +
            "a=${String.format("%.2f", plan.desiredAcceleration)} " +
            "cal=${if (plan.calibration.valid) "on" else "warm"}/${plan.calibration.sampleCount} " +
            "warpP=${String.format("%.1f", plan.calibration.pitchDeg)} y=${String.format("%.1f", plan.calibration.yawDeg)} h=${String.format("%.2f", plan.calibration.heightM)}"
        onDebugCallback("openpilot ${plan.action} ${planInfo}")
        
        lastDetectionTime = currentTime
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
        val horizontalFov = horizontalFovProvider()
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
            horizontalFovDeg = horizontalFov,
            currentState = currentState,
            confidence = confidence,
            fps = fps,
            roiInfo = roiInfo,
            debugInfo = debugInfo,
            pathPoints = pathPoints,
            shouldStop = shouldStop,
            shouldGo = shouldGo,
            cameraCalibration = cameraCalibration
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
        
        return "Rot:${rotation} Det:${detectionAge}ms Cls:${classificationAge}ms Votes:$votingInfo $planInfo"
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
