package com.example.trafficlight.logic

import android.graphics.RectF
import com.example.trafficlight.inference.DetectionResult

class RoiSelector {
    private var lastSelectedRoi: RectF? = null
    private var roiStabilityCounter = 0
    private val stabilityThreshold = 3
    private val roiSmoothingFactor = 0.7f
    
    private val minRoiSize = 32f
    private val maxRoiSize = 300f
    private val aspectRatioTolerance = 0.5f
    
    fun selectBestRoi(detections: List<DetectionResult>, imageWidth: Int, imageHeight: Int): RectF? {
        if (detections.isEmpty()) {
            roiStabilityCounter = 0
            return lastSelectedRoi
        }
        
        val filteredDetections = filterValidDetections(detections, imageWidth, imageHeight)
        if (filteredDetections.isEmpty()) {
            roiStabilityCounter = 0
            return lastSelectedRoi
        }
        
        val bestDetection = selectBestDetection(filteredDetections)
        val newRoi = bestDetection.bbox
        
        val smoothedRoi = smoothRoi(newRoi)
        updateStability(smoothedRoi)
        
        return smoothedRoi
    }
    
    private fun filterValidDetections(detections: List<DetectionResult>, imageWidth: Int, imageHeight: Int): List<DetectionResult> {
        return detections.filter { detection ->
            val bbox = detection.bbox
            val width = bbox.width()
            val height = bbox.height()
            
            val isValidSize = width >= minRoiSize && height >= minRoiSize && 
                             width <= maxRoiSize && height <= maxRoiSize
            
            val aspectRatio = width / height
            val isValidAspectRatio = aspectRatio >= (1 - aspectRatioTolerance) && 
                                   aspectRatio <= (1 + aspectRatioTolerance)
            
            val isInBounds = bbox.left >= 0 && bbox.top >= 0 && 
                           bbox.right <= imageWidth && bbox.bottom <= imageHeight
            
            isValidSize && isValidAspectRatio && isInBounds && detection.confidence > 0.3f
        }
    }
    
    private fun selectBestDetection(detections: List<DetectionResult>): DetectionResult {
        return if (lastSelectedRoi != null) {
            detections.minByOrNull { detection ->
                val distance = calculateRoiDistance(detection.bbox, lastSelectedRoi!!)
                val confidenceWeight = 1.0f - detection.confidence
                distance * 0.7f + confidenceWeight * 0.3f
            } ?: detections.maxByOrNull { it.confidence }!!
        } else {
            detections.maxByOrNull { it.confidence }!!
        }
    }
    
    private fun calculateRoiDistance(roi1: RectF, roi2: RectF): Float {
        val centerX1 = roi1.centerX()
        val centerY1 = roi1.centerY()
        val centerX2 = roi2.centerX()
        val centerY2 = roi2.centerY()
        
        val dx = centerX1 - centerX2
        val dy = centerY1 - centerY2
        
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
    
    private fun smoothRoi(newRoi: RectF): RectF {
        val lastRoi = lastSelectedRoi
        
        return if (lastRoi != null && roiStabilityCounter > 0) {
            val alpha = roiSmoothingFactor
            val smoothedLeft = alpha * lastRoi.left + (1 - alpha) * newRoi.left
            val smoothedTop = alpha * lastRoi.top + (1 - alpha) * newRoi.top
            val smoothedRight = alpha * lastRoi.right + (1 - alpha) * newRoi.right
            val smoothedBottom = alpha * lastRoi.bottom + (1 - alpha) * newRoi.bottom
            
            RectF(smoothedLeft, smoothedTop, smoothedRight, smoothedBottom)
        } else {
            newRoi
        }
    }
    
    private fun updateStability(roi: RectF) {
        val lastRoi = lastSelectedRoi
        
        if (lastRoi != null) {
            val distance = calculateRoiDistance(roi, lastRoi)
            val sizeChange = kotlin.math.abs(roi.width() - lastRoi.width()) + 
                           kotlin.math.abs(roi.height() - lastRoi.height())
            
            val isStable = distance < 30f && sizeChange < 20f
            
            if (isStable) {
                roiStabilityCounter = kotlin.math.min(roiStabilityCounter + 1, stabilityThreshold)
            } else {
                roiStabilityCounter = kotlin.math.max(roiStabilityCounter - 1, 0)
            }
        } else {
            roiStabilityCounter = 1
        }
        
        lastSelectedRoi = roi
    }
    
    fun isRoiStable(): Boolean {
        return roiStabilityCounter >= stabilityThreshold
    }
    
    fun getRoiStability(): Float {
        return roiStabilityCounter.toFloat() / stabilityThreshold
    }
    
    fun expandRoi(roi: RectF, expansionFactor: Float = 1.2f): RectF {
        val centerX = roi.centerX()
        val centerY = roi.centerY()
        val width = roi.width() * expansionFactor
        val height = roi.height() * expansionFactor
        
        return RectF(
            centerX - width / 2,
            centerY - height / 2,
            centerX + width / 2,
            centerY + height / 2
        )
    }
    
    fun cropRoiToImageBounds(roi: RectF, imageWidth: Int, imageHeight: Int): RectF {
        return RectF(
            kotlin.math.max(0f, roi.left),
            kotlin.math.max(0f, roi.top),
            kotlin.math.min(imageWidth.toFloat(), roi.right),
            kotlin.math.min(imageHeight.toFloat(), roi.bottom)
        )
    }
    
    fun reset() {
        lastSelectedRoi = null
        roiStabilityCounter = 0
    }
    
    fun getCurrentRoi(): RectF? {
        return lastSelectedRoi
    }
}