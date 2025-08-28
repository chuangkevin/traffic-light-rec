package com.example.trafficlight.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.trafficlight.R
import com.example.trafficlight.logic.TrafficLightState
import com.example.trafficlight.inference.DetectionResult

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var currentState: TrafficLightState = TrafficLightState.UNKNOWN
    private var confidence: Float = 0f
    private var currentRoi: RectF? = null
    private var debugInfo: String = ""
    private var allDetections: List<DetectionResult> = emptyList()
    
    private val statePaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val roiPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
        color = ContextCompat.getColor(context, android.R.color.white)
    }
    
    private val detectionPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
        color = ContextCompat.getColor(context, android.R.color.holo_green_light)
    }
    
    private val labelPaint = Paint().apply {
        textSize = 32f
        isAntiAlias = true
        color = ContextCompat.getColor(context, android.R.color.white)
        style = Paint.Style.FILL
    }
    
    private val labelBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
        color = ContextCompat.getColor(context, android.R.color.black)
        alpha = 180
    }
    
    private val textPaint = Paint().apply {
        textSize = 48f
        isAntiAlias = true
        color = ContextCompat.getColor(context, android.R.color.white)
        textAlign = Paint.Align.CENTER
    }
    
    private val indicatorHeight = 80f
    private val indicatorMargin = 40f
    private val cornerRadius = 20f
    
    fun updateState(state: TrafficLightState, confidence: Float) {
        this.currentState = state
        this.confidence = confidence
        invalidate()
    }
    
    fun updateDetections(detections: List<DetectionResult>) {
        android.util.Log.d("OverlayView", "收到 ${detections.size} 個檢測結果")
        for (detection in detections) {
            android.util.Log.d("OverlayView", "檢測: ${detection.label}, 信心度: ${detection.confidence}, bbox: ${detection.bbox}")
        }
        this.allDetections = detections
        invalidate()
    }
    
    fun updateDebugInfo(debugInfo: String) {
        this.debugInfo = debugInfo
        invalidate()
    }
    
    fun updateRoi(roi: RectF?) {
        this.currentRoi = roi
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // 記錄畫布尺寸以供調試
        android.util.Log.d("OverlayView", "Canvas 尺寸: ${canvas.width}x${canvas.height}, View 尺寸: ${width}x${height}")
        
        drawAllDetections(canvas)
        drawStateIndicator(canvas)
        drawRoi(canvas)
        drawDebugInfo(canvas)
    }
    
    private fun drawAllDetections(canvas: Canvas) {
        android.util.Log.d("OverlayView", "drawAllDetections: 繪製 ${allDetections.size} 個檢測結果，canvas 尺寸: ${width}x${height}")
        for ((index, detection) in allDetections.withIndex()) {
            android.util.Log.d("OverlayView", "繪製檢測 $index: ${detection.label}, bbox=${detection.bbox}")
            // 設定框線顏色 - 交通燈用特殊顏色標示
            val boxColor = if (detection.classId == 9) { // traffic light
                ContextCompat.getColor(context, android.R.color.holo_red_light)
            } else {
                ContextCompat.getColor(context, android.R.color.holo_green_light)
            }
            
            detectionPaint.color = boxColor
            detectionPaint.alpha = (255 * 0.8f).toInt()
            
            // 畫邊界框
            canvas.drawRect(detection.bbox, detectionPaint)
            
            // 準備標籤文字
            val label = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val textBounds = android.graphics.Rect()
            labelPaint.getTextBounds(label, 0, label.length, textBounds)
            
            // 計算標籤位置
            val labelX = detection.bbox.left
            val labelY = detection.bbox.top - 5f
            val labelWidth = textBounds.width() + 16f
            val labelHeight = textBounds.height() + 8f
            
            // 確保標籤在螢幕範圍內
            val adjustedY = if (labelY - labelHeight < 0) detection.bbox.top + labelHeight + 5f else labelY
            
            // 畫標籤背景
            val labelRect = RectF(
                labelX,
                adjustedY - labelHeight,
                labelX + labelWidth,
                adjustedY
            )
            canvas.drawRect(labelRect, labelBackgroundPaint)
            
            // 畫標籤文字
            canvas.drawText(
                label,
                labelX + 8f,
                adjustedY - 4f,
                labelPaint
            )
        }
    }

    private fun drawStateIndicator(canvas: Canvas) {
        val indicatorRect = RectF(
            indicatorMargin,
            indicatorMargin,
            width - indicatorMargin,
            indicatorMargin + indicatorHeight
        )
        
        val color = when (currentState) {
            TrafficLightState.RED -> ContextCompat.getColor(context, R.color.red_light)
            TrafficLightState.YELLOW -> ContextCompat.getColor(context, R.color.yellow_light)
            TrafficLightState.GREEN -> ContextCompat.getColor(context, R.color.green_light)
            TrafficLightState.OFF -> ContextCompat.getColor(context, R.color.light_off)
            TrafficLightState.UNKNOWN -> ContextCompat.getColor(context, android.R.color.darker_gray)
        }
        
        val alpha = when (currentState) {
            TrafficLightState.UNKNOWN, TrafficLightState.OFF -> 128
            else -> (255 * (0.3f + 0.7f * confidence)).toInt().coerceIn(77, 255)
        }
        
        statePaint.color = color
        statePaint.alpha = alpha
        
        canvas.drawRoundRect(indicatorRect, cornerRadius, cornerRadius, statePaint)
        
        val stateText = when (currentState) {
            TrafficLightState.RED -> "紅燈"
            TrafficLightState.YELLOW -> "黃燈"
            TrafficLightState.GREEN -> "綠燈"
            TrafficLightState.OFF -> "關閉"
            TrafficLightState.UNKNOWN -> "偵測中"
        }
        
        val textY = indicatorRect.centerY() + textPaint.textSize / 3
        canvas.drawText(stateText, indicatorRect.centerX(), textY, textPaint)
        
        if (confidence > 0) {
            val confidenceText = "${(confidence * 100).toInt()}%"
            val confidenceTextPaint = Paint(textPaint).apply {
                textSize = 28f
            }
            confidenceTextPaint.alpha = 180
            val confidenceY = indicatorRect.bottom + 35f
            canvas.drawText(confidenceText, indicatorRect.centerX(), confidenceY, confidenceTextPaint)
        }
    }
    
    private fun drawRoi(canvas: Canvas) {
        val roi = currentRoi ?: return
        
        val roiColor = when (currentState) {
            TrafficLightState.RED -> ContextCompat.getColor(context, R.color.red_light)
            TrafficLightState.YELLOW -> ContextCompat.getColor(context, R.color.yellow_light)  
            TrafficLightState.GREEN -> ContextCompat.getColor(context, R.color.green_light)
            else -> ContextCompat.getColor(context, android.R.color.white)
        }
        
        roiPaint.color = roiColor
        roiPaint.alpha = (255 * (0.5f + 0.5f * confidence)).toInt().coerceIn(128, 255)
        
        canvas.drawRect(roi, roiPaint)
        
        val cornerLength = 30f
        val cornerPaint = Paint(roiPaint).apply {
            strokeWidth = 8f
        }
        
        canvas.drawLine(roi.left, roi.top, roi.left + cornerLength, roi.top, cornerPaint)
        canvas.drawLine(roi.left, roi.top, roi.left, roi.top + cornerLength, cornerPaint)
        
        canvas.drawLine(roi.right - cornerLength, roi.top, roi.right, roi.top, cornerPaint)
        canvas.drawLine(roi.right, roi.top, roi.right, roi.top + cornerLength, cornerPaint)
        
        canvas.drawLine(roi.left, roi.bottom - cornerLength, roi.left, roi.bottom, cornerPaint)
        canvas.drawLine(roi.left, roi.bottom, roi.left + cornerLength, roi.bottom, cornerPaint)
        
        canvas.drawLine(roi.right, roi.bottom - cornerLength, roi.right, roi.bottom, cornerPaint)
        canvas.drawLine(roi.right - cornerLength, roi.bottom, roi.right, roi.bottom, cornerPaint)
    }
    
    fun animateStateChange(newState: TrafficLightState) {
        if (newState != currentState) {
            val animator = android.animation.ValueAnimator.ofFloat(0f, 1f)
            animator.duration = 300
            animator.addUpdateListener { animation ->
                val progress = animation.animatedValue as Float
                alpha = 0.5f + 0.5f * progress
                invalidate()
            }
            animator.start()
        }
    }
    
    fun showDetectionPulse() {
        val pulseAnimator = android.animation.ValueAnimator.ofFloat(1f, 1.2f, 1f)
        pulseAnimator.duration = 200
        pulseAnimator.addUpdateListener { animation ->
            val scale = animation.animatedValue as Float
            scaleX = scale
            scaleY = scale
        }
        pulseAnimator.start()
    }
    
    private fun drawDebugInfo(canvas: Canvas) {
        if (debugInfo.isNotEmpty()) {
            val debugPaint = Paint(textPaint).apply {
                textSize = 24f
                color = ContextCompat.getColor(context, android.R.color.holo_orange_light)
            }
            
            // 在螢幕右上角顯示 debug 信息
            val debugX = width - 20f
            val debugY = 60f
            
            canvas.drawText(debugInfo, debugX, debugY, debugPaint)
        }
    }
}