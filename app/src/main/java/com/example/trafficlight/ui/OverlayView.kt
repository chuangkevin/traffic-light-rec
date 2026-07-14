package com.example.trafficlight.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.example.trafficlight.R
import com.example.trafficlight.core.geometry.roadToImage
import com.example.trafficlight.inference.CameraCalibrationEstimate
import com.example.trafficlight.inference.DetectionResult
import com.example.trafficlight.inference.PlanPoint

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<DetectionResult> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var imageRotation: Int = 0
    private var pathPoints: List<PlanPoint> = emptyList()
    private var smoothedPathPoints: List<PlanPoint> = emptyList()
    private var routeShouldStop: Boolean = false
    private var routeShouldGo: Boolean = false
    private var routeVerticalOffset = 0f
    private var routeWidthScale = 1f
    private var statusFrameWidth = 28f
    private var cameraHeightOffsetM = 0f
    private var cameraPitchOffsetDeg = 0f
    private var cameraFovOffsetDeg = 0f
    private var cameraLateralOffsetM = 0f
    private var dynamicFovDeg = 72f
    private var autoCalibration = CameraCalibrationEstimate()

    private val prefs = context.getSharedPreferences("route_calibration", Context.MODE_PRIVATE)

    private val transformationMatrix = Matrix()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
        color = ContextCompat.getColor(context, R.color.detection_box_color)
    }

    private val textPaint = Paint().apply {
        textSize = 40f
        isAntiAlias = true
        color = ContextCompat.getColor(context, R.color.white)
    }

    private val textBackgroundPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.black)
        alpha = 160 // semi-transparent
        style = Paint.Style.FILL
    }

    private val routePaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
        color = android.graphics.Color.argb(145, 0, 255, 120)
    }

    private val routeGlowPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
        color = android.graphics.Color.argb(55, 0, 255, 120)
    }

    private val statusFramePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 18f
        isAntiAlias = true
        color = android.graphics.Color.argb(220, 0, 255, 120)
    }

    init {
        routeVerticalOffset = prefs.getFloat("verticalOffset", 0f)
        routeWidthScale = prefs.getFloat("widthScale", 1f)
        statusFrameWidth = prefs.getFloat("statusFrameWidth", 28f)
        cameraHeightOffsetM = prefs.getFloat("cameraHeightOffsetM", 0f)
        cameraPitchOffsetDeg = prefs.getFloat("cameraPitchOffsetDeg", 0f)
        cameraFovOffsetDeg = prefs.getFloat("cameraFovOffsetDeg", 0f)
        cameraLateralOffsetM = prefs.getFloat("cameraLateralOffsetM", 0f)
    }

    fun setResults(
        detections: List<DetectionResult>,
        imageWidth: Int,
        imageHeight: Int,
        imageRotation: Int,
        pathPoints: List<PlanPoint> = emptyList(),
        shouldStop: Boolean = false,
        shouldGo: Boolean = false,
        cameraCalibration: CameraCalibrationEstimate = CameraCalibrationEstimate(),
        horizontalFovDeg: Float = 72f
    ) {
        this.detections = detections
        this.dynamicFovDeg = horizontalFovDeg
        this.pathPoints = pathPoints
        this.routeShouldStop = shouldStop
        this.routeShouldGo = shouldGo
        this.autoCalibration = cameraCalibration
        this.smoothedPathPoints = smoothPath(pathPoints)

        if (this.imageWidth != imageWidth || this.imageHeight != imageHeight || this.imageRotation != imageRotation || this.width != 0 || this.height != 0) {
            this.imageWidth = imageWidth
            this.imageHeight = imageHeight
            this.imageRotation = imageRotation
            updateTransformationMatrix()
        }
        invalidate()
    }

    fun adjustRouteVertical(delta: Float) {
        routeVerticalOffset = (routeVerticalOffset + delta).coerceIn(-0.25f, 0.25f)
        saveCalibration()
        Toast.makeText(context, "線位移 ${String.format("%.2f", routeVerticalOffset)}", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustRouteWidth(delta: Float) {
        routeWidthScale = (routeWidthScale + delta).coerceIn(0.45f, 2.0f)
        saveCalibration()
        Toast.makeText(context, "線寬 ${String.format("%.2f", routeWidthScale)}", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustStatusFrameWidth(delta: Float) {
        statusFrameWidth = (statusFrameWidth + delta).coerceIn(8f, 64f)
        saveCalibration()
        Toast.makeText(context, "框線 ${statusFrameWidth.toInt()}px", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustCameraPitch(deltaDeg: Float) {
        cameraPitchOffsetDeg = (cameraPitchOffsetDeg + deltaDeg).coerceIn(-8f, 8f)
        saveCalibration()
        Toast.makeText(context, "俯仰微調 ${String.format("%.1f", cameraPitchOffsetDeg)}°", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustCameraFov(deltaDeg: Float) {
        cameraFovOffsetDeg = (cameraFovOffsetDeg + deltaDeg).coerceIn(-20f, 20f)
        saveCalibration()
        Toast.makeText(context, "FOV 微調 ${String.format("%.0f", cameraFovOffsetDeg)}°", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustCameraHeight(deltaM: Float) {
        cameraHeightOffsetM = (cameraHeightOffsetM + deltaM).coerceIn(-0.50f, 0.50f)
        saveCalibration()
        Toast.makeText(context, "高度微調 ${String.format("%.2f", cameraHeightOffsetM)}m", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun adjustCameraLateralOffset(deltaM: Float) {
        cameraLateralOffsetM = (cameraLateralOffsetM + deltaM).coerceIn(-0.60f, 0.60f)
        saveCalibration()
        Toast.makeText(context, "左右 ${String.format("%.2f", cameraLateralOffsetM)}m", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    fun resetCameraCalibration() {
        routeVerticalOffset = 0f
        routeWidthScale = 1f
        cameraHeightOffsetM = 0f
        cameraPitchOffsetDeg = 0f
        cameraFovOffsetDeg = 0f
        cameraLateralOffsetM = 0f
        saveCalibration()
        Toast.makeText(context, "校正已重設", Toast.LENGTH_SHORT).show()
        invalidate()
    }

    private fun saveCalibration() {
        prefs.edit()
            .putFloat("verticalOffset", routeVerticalOffset)
            .putFloat("widthScale", routeWidthScale)
            .putFloat("statusFrameWidth", statusFrameWidth)
            .putFloat("cameraHeightOffsetM", cameraHeightOffsetM)
            .putFloat("cameraPitchOffsetDeg", cameraPitchOffsetDeg)
            .putFloat("cameraFovOffsetDeg", cameraFovOffsetDeg)
            .putFloat("cameraLateralOffsetM", cameraLateralOffsetM)
            .apply()
    }

    // TODO: The coordinate transformation logic is still incorrect and needs debugging.
    private fun updateTransformationMatrix() {
        val matrix = Matrix()
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        if (viewWidth == 0f || viewHeight == 0f || imageWidth == 0 || imageHeight == 0) {
            transformationMatrix.reset()
            return
        }

        val imgWidth = this.imageWidth.toFloat()
        val imgHeight = this.imageHeight.toFloat()

        val viewRect = RectF(0f, 0f, viewWidth, viewHeight)
        val bufferRect = RectF(0f, 0f, imgWidth, imgHeight)
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()

        // Configure the matrix to scale the buffer rectangle to fit the view rectangle.
        matrix.setRectToRect(bufferRect, viewRect, Matrix.ScaleToFit.CENTER)

        // Apply the rotation around the center of the view.
        matrix.postRotate(imageRotation.toFloat(), centerX, centerY)

        transformationMatrix.set(matrix)
    }


    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        drawRoute(canvas)
        drawStatusFrame(canvas)

        for (detection in detections) {
            val originalBox = detection.bbox
            val transformedBox = RectF()

            transformationMatrix.mapRect(transformedBox, originalBox)

            // Draw the bounding box
            canvas.drawRect(transformedBox, boxPaint)

            // Draw the label
            drawLabel(canvas, transformedBox, detection)
        }
    }

    private fun drawStatusFrame(canvas: Canvas) {
        if (!routeShouldStop && !routeShouldGo) return
        statusFramePaint.color = if (routeShouldStop) android.graphics.Color.argb(230, 255, 35, 25)
        else android.graphics.Color.argb(220, 0, 255, 120)
        statusFramePaint.strokeWidth = statusFrameWidth
        val inset = statusFramePaint.strokeWidth / 2f
        canvas.drawRoundRect(
            inset,
            inset,
            width - inset,
            height - inset,
            22f,
            22f,
            statusFramePaint
        )
    }

    private fun drawRoute(canvas: Canvas) {
        if (smoothedPathPoints.size < 2 || width == 0 || height == 0) return

        val projected = smoothedPathPoints.mapNotNull { point ->
            projectPlanPoint(point)?.let { ProjectedPlanPoint(it, point.x) }
        }
        if (projected.size < 2) return

        val left = mutableListOf<PointF>()
        val right = mutableListOf<PointF>()
        for (projection in projected) {
            val point = projection.screenPoint
            val forwardMeters = projection.forwardMeters.coerceAtLeast(0f)
            val halfWidth = routeHalfWidth(forwardMeters)
            left.add(PointF(point.x - halfWidth, point.y))
            right.add(PointF(point.x + halfWidth, point.y))
        }

        val routePath = Path()
        routePath.moveTo(left.first().x, left.first().y)
        for (i in 1 until left.size) routePath.lineTo(left[i].x, left[i].y)
        for (i in right.indices.reversed()) routePath.lineTo(right[i].x, right[i].y)
        routePath.close()

        val glowPath = Path()
        val glowScale = 1.18f
        glowPath.moveTo(projected.first().screenPoint.x - routeHalfWidth(projected.first().forwardMeters) * glowScale, projected.first().screenPoint.y)
        for (projection in projected) {
            glowPath.lineTo(projection.screenPoint.x - routeHalfWidth(projection.forwardMeters) * glowScale, projection.screenPoint.y)
        }
        for (projection in projected.asReversed()) {
            glowPath.lineTo(projection.screenPoint.x + routeHalfWidth(projection.forwardMeters) * glowScale, projection.screenPoint.y)
        }
        glowPath.close()

        when {
            routeShouldStop -> {
                routeGlowPaint.color = android.graphics.Color.argb(90, 255, 210, 0)
                routePaint.color = android.graphics.Color.argb(175, 255, 35, 25)
            }
            routeShouldGo -> {
                routeGlowPaint.color = android.graphics.Color.argb(55, 0, 255, 120)
                routePaint.color = android.graphics.Color.argb(145, 0, 255, 120)
            }
            else -> {
                routeGlowPaint.color = android.graphics.Color.argb(35, 180, 190, 200)
                routePaint.color = android.graphics.Color.argb(85, 180, 190, 200)
            }
        }

        canvas.drawPath(glowPath, routeGlowPaint)
        canvas.drawPath(routePath, routePaint)
    }

    private fun smoothPath(newPath: List<PlanPoint>): List<PlanPoint> {
        if (newPath.isEmpty()) return smoothedPathPoints
        if (smoothedPathPoints.size != newPath.size) return newPath

        val alpha = 0.28f
        return newPath.indices.map { i ->
            val old = smoothedPathPoints[i]
            val next = newPath[i]
            PlanPoint(
                x = old.x + (next.x - old.x) * alpha,
                y = old.y + (next.y - old.y) * alpha
            )
        }
    }

    private fun projectPlanPoint(point: PlanPoint): PointF? {
        if (imageWidth <= 1 || imageHeight <= 1 || width == 0 || height == 0) return null
        // 與 core pipeline 共用同一套相機模型:roll/pitch/yaw 來自 IMU+模型融合校正
        val rollRad = Math.toRadians(autoCalibration.rollDeg.toDouble()).toFloat()
        val pitchRad = Math.toRadians(
            (autoCalibration.pitchDeg + cameraPitchOffsetDeg).toDouble()).toFloat()
        val yawRad = Math.toRadians(autoCalibration.yawDeg.toDouble()).toFloat()
        val cameraHeightM = (autoCalibration.heightM + cameraHeightOffsetM).coerceIn(0.70f, 2.20f)

        val img = roadToImage(
            forwardM = point.x.coerceIn(0.5f, 90f),
            lateralLeftM = point.y + cameraLateralOffsetM,
            heightM = cameraHeightM,
            imageWidth = imageWidth.toFloat(),
            imageHeight = imageHeight.toFloat(),
            horizontalFovDeg = dynamicFovDeg + cameraFovOffsetDeg,
            rollRad = rollRad,
            pitchRad = pitchRad,
            yawRad = yawRad
        ) ?: return null

        // 影像座標 → 螢幕座標:PreviewView 預設 FILL_CENTER(等比放大、置中裁切)
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scale = maxOf(viewW / imageWidth, viewH / imageHeight)
        val dx = (viewW - imageWidth * scale) / 2f
        val dy = (viewH - imageHeight * scale) / 2f
        val screenX = img.first * scale + dx
        val screenY = img.second * scale + dy + viewH * routeVerticalOffset
        if (screenY !in -viewH * 0.15f..viewH * 1.15f) return null
        return PointF(screenX, screenY)
    }

    private data class ProjectedPlanPoint(
        val screenPoint: PointF,
        val forwardMeters: Float
    )

    private fun routeHalfWidth(forwardMeters: Float): Float {
        val left = projectPlanPoint(PlanPoint(forwardMeters, -1.75f))
        val right = projectPlanPoint(PlanPoint(forwardMeters, 1.75f))
        return if (left != null && right != null) {
            kotlin.math.abs(right.x - left.x) * 0.5f * routeWidthScale
        } else {
            width * 0.04f * routeWidthScale
        }
    }

    private fun drawLabel(canvas: Canvas, box: RectF, detection: DetectionResult) {
        val label = "${detection.label} ${String.format("%.2f", detection.confidence)}"
        val textBounds = Rect()
        textPaint.getTextBounds(label, 0, label.length, textBounds)

        val textX = box.left
        val textY = box.top - 10

        canvas.drawRect(
            textX,
            textY - textBounds.height(),
            textX + textBounds.width() + 10,
            textY + 10,
            textBackgroundPaint
        )

        canvas.drawText(label, textX + 5, textY, textPaint)
    }
}
