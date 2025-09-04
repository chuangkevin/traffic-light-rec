package com.example.trafficlight.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.trafficlight.R
import com.example.trafficlight.inference.DetectionResult

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<DetectionResult> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var imageRotation: Int = 0

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

    fun setResults(
        detections: List<DetectionResult>,
        imageWidth: Int,
        imageHeight: Int,
        imageRotation: Int
    ) {
        this.detections = detections

        if (this.imageWidth != imageWidth || this.imageHeight != imageHeight || this.imageRotation != imageRotation || this.width != 0 || this.height != 0) {
            this.imageWidth = imageWidth
            this.imageHeight = imageHeight
            this.imageRotation = imageRotation
            updateTransformationMatrix()
        }
        invalidate()
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
