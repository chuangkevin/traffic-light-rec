package com.example.trafficlight.core.geometry

import kotlin.math.tan

object ModelFrames {
    const val MEDMODEL_FL = 910f
    const val MEDMODEL_CX = 256f
    const val MEDMODEL_CY = 47.6f
    const val SBIGMODEL_FL = 455f
    const val SBIGMODEL_CX = 256f
    const val SBIGMODEL_CY = 151.8f
    const val MODEL_WIDTH = 512
    const val MODEL_HEIGHT = 256
}

private val VIEW_FROM_DEVICE = Mat3(floatArrayOf(
    0f, 1f, 0f,
    0f, 0f, 1f,
    1f, 0f, 0f
))

/** 模型幀像素座標 → 來源影像像素座標(inverse-mapping 取樣用)。 */
fun sourceFromModelFrame(
    sourceWidth: Float,
    sourceHeight: Float,
    horizontalFovDeg: Float,
    rollRad: Float,
    pitchRad: Float,
    yawRad: Float,
    bigModelFrame: Boolean
): Mat3 {
    val fovRad = Math.toRadians(horizontalFovDeg.toDouble() / 2.0)
    val sourceFl = sourceWidth / (2f * tan(fovRad).toFloat())
    val sourceIntrinsics = Mat3.intrinsics(sourceFl, sourceWidth / 2f, sourceHeight / 2f)
    val deviceFromCalib = Mat3.rotationFromEuler(rollRad, pitchRad, yawRad)
    val cameraFromCalib = sourceIntrinsics * VIEW_FROM_DEVICE * deviceFromCalib

    val modelFl = if (bigModelFrame) ModelFrames.SBIGMODEL_FL else ModelFrames.MEDMODEL_FL
    val modelCx = if (bigModelFrame) ModelFrames.SBIGMODEL_CX else ModelFrames.MEDMODEL_CX
    val modelCy = if (bigModelFrame) ModelFrames.SBIGMODEL_CY else ModelFrames.MEDMODEL_CY
    val calibFromModel = (Mat3.intrinsics(modelFl, modelCx, modelCy) * VIEW_FROM_DEVICE).invert()
        ?: Mat3.identity()
    return cameraFromCalib * calibFromModel
}
