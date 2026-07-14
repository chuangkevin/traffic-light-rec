package com.example.trafficlight.core.geometry

/**
 * 把路面(calib)座標系的點投影到「轉正後影像」像素座標。
 * 與 warp(sourceFromModelFrame)共用同一套旋轉/內參數學,
 * 保證模型輸入修正與路徑繪製使用一致的相機模型。
 *
 * calib 座標(openpilot 慣例):x 前、y 右、z 下。
 * @param forwardM  前方距離(m)
 * @param lateralLeftM 橫向偏移,左正(openpilot plan y)
 * @param heightM 相機離地高(m)
 * @return 影像像素 (x, y);點在相機後方或過近時 null
 */
fun roadToImage(
    forwardM: Float,
    lateralLeftM: Float,
    heightM: Float,
    imageWidth: Float,
    imageHeight: Float,
    horizontalFovDeg: Float,
    rollRad: Float,
    pitchRad: Float,
    yawRad: Float
): Pair<Float, Float>? {
    val fovRad = Math.toRadians(horizontalFovDeg.toDouble() / 2.0)
    val fl = imageWidth / (2f * kotlin.math.tan(fovRad).toFloat())
    val intrinsics = Mat3.intrinsics(fl, imageWidth / 2f, imageHeight / 2f)
    val viewFromDevice = Mat3(floatArrayOf(
        0f, 1f, 0f,
        0f, 0f, 1f,
        1f, 0f, 0f
    ))
    val deviceFromCalib = Mat3.rotationFromEuler(rollRad, pitchRad, yawRad)
    val cameraFromCalib = intrinsics * viewFromDevice * deviceFromCalib

    // calib frame 中的路面點:x 前、y 右(左正取負)、z 下(相機高度為正)
    val x = forwardM
    val y = -lateralLeftM
    val z = heightM

    val m = cameraFromCalib.m
    val w = m[6] * x + m[7] * y + m[8] * z
    if (w <= 0.1f) return null
    val px = (m[0] * x + m[1] * y + m[2] * z) / w
    val py = (m[3] * x + m[4] * y + m[5] * z) / w
    return Pair(px, py)
}
