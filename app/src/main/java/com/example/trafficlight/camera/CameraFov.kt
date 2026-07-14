package com.example.trafficlight.camera

import android.hardware.camera2.CameraCharacteristics
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import kotlin.math.atan

/** 從 Camera2 characteristics 計算水平 FOV(度);讀不到時回傳 fallback。 */
@OptIn(ExperimentalCamera2Interop::class)
fun horizontalFovDeg(cameraInfo: CameraInfo, zoomRatio: Float, fallbackDeg: Float = 72f): Float {
    return try {
        val c2 = Camera2CameraInfo.from(cameraInfo)
        val focalLengths = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val sensorSize = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        if (focalLengths == null || focalLengths.isEmpty() || sensorSize == null) return fallbackDeg
        val fl = focalLengths[0]
        // 變焦是裁切:half-FOV 的 tan 除以 zoom,再轉回角度(FOV/zoom 是錯的近似)
        val baseHalfTan = sensorSize.width / (2f * fl)
        val effHalfTan = baseHalfTan / zoomRatio.coerceAtLeast(1f)
        Math.toDegrees(2.0 * atan(effHalfTan.toDouble())).toFloat()
    } catch (e: Exception) {
        fallbackDeg
    }
}
