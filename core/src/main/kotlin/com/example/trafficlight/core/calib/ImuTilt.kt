package com.example.trafficlight.core.calib

import kotlin.math.atan2
import kotlin.math.sqrt

data class Tilt(val rollDeg: Float, val pitchDeg: Float)

/**
 * 由重力向量(裝置座標:x 右、y 上、z 出螢幕)計算「轉正後幀」的殘餘滾轉與相機仰角。
 * rotationDegrees:CameraX rotationDegrees(幀轉正所需的順時針角度)。
 */
fun tiltFromGravity(gx: Float, gy: Float, gz: Float, rotationDegrees: Int): Tilt {
    // 裝置在螢幕平面內相對「直立」的角度(順時針為正)
    val screenAngleDeg = Math.toDegrees(atan2(gx.toDouble(), gy.toDouble())).toFloat()
    // 轉正後殘餘 roll:扣掉 90° 步進(rotationDegrees=90 表示直向)
    val stepDeg = when (((rotationDegrees % 360) + 360) % 360) {
        90 -> 0f     // 直向:直立即 0
        0 -> -90f    // 橫向(頂朝左)
        180 -> 90f   // 反向橫向
        270 -> 180f
        else -> 0f
    }
    var roll = screenAngleDeg + stepDeg
    while (roll > 180f) roll -= 360f
    while (roll < -180f) roll += 360f

    // 投影矩陣慣例:正 pitch = 相機朝下。手機後仰(相機朝上,gz>0)時必須回報負值,
    // 否則 warp 與路徑投影會往錯誤方向補償兩倍(實測:路線完全偏離路面)。
    val gPlane = sqrt((gx * gx + gy * gy).toDouble()).toFloat()
    val pitch = -Math.toDegrees(atan2(gz.toDouble(), gPlane.toDouble())).toFloat()
    return Tilt(roll, pitch)
}
