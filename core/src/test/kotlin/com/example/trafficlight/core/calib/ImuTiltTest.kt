package com.example.trafficlight.core.calib

import org.junit.Assert.assertEquals
import org.junit.Test

class ImuTiltTest {
    // 手機直立(直向)、相機水平朝前:重力沿裝置 +y
    @Test fun uprightPortraitIsZero() {
        val t = tiltFromGravity(0f, 9.81f, 0f, rotationDegrees = 90)
        assertEquals(0f, t.rollDeg, 0.5f)
        assertEquals(0f, t.pitchDeg, 0.5f)
    }

    // 橫向(rotationDegrees=0):重力沿裝置 +x
    @Test fun landscapeIsZero() {
        val t = tiltFromGravity(9.81f, 0f, 0f, rotationDegrees = 0)
        assertEquals(0f, t.rollDeg, 0.5f)
        assertEquals(0f, t.pitchDeg, 0.5f)
    }

    // 直向、順時針歪 10°:gx = sin10°·g, gy = cos10°·g
    @Test fun tenDegreeRollDetected() {
        val g = 9.81f
        val gx = (g * kotlin.math.sin(Math.toRadians(10.0))).toFloat()
        val gy = (g * kotlin.math.cos(Math.toRadians(10.0))).toFloat()
        val t = tiltFromGravity(gx, gy, 0f, rotationDegrees = 90)
        assertEquals(10f, kotlin.math.abs(t.rollDeg), 1.0f)
    }

    // 直向、上仰 15°(頂往後倒):gz = sin15°·g
    @Test fun pitchUpDetected() {
        val g = 9.81f
        val gy = (g * kotlin.math.cos(Math.toRadians(15.0))).toFloat()
        val gz = (g * kotlin.math.sin(Math.toRadians(15.0))).toFloat()
        val t = tiltFromGravity(0f, gy, gz, rotationDegrees = 90)
        assertEquals(15f, t.pitchDeg, 1.0f)
    }
}
