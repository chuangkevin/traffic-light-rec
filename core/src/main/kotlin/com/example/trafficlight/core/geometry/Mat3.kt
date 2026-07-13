package com.example.trafficlight.core.geometry

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class Mat3(val m: FloatArray) {
    init { require(m.size == 9) }

    operator fun times(o: Mat3): Mat3 {
        val r = FloatArray(9)
        for (row in 0..2) for (col in 0..2) {
            r[row * 3 + col] = m[row * 3] * o.m[col] +
                m[row * 3 + 1] * o.m[3 + col] +
                m[row * 3 + 2] * o.m[6 + col]
        }
        return Mat3(r)
    }

    fun map(x: Float, y: Float): Pair<Float, Float> {
        val w = m[6] * x + m[7] * y + m[8]
        return Pair(
            (m[0] * x + m[1] * y + m[2]) / w,
            (m[3] * x + m[4] * y + m[5]) / w
        )
    }

    fun invert(): Mat3? {
        val det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
            m[1] * (m[3] * m[8] - m[5] * m[6]) +
            m[2] * (m[3] * m[7] - m[4] * m[6])
        if (!det.isFinite() || abs(det) < 1e-6f) return null
        val i = 1f / det
        return Mat3(floatArrayOf(
            (m[4] * m[8] - m[5] * m[7]) * i, (m[2] * m[7] - m[1] * m[8]) * i, (m[1] * m[5] - m[2] * m[4]) * i,
            (m[5] * m[6] - m[3] * m[8]) * i, (m[0] * m[8] - m[2] * m[6]) * i, (m[2] * m[3] - m[0] * m[5]) * i,
            (m[3] * m[7] - m[4] * m[6]) * i, (m[1] * m[6] - m[0] * m[7]) * i, (m[0] * m[4] - m[1] * m[3]) * i
        ))
    }

    companion object {
        fun identity() = Mat3(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f))

        fun intrinsics(fl: Float, cx: Float, cy: Float) =
            Mat3(floatArrayOf(fl, 0f, cx, 0f, fl, cy, 0f, 0f, 1f))

        /** openpilot 順序:yaw * pitch * roll */
        fun rotationFromEuler(roll: Float, pitch: Float, yaw: Float): Mat3 {
            val cr = cos(roll); val sr = sin(roll)
            val cp = cos(pitch); val sp = sin(pitch)
            val cy = cos(yaw); val sy = sin(yaw)
            val rollM = Mat3(floatArrayOf(1f, 0f, 0f, 0f, cr, -sr, 0f, sr, cr))
            val pitchM = Mat3(floatArrayOf(cp, 0f, sp, 0f, 1f, 0f, -sp, 0f, cp))
            val yawM = Mat3(floatArrayOf(cy, -sy, 0f, sy, cy, 0f, 0f, 0f, 1f))
            return yawM * (pitchM * rollM)
        }
    }
}
