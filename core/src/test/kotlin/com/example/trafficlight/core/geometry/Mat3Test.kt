package com.example.trafficlight.core.geometry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class Mat3Test {
    private fun assertMat(expected: FloatArray, actual: Mat3, eps: Float = 1e-4f) {
        for (i in 0..8) assertEquals("index $i", expected[i], actual.m[i], eps)
    }

    @Test fun identityTimesIsSame() {
        val a = Mat3(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 10f))
        assertMat(a.m, Mat3.identity() * a)
    }

    @Test fun invertRecoversIdentity() {
        val a = Mat3(floatArrayOf(2f, 0f, 1f, 0f, 3f, 0f, 0f, 0f, 1f))
        val inv = a.invert()
        assertNotNull(inv)
        assertMat(Mat3.identity().m, a * inv!!)
    }

    @Test fun singularReturnsNull() {
        val a = Mat3(FloatArray(9) { 1f })
        assertEquals(null, a.invert())
    }

    @Test fun eulerZeroIsIdentity() {
        assertMat(Mat3.identity().m, Mat3.rotationFromEuler(0f, 0f, 0f))
    }

    @Test fun rollRotatesAroundX() {
        // openpilot 順序 yaw*pitch*roll;roll=90° 時繞 x 軸旋轉
        val r = Mat3.rotationFromEuler((Math.PI / 2).toFloat(), 0f, 0f)
        assertEquals(0f, r.m[4], 1e-4f)  // m[1][1] = cos90 = 0
        assertEquals(-1f, r.m[5], 1e-4f) // m[1][2] = -sin90
    }

    @Test fun mapAppliesHomogeneousDivide() {
        val scale2 = Mat3(floatArrayOf(2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f))
        val (x, y) = scale2.map(3f, 4f)
        assertEquals(3f, x, 1e-4f)
        assertEquals(4f, y, 1e-4f)
    }
}
