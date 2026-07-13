package com.example.trafficlight.core.frame

import org.junit.Assert.assertEquals
import org.junit.Test

class YuvPackerTest {
    private val w = 512
    private val h = 256
    private val half = (w / 2) * (h / 2) // 32768

    @Test fun outputSizeIs6Planes() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFF000000.toInt() })
        assertEquals(6 * half, packYuv12(img).size)
    }

    @Test fun whiteImageYIs255UvIs128() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFFFFFFFF.toInt() })
        val p = packYuv12(img)
        assertEquals(255, p[0].toInt() and 0xFF)          // Y plane 0 (top-left)
        assertEquals(128, p[4 * half].toInt() and 0xFF)   // U
        assertEquals(128, p[5 * half].toInt() and 0xFF)   // V
    }

    @Test fun pureRedHasHighV() {
        val img = IntImage(w, h, IntArray(w * h) { 0xFFFF0000.toInt() })
        val p = packYuv12(img)
        val v = p[5 * half].toInt() and 0xFF
        assertEquals(255, v) // 0.5*255+128 clamp → 255
        val u = p[4 * half].toInt() and 0xFF
        assertEquals(84.0, u.toDouble(), 2.0) // -0.169*255+128 ≈ 84.9
    }

    @Test fun subsamplePositionsAreQuadrants() {
        // 只有像素 (1,0)(2x2 block 0 的 top-right)是白,其他黑
        val px = IntArray(w * h) { 0xFF000000.toInt() }
        px[1] = 0xFFFFFFFF.toInt()
        val p = packYuv12(IntImage(w, h, px))
        assertEquals(0, p[0].toInt() and 0xFF)             // Y TL
        assertEquals(0, p[half].toInt() and 0xFF)          // Y BL
        assertEquals(255, p[2 * half].toInt() and 0xFF)    // Y TR ← 白像素在這
        assertEquals(0, p[3 * half].toInt() and 0xFF)      // Y BR
    }
}
