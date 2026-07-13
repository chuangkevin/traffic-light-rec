package com.example.trafficlight.core.frame

import com.example.trafficlight.core.geometry.Mat3
import com.example.trafficlight.core.geometry.ModelFrames
import org.junit.Assert.assertEquals
import org.junit.Test

class IntImageTest {
    // 2x1 影像:左紅右綠
    private val redGreen = IntImage(2, 1, intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt()))

    @Test fun rotate0IsIdentity() {
        val r = redGreen.rotate90(0)
        assertEquals(0xFFFF0000.toInt(), r.pixels[0])
    }

    @Test fun rotate90MakesPortrait() {
        val r = redGreen.rotate90(90)
        assertEquals(1, r.width)
        assertEquals(2, r.height)
        // 順時針 90°:紅(左)轉到上、綠(右)轉到下
        assertEquals(0xFFFF0000.toInt(), r.pixels[0])
        assertEquals(0xFF00FF00.toInt(), r.pixels[1])
    }

    @Test fun rotate180Reverses() {
        val r = redGreen.rotate90(180)
        assertEquals(0xFF00FF00.toInt(), r.pixels[0])
        assertEquals(0xFFFF0000.toInt(), r.pixels[1])
    }

    @Test fun rotate360ViaTwo180sMatches() {
        val r = redGreen.rotate90(180).rotate90(180)
        assertEquals(redGreen.pixels.toList(), r.pixels.toList())
    }

    @Test fun rotate90Plus270IsIdentity() {
        val src = IntImage(3, 2, IntArray(6) { it })
        val r = src.rotate90(90).rotate90(270)
        assertEquals(src.pixels.toList(), r.pixels.toList())
    }

    @Test fun warpIdentityMatrixSamplesDirectly() {
        // 用單位矩陣:模型像素 (x,y) 直接取來源 (x,y)
        val src = IntImage(ModelFrames.MODEL_WIDTH, ModelFrames.MODEL_HEIGHT,
            IntArray(ModelFrames.MODEL_WIDTH * ModelFrames.MODEL_HEIGHT) { it })
        val out = src.warpToModelFrame(Mat3.identity(), big = false)
        assertEquals(src.pixels[5000], out.pixels[5000])
    }

    @Test fun warpOutOfBoundsIsBlack() {
        // 平移超出來源範圍 → 黑
        val shift = Mat3(floatArrayOf(1f, 0f, 99999f, 0f, 1f, 0f, 0f, 0f, 1f))
        val src = IntImage(4, 4, IntArray(16) { 0xFFFFFFFF.toInt() })
        val out = src.warpToModelFrame(shift, big = false)
        assertEquals(0xFF000000.toInt(), out.pixels[0])
    }
}
