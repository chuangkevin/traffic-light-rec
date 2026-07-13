package com.example.trafficlight.harness

import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.frame.warpToModelFrame
import com.example.trafficlight.core.geometry.sourceFromModelFrame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.Color
import java.awt.image.BufferedImage

class TiltRegressionTest {

    /** 產生上灰下黑的地平線合成幀 */
    private fun horizonFrame(w: Int = 1280, h: Int = 720): BufferedImage {
        val img = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        g.color = Color(180, 180, 220); g.fillRect(0, 0, w, h / 2)
        g.color = Color(40, 40, 40); g.fillRect(0, h / 2, w, h / 2)
        g.dispose()
        return img
    }

    private fun meanAbsDiff(a: IntImage, b: IntImage): Double {
        var sum = 0.0
        for (i in a.pixels.indices) {
            val pa = a.pixels[i] and 0xFF
            val pb = b.pixels[i] and 0xFF
            sum += kotlin.math.abs(pa - pb)
        }
        return sum / a.pixels.size
    }

    @Test
    fun rollCorrectionRecoversUprightWarp() {
        val upright = horizonFrame()
        val tilted = rotateByDegrees(upright, 8.0)

        val baseline = bufferedToIntImage(upright).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, 0f, 0f, 0f, false), false)

        // 無修正:歪斜輸入 + roll=0 warp → 與 baseline 差異大
        val uncorrected = bufferedToIntImage(tilted).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, 0f, 0f, 0f, false), false)

        // 有修正:歪斜輸入 + roll=8° warp → 接近 baseline
        val rollRad = Math.toRadians(8.0).toFloat()
        val corrected = bufferedToIntImage(tilted).warpToModelFrame(
            sourceFromModelFrame(1280f, 720f, 72f, rollRad, 0f, 0f, false), false)

        val diffUncorrected = meanAbsDiff(baseline, uncorrected)
        val diffCorrected = meanAbsDiff(baseline, corrected)
        assertTrue(
            "corrected ($diffCorrected) should beat uncorrected ($diffUncorrected)",
            diffCorrected < diffUncorrected * 0.5
        )
    }
}
