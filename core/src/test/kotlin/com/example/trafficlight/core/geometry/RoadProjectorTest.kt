package com.example.trafficlight.core.geometry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RoadProjectorTest {
    private val w = 1280f
    private val h = 720f
    private val fov = 72f

    // 零角度、正前方路面點 → 水平置中、垂直在中線以下(路面在地平線下)
    @Test fun straightAheadPointIsCenteredBelowHorizon() {
        val p = roadToImage(20f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        assertEquals(w / 2f, p.first, 1f)
        assertTrue("below center, got ${p.second}", p.second > h / 2f)
    }

    // 越遠的點越接近地平線(畫面上方)
    @Test fun fartherPointsRiseTowardHorizon() {
        val near = roadToImage(10f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        val far = roadToImage(60f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        assertTrue(far.second < near.second)
    }

    // 左側點(lateral 左正)投影在畫面左半
    @Test fun leftLateralProjectsLeft() {
        val p = roadToImage(20f, 2f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        assertTrue(p.first < w / 2f)
    }

    // roll 使正前方點水平偏移(地平線傾斜的補償方向)
    @Test fun rollShiftsProjection() {
        val flat = roadToImage(20f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        val rolled = roadToImage(20f, 0f, 1.35f, w, h, fov, Math.toRadians(8.0).toFloat(), 0f, 0f)!!
        assertTrue(kotlin.math.abs(rolled.first - flat.first) > 5f)
    }

    // pitch 使正前方點垂直移動
    @Test fun pitchShiftsVertically() {
        val flat = roadToImage(20f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        val pitched = roadToImage(20f, 0f, 1.35f, w, h, fov, 0f, Math.toRadians(6.0).toFloat(), 0f)!!
        assertTrue(kotlin.math.abs(pitched.second - flat.second) > 20f)
    }

    // 相機後方的點 → null
    @Test fun behindCameraIsNull() {
        assertNull(roadToImage(-5f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f))
    }

    // 與 warp 的一致性:warp 用的 sourceFromModelFrame 把「模型幀中心」映到來源像素;
    // 同一組角度下,roadToImage 的地平線方向必須一致(pitch 增加時兩者同向移動)
    @Test fun consistentDirectionWithWarpMatrix() {
        val p0 = roadToImage(50f, 0f, 1.35f, w, h, fov, 0f, 0f, 0f)!!
        val m0 = sourceFromModelFrame(w, h, fov, 0f, 0f, 0f, false)
            .map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        val pitchRad = Math.toRadians(5.0).toFloat()
        val p1 = roadToImage(50f, 0f, 1.35f, w, h, fov, 0f, pitchRad, 0f)!!
        val m1 = sourceFromModelFrame(w, h, fov, 0f, pitchRad, 0f, false)
            .map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        val projDelta = p1.second - p0.second
        val warpDelta = m1.second - m0.second
        assertTrue("same direction: proj=$projDelta warp=$warpDelta", projDelta * warpDelta > 0)
    }
}
