package com.example.trafficlight.core.geometry

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraProjectionTest {
    // 模型幀中心(cx, cy)在零角度時應映射到來源影像中心
    @Test fun zeroAnglesMapsPrincipalPointToImageCenter() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, bigModelFrame = false)
        val (x, y) = m.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        assertEquals(960f, x, 1.0f)
        assertEquals(540f, y, 1.0f)
    }

    // roll > 0 時,模型幀主點左右兩側的取樣點在來源影像中的 y 應不同(地平線傾斜補償)
    @Test fun rollTiltsSampling() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0.14f, 0f, 0f, bigModelFrame = false)
        val (_, yLeft) = m.map(ModelFrames.MEDMODEL_CX - 100f, ModelFrames.MEDMODEL_CY)
        val (_, yRight) = m.map(ModelFrames.MEDMODEL_CX + 100f, ModelFrames.MEDMODEL_CY)
        assertTrue("roll should skew sampling rows", kotlin.math.abs(yLeft - yRight) > 10f)
    }

    // roll = 0 時左右對稱
    @Test fun zeroRollIsSymmetric() {
        val m = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, bigModelFrame = false)
        val (_, yLeft) = m.map(ModelFrames.MEDMODEL_CX - 100f, ModelFrames.MEDMODEL_CY)
        val (_, yRight) = m.map(ModelFrames.MEDMODEL_CX + 100f, ModelFrames.MEDMODEL_CY)
        assertEquals(yLeft, yRight, 0.5f)
    }

    // pitch 改變時取樣區應垂直移動
    @Test fun pitchShiftsVertically() {
        val m0 = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, false)
        val m1 = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0.1f, 0f, false)
        val (_, y0) = m0.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        val (_, y1) = m1.map(ModelFrames.MEDMODEL_CX, ModelFrames.MEDMODEL_CY)
        assertTrue(kotlin.math.abs(y0 - y1) > 20f)
    }

    @Test fun bigFrameUsesWiderIntrinsics() {
        val med = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, false)
        val big = sourceFromModelFrame(1920f, 1080f, 72f, 0f, 0f, 0f, true)
        // 大幀焦距一半 → 同一模型像素位移對應兩倍來源位移
        val (xm, _) = med.map(ModelFrames.MEDMODEL_CX + 50f, ModelFrames.MEDMODEL_CY)
        val (xb, _) = big.map(ModelFrames.SBIGMODEL_CX + 50f, ModelFrames.SBIGMODEL_CY)
        assertTrue((xb - 960f) > (xm - 960f) * 1.5f)
    }
}
