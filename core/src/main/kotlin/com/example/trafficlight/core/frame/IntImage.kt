package com.example.trafficlight.core.frame

import com.example.trafficlight.core.geometry.Mat3
import com.example.trafficlight.core.geometry.ModelFrames

class IntImage(val width: Int, val height: Int, val pixels: IntArray) {
    init { require(pixels.size == width * height) }
}

/** 順時針旋轉,degrees ∈ {0, 90, 180, 270}。用於把 CameraX 幀轉正。 */
fun IntImage.rotate90(degrees: Int): IntImage {
    return when (((degrees % 360) + 360) % 360) {
        0 -> this
        90 -> {
            val out = IntArray(pixels.size)
            for (y in 0 until height) for (x in 0 until width) {
                out[x * height + (height - 1 - y)] = pixels[y * width + x]
            }
            IntImage(height, width, out)
        }
        180 -> {
            val out = IntArray(pixels.size)
            for (i in pixels.indices) out[pixels.size - 1 - i] = pixels[i]
            IntImage(width, height, out)
        }
        270 -> {
            val out = IntArray(pixels.size)
            for (y in 0 until height) for (x in 0 until width) {
                out[(width - 1 - x) * height + y] = pixels[y * width + x]
            }
            IntImage(height, width, out)
        }
        else -> throw IllegalArgumentException("unsupported rotation: $degrees")
    }
}

/** inverse mapping:對每個模型幀像素,經 sourceFromModel 找來源像素,最近鄰取樣,出界填黑。
 *  熱路徑:矩陣運算全部 inline,避免每像素配置 Pair(每幀 26 萬次配置的 GC 壓力)。 */
fun IntImage.warpToModelFrame(sourceFromModel: Mat3, big: Boolean): IntImage {
    val w = ModelFrames.MODEL_WIDTH
    val h = ModelFrames.MODEL_HEIGHT
    val out = IntArray(w * h) { 0xFF000000.toInt() }
    val m = sourceFromModel.m
    val m0 = m[0]; val m1 = m[1]; val m2 = m[2]
    val m3 = m[3]; val m4 = m[4]; val m5 = m[5]
    val m6 = m[6]; val m7 = m[7]; val m8 = m[8]
    val srcW = width
    val srcH = height
    for (y in 0 until h) {
        val yf = y.toFloat()
        // 行起點與 x 方向增量(矩陣對 x 是線性的,逐行遞增省 2/3 乘法)
        var nx = m1 * yf + m2
        var ny = m4 * yf + m5
        var nw = m7 * yf + m8
        val rowBase = y * w
        for (x in 0 until w) {
            if (nw > 1e-6f) {
                val ix = (nx / nw + 0.5f).toInt()
                val iy = (ny / nw + 0.5f).toInt()
                if (ix in 0 until srcW && iy in 0 until srcH) {
                    out[rowBase + x] = pixels[iy * srcW + ix]
                }
            }
            nx += m0
            ny += m3
            nw += m6
        }
    }
    return IntImage(w, h, out)
}
