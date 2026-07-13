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

/** inverse mapping:對每個模型幀像素,經 sourceFromModel 找來源像素,最近鄰取樣,出界填黑。 */
fun IntImage.warpToModelFrame(sourceFromModel: Mat3, big: Boolean): IntImage {
    val w = ModelFrames.MODEL_WIDTH
    val h = ModelFrames.MODEL_HEIGHT
    val out = IntArray(w * h) { 0xFF000000.toInt() }
    for (y in 0 until h) {
        for (x in 0 until w) {
            val (sx, sy) = sourceFromModel.map(x.toFloat(), y.toFloat())
            val ix = (sx + 0.5f).toInt()
            val iy = (sy + 0.5f).toInt()
            if (ix in 0 until width && iy in 0 until height) {
                out[y * w + x] = pixels[iy * width + ix]
            }
        }
    }
    return IntImage(w, h, out)
}
