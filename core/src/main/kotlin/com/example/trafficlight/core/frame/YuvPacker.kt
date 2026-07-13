package com.example.trafficlight.core.frame

/**
 * RGB(IntImage 512x256)→ YUV12 打包,佈局對齊 openpilot frames_to_tensor:
 * Y top-left / bottom-left / top-right / bottom-right,然後 U、V(各半解析度)。
 */
fun packYuv12(img: IntImage): ByteArray {
    val w = img.width
    val h = img.height
    require(w == 512 && h == 256) { "expected 512x256, got ${w}x$h" }
    val halfW = w / 2
    val halfH = h / 2
    val yPlane = IntArray(w * h)
    val uPlane = IntArray(halfW * halfH)
    val vPlane = IntArray(halfW * halfH)

    for (blockY in 0 until h step 2) {
        for (blockX in 0 until w step 2) {
            var uSum = 0
            var vSum = 0
            for (dy in 0..1) for (dx in 0..1) {
                val x = blockX + dx
                val y = blockY + dy
                val pixel = img.pixels[y * w + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                yPlane[y * w + x] = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
                uSum += (-0.169f * r - 0.331f * g + 0.5f * b + 128f).toInt().coerceIn(0, 255)
                vSum += (0.5f * r - 0.419f * g - 0.081f * b + 128f).toInt().coerceIn(0, 255)
            }
            val uvIndex = (blockY / 2) * halfW + (blockX / 2)
            uPlane[uvIndex] = uSum / 4
            vPlane[uvIndex] = vSum / 4
        }
    }

    val packed = ByteArray(6 * halfW * halfH)
    for (y in 0 until halfH) {
        for (x in 0 until halfW) {
            val base = y * halfW + x
            packed[base] = yPlane[(y * 2) * w + x * 2].toByte()
            packed[halfW * halfH + base] = yPlane[(y * 2 + 1) * w + x * 2].toByte()
            packed[2 * halfW * halfH + base] = yPlane[(y * 2) * w + x * 2 + 1].toByte()
            packed[3 * halfW * halfH + base] = yPlane[(y * 2 + 1) * w + x * 2 + 1].toByte()
            packed[4 * halfW * halfH + base] = uPlane[base].toByte()
            packed[5 * halfW * halfH + base] = vPlane[base].toByte()
        }
    }
    return packed
}
