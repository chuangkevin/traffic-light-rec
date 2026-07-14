package com.example.trafficlight.harness

import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.pipeline.DrivingPipeline
import java.awt.Color
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

fun bufferedToIntImage(img: BufferedImage): IntImage {
    val pixels = IntArray(img.width * img.height)
    img.getRGB(0, 0, img.width, img.height, pixels, 0, img.width)
    return IntImage(img.width, img.height, pixels)
}

/** 以影像中心旋轉 rollDeg(模擬歪斜的手機),尺寸不變、出界填黑。 */
fun rotateByDegrees(img: BufferedImage, rollDeg: Double): BufferedImage {
    val out = BufferedImage(img.width, img.height, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.color = Color.BLACK
    g.fillRect(0, 0, img.width, img.height)
    val t = AffineTransform.getRotateInstance(
        Math.toRadians(rollDeg), img.width / 2.0, img.height / 2.0)
    g.drawImage(img, t, null)
    g.dispose()
    return out
}

fun main(args: Array<String>) {
    require(args.size >= 2) { "usage: <framesDir> <outDir> [rollDeg] [fovDeg]" }
    val framesDir = File(args[0])
    val outDir = File(args[1]).apply { mkdirs() }
    val rollDeg = if (args.size >= 3) args[2].toDouble() else 0.0
    val fovDeg = if (args.size >= 4) args[3].toFloat() else 72f
    val annotatedDir = File(outDir, "annotated").apply { mkdirs() }

    val modelsDir = File("app/src/main/assets/models")
    val runner = DesktopModelRunner(
        File(modelsDir, "openpilot_driving_vision.onnx"),
        File(modelsDir, "openpilot_driving_policy.onnx")
    )
    val pipeline = DrivingPipeline(runner)
    pipeline.onSpeed(10f) // 模擬行駛中(36 km/h),讓校正速度閘門放行

    val frames = framesDir.listFiles { f -> f.extension.lowercase() in setOf("png", "jpg", "jpeg") }
        ?.sortedBy { it.name } ?: emptyList()
    require(frames.isNotEmpty()) { "no frames in $framesDir" }

    val events = StringBuilder("frameIndex,action,confidence,nearVel,desiredAccel,calibValid\n")
    var timestampMs = 0L

    frames.forEachIndexed { index, file ->
        val original = ImageIO.read(file)
        val input = if (rollDeg != 0.0) rotateByDegrees(original, rollDeg) else original
        // 模擬 IMU:回報與影像旋轉相同的 roll(每幀 3 個 20ms 樣本 ≈ 15fps 下的 50Hz)
        repeat(3) {
            timestampMs += 20
            pipeline.onImuTilt(Tilt(rollDeg.toFloat(), 0f), timestampMs)
        }
        val result = pipeline.processFrame(bufferedToIntImage(input), rotationDegrees = 0, horizontalFovDeg = fovDeg)
        val plan = result.plan

        events.append("$index,${plan.action},${"%.2f".format(plan.confidence)},")
        events.append("${"%.2f".format(plan.nearVelocity)},${"%.2f".format(plan.desiredAcceleration)},")
        events.append("${result.calibration.valid}\n")

        val annotated = BufferedImage(input.width, input.height, BufferedImage.TYPE_INT_RGB)
        val g = annotated.createGraphics()
        g.drawImage(input, 0, 0, null)
        g.color = when (plan.action.name) {
            "STOP" -> Color.RED
            "GO" -> Color.GREEN
            else -> Color.GRAY
        }
        g.fillRect(10, 10, 200, 40)
        g.color = Color.WHITE
        g.drawString("${plan.action} v=${"%.1f".format(plan.nearVelocity)}m/s", 20, 35)
        g.dispose()
        ImageIO.write(annotated, "png", File(annotatedDir, "%05d.png".format(index)))

        if (index % 20 == 0) println("frame $index/${frames.size}: ${plan.action}")
    }

    File(outDir, "events.csv").writeText(events.toString())
    runner.close()
    println("done. events: ${File(outDir, "events.csv").absolutePath}")
}
