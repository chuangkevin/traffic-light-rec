package com.example.trafficlight.harness

import com.example.trafficlight.core.calib.CalibrationState
import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.frame.IntImage
import com.example.trafficlight.core.geometry.roadToImage
import com.example.trafficlight.core.pipeline.DrivingPipeline
import com.example.trafficlight.core.plan.PlanPoint
import java.awt.BasicStroke
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

/** 用與 app OverlayView 相同的 core 投影把規劃路徑畫到幀上(驗證路徑壓在馬路上)。 */
fun drawPath(
    g: java.awt.Graphics2D,
    path: List<PlanPoint>,
    cal: CalibrationState,
    imgW: Float,
    imgH: Float,
    fovDeg: Float
) {
    val rollRad = Math.toRadians(cal.rollDeg.toDouble()).toFloat()
    val pitchRad = Math.toRadians(cal.pitchDeg.toDouble()).toFloat()
    val yawRad = Math.toRadians(cal.yawDeg.toDouble()).toFloat()
    val pts = path.filter { it.x > 0.5f }.mapNotNull { p ->
        roadToImage(p.x, p.y, cal.heightM, imgW, imgH, fovDeg, rollRad, pitchRad, yawRad)
    }
    if (pts.size < 2) return
    g.color = Color(0, 255, 120, 200)
    g.stroke = BasicStroke(4f)
    for (i in 1 until pts.size) {
        g.drawLine(pts[i - 1].first.toInt(), pts[i - 1].second.toInt(),
            pts[i].first.toInt(), pts[i].second.toInt())
    }
    // 車道寬參考線(±1.75m)
    g.color = Color(255, 255, 0, 130)
    g.stroke = BasicStroke(2f)
    for (edge in floatArrayOf(-1.75f, 1.75f)) {
        val edgePts = path.filter { it.x > 0.5f }.mapNotNull { p ->
            roadToImage(p.x, p.y + edge, cal.heightM, imgW, imgH, fovDeg, rollRad, pitchRad, yawRad)
        }
        for (i in 1 until edgePts.size) {
            g.drawLine(edgePts[i - 1].first.toInt(), edgePts[i - 1].second.toInt(),
                edgePts[i].first.toInt(), edgePts[i].second.toInt())
        }
    }
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
        drawPath(g, plan.path, result.calibration, input.width.toFloat(), input.height.toFloat(), fovDeg)
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
