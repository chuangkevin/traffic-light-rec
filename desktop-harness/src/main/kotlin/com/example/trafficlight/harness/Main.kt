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

/** 畫路面網格(橫線每 10m、縱線 -3.5/0/+3.5m)供地平線/透視比對。 */
private fun drawRoadGrid(
    g: java.awt.Graphics2D, cal: CalibrationState, imgW: Float, imgH: Float, fovDeg: Float
) {
    val roll = Math.toRadians(cal.rollDeg.toDouble()).toFloat()
    val pitch = Math.toRadians(cal.pitchDeg.toDouble()).toFloat()
    val yaw = Math.toRadians(cal.yawDeg.toDouble()).toFloat()
    g.color = Color(80, 180, 255, 160)
    g.stroke = BasicStroke(2f)
    for (f in intArrayOf(5, 10, 20, 30, 45)) {
        val a = roadToImage(f.toFloat(), -5f, cal.heightM, imgW, imgH, fovDeg, roll, pitch, yaw)
        val b = roadToImage(f.toFloat(), 5f, cal.heightM, imgW, imgH, fovDeg, roll, pitch, yaw)
        if (a != null && b != null) {
            g.drawLine(a.first.toInt(), a.second.toInt(), b.first.toInt(), b.second.toInt())
            g.drawString("${f}m", (b.first + 4).toInt(), b.second.toInt())
        }
    }
    for (lat in floatArrayOf(-3.5f, 0f, 3.5f)) {
        var prev: Pair<Float, Float>? = null
        var f = 4f
        while (f <= 60f) {
            val p = roadToImage(f, lat, cal.heightM, imgW, imgH, fovDeg, roll, pitch, yaw)
            if (prev != null && p != null) {
                g.drawLine(prev.first.toInt(), prev.second.toInt(), p.first.toInt(), p.second.toInt())
            }
            prev = p
            f += 4f
        }
    }
}

/** render-dump 模式:讀手機偵錯快照,用四種 roll/pitch 正負號組合各輸出一張比對圖。 */
fun renderDump(dumpDir: File, outDir: File) {
    val meta = File(dumpDir, "meta.txt").readLines()
        .mapNotNull { line -> line.split("=", limit = 2).takeIf { it.size == 2 }?.let { it[0] to it[1] } }
        .groupBy({ it.first }, { it.second })
    fun v(key: String) = meta[key]?.first()?.toFloat() ?: 0f
    val path = meta["path"].orEmpty().map {
        val (x, y) = it.split(","); PlanPoint(x.toFloat(), y.toFloat())
    }
    val frame = ImageIO.read(File(dumpDir, "frame.png"))
    val fov = v("fovDeg")
    outDir.mkdirs()

    for ((label, rollSign, pitchSign) in listOf(
        Triple("r+p+", 1f, 1f), Triple("r+p-", 1f, -1f),
        Triple("r-p+", -1f, 1f), Triple("r-p-", -1f, -1f)
    )) {
        val cal = CalibrationState(
            rollDeg = v("rollDeg") * rollSign,
            pitchDeg = v("pitchDeg") * pitchSign,
            yawDeg = v("yawDeg"),
            heightM = v("heightM"),
            valid = true, sampleCount = 0
        )
        val out = BufferedImage(frame.width, frame.height, BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        g.drawImage(frame, 0, 0, null)
        drawRoadGrid(g, cal, frame.width.toFloat(), frame.height.toFloat(), fov)
        drawPath(g, path, cal, frame.width.toFloat(), frame.height.toFloat(), fov)
        g.color = Color.WHITE
        g.fillRect(0, 0, 340, 28)
        g.color = Color.BLACK
        g.drawString("$label roll=${cal.rollDeg} pitch=${cal.pitchDeg} fov=$fov", 6, 20)
        g.dispose()
        ImageIO.write(out, "png", File(outDir, "${dumpDir.name}-$label.png"))
    }
    println("rendered 4 variants for ${dumpDir.name} -> $outDir")
}

fun main(args: Array<String>) {
    if (args.isNotEmpty() && args[0] == "render-dump") {
        require(args.size >= 3) { "usage: render-dump <dumpRoot> <outDir>" }
        val root = File(args[1])
        val outDir = File(args[2])
        val dumps = root.listFiles { f -> f.isDirectory && File(f, "meta.txt").exists() }
            ?.sortedBy { it.name } ?: emptyList()
        require(dumps.isNotEmpty()) { "no dumps in $root" }
        dumps.forEach { renderDump(it, outDir) }
        return
    }
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
