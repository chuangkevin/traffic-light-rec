package com.example.trafficlight

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.view.View
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.trafficlight.analyzer.FrameAnalyzer
import com.example.trafficlight.camera.horizontalFovDeg
import com.example.trafficlight.inference.InferenceEngine
import com.example.trafficlight.sensor.ImuManager
import com.example.trafficlight.sensor.SpeedMonitor
import com.example.trafficlight.logic.RoiSelector
import com.example.trafficlight.logic.StateMachine
import com.example.trafficlight.logic.TrafficLightState
import com.example.trafficlight.ui.OverlayView
import com.example.trafficlight.util.hasCameraPermission
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var calibStatusText: TextView
    private lateinit var fpsText: TextView
    private lateinit var debugText: TextView
    private lateinit var controlPanel: View
    private lateinit var soundPanel: View
    private lateinit var statusPanel: View
    private lateinit var zoomOutButton: Button
    private lateinit var zoomInButton: Button
    private lateinit var routeUpButton: Button
    private lateinit var routeDownButton: Button
    private lateinit var routeNarrowButton: Button
    private lateinit var routeWideButton: Button
    private lateinit var pitchUpButton: Button
    private lateinit var pitchDownButton: Button
    private lateinit var fovWideButton: Button
    private lateinit var fovNarrowButton: Button
    private lateinit var cameraHeightUpButton: Button
    private lateinit var cameraHeightDownButton: Button
    private lateinit var cameraLeftButton: Button
    private lateinit var cameraRightButton: Button
    private lateinit var calibrationResetButton: Button
    private lateinit var frameThinButton: Button
    private lateinit var frameThickButton: Button
    private lateinit var stopSoundSwitch: Switch
    private lateinit var goSoundSwitch: Switch
    
    private lateinit var inferenceEngine: InferenceEngine
    private lateinit var stateMachine: StateMachine
    private lateinit var roiSelector: RoiSelector
    private lateinit var frameAnalyzer: FrameAnalyzer
    
    private var imuManager: ImuManager? = null
    private var speedMonitor: SpeedMonitor? = null
    private var toneGenerator: ToneGenerator? = null
    private var stopSoundEnabled = true
    private var goSoundEnabled = true
    
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null
    private var currentZoomRatio = 1.4f
    private var minZoomRatio = 1f
    private var maxZoomRatio = 1f
    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        val audioGranted = permissions[Manifest.permission.RECORD_AUDIO] == true

        if (cameraGranted && audioGranted) {
            if (permissions[Manifest.permission.ACCESS_FINE_LOCATION] != true) {
                Toast.makeText(this, "未授權定位:校正需要 GPS 車速,將無法自動校正", Toast.LENGTH_LONG).show()
            }
            speedMonitor?.start()
            startCamera()
        } else {
            Toast.makeText(this, "需要相機和音頻權限", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        initViews()
        initComponents()
        checkPermissions()
    }

    private fun initViews() {
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        calibStatusText = findViewById(R.id.calibStatusText)
        fpsText = findViewById(R.id.fpsText)
        debugText = findViewById(R.id.debugText)
        controlPanel = findViewById(R.id.controlPanel)
        soundPanel = findViewById(R.id.soundPanel)
        statusPanel = findViewById(R.id.statusPanel)
        zoomOutButton = findViewById(R.id.zoomOutButton)
        zoomInButton = findViewById(R.id.zoomInButton)
        routeUpButton = findViewById(R.id.routeUpButton)
        routeDownButton = findViewById(R.id.routeDownButton)
        routeNarrowButton = findViewById(R.id.routeNarrowButton)
        routeWideButton = findViewById(R.id.routeWideButton)
        pitchUpButton = findViewById(R.id.pitchUpButton)
        pitchDownButton = findViewById(R.id.pitchDownButton)
        fovWideButton = findViewById(R.id.fovWideButton)
        fovNarrowButton = findViewById(R.id.fovNarrowButton)
        cameraHeightUpButton = findViewById(R.id.cameraHeightUpButton)
        cameraHeightDownButton = findViewById(R.id.cameraHeightDownButton)
        cameraLeftButton = findViewById(R.id.cameraLeftButton)
        cameraRightButton = findViewById(R.id.cameraRightButton)
        calibrationResetButton = findViewById(R.id.calibrationResetButton)
        frameThinButton = findViewById(R.id.frameThinButton)
        frameThickButton = findViewById(R.id.frameThickButton)
        stopSoundSwitch = findViewById(R.id.stopSoundSwitch)
        goSoundSwitch = findViewById(R.id.goSoundSwitch)

        setupControlButtons()
        previewView.setOnClickListener { toggleControls() }
        overlayView.setOnClickListener { toggleControls() }
    }

    private fun toggleControls() {
        val nextVisibility = if (controlPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        controlPanel.visibility = nextVisibility
        soundPanel.visibility = nextVisibility
    }

    private fun setupControlButtons() {
        zoomOutButton.setOnClickListener { adjustZoom(-0.2f) }
        zoomInButton.setOnClickListener { adjustZoom(0.2f) }
        routeUpButton.setOnClickListener { overlayView.adjustRouteVertical(-0.025f) }
        routeDownButton.setOnClickListener { overlayView.adjustRouteVertical(0.025f) }
        routeNarrowButton.setOnClickListener { overlayView.adjustRouteWidth(-0.08f) }
        routeWideButton.setOnClickListener { overlayView.adjustRouteWidth(0.08f) }
        pitchUpButton.setOnClickListener { overlayView.adjustCameraPitch(0.5f) }
        pitchDownButton.setOnClickListener { overlayView.adjustCameraPitch(-0.5f) }
        fovWideButton.setOnClickListener { overlayView.adjustCameraFov(2f) }
        fovNarrowButton.setOnClickListener { overlayView.adjustCameraFov(-2f) }
        cameraHeightUpButton.setOnClickListener { overlayView.adjustCameraHeight(0.05f) }
        cameraHeightDownButton.setOnClickListener { overlayView.adjustCameraHeight(-0.05f) }
        cameraLeftButton.setOnClickListener { overlayView.adjustCameraLateralOffset(-0.02f) }
        cameraRightButton.setOnClickListener { overlayView.adjustCameraLateralOffset(0.02f) }
        calibrationResetButton.setOnClickListener { overlayView.resetCameraCalibration() }
        frameThinButton.setOnClickListener { overlayView.adjustStatusFrameWidth(-4f) }
        frameThickButton.setOnClickListener { overlayView.adjustStatusFrameWidth(4f) }

        val prefs = getSharedPreferences("sound_settings", MODE_PRIVATE)
        stopSoundEnabled = prefs.getBoolean("stopSoundEnabled", true)
        goSoundEnabled = prefs.getBoolean("goSoundEnabled", true)
        stopSoundSwitch.isChecked = stopSoundEnabled
        goSoundSwitch.isChecked = goSoundEnabled
        stopSoundSwitch.setOnCheckedChangeListener { _, isChecked ->
            stopSoundEnabled = isChecked
            prefs.edit().putBoolean("stopSoundEnabled", isChecked).apply()
        }
        goSoundSwitch.setOnCheckedChangeListener { _, isChecked ->
            goSoundEnabled = isChecked
            prefs.edit().putBoolean("goSoundEnabled", isChecked).apply()
        }
    }
    
    private fun updateDebugText(message: String) {
        runOnUiThread {
            val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
            debugText.text = "[$timestamp] $message"
            Log.d("MainActivity", message)
        }
    }

    private fun initComponents() {
        inferenceEngine = InferenceEngine(this)
        stateMachine = StateMachine()
        roiSelector = RoiSelector()
        
        imuManager = ImuManager(this) { tilt, ts -> inferenceEngine.onImuTilt(tilt, ts) }
        speedMonitor = SpeedMonitor(this) { mps -> inferenceEngine.onSpeed(mps) }

        frameAnalyzer = FrameAnalyzer(
            inferenceEngine = inferenceEngine,
            stateMachine = stateMachine,
            roiSelector = roiSelector,
            onResultCallback = ::onAnalysisResult,
            onDebugCallback = ::updateDebugText,
            horizontalFovProvider = {
                cameraInfo?.let { horizontalFovDeg(it, currentZoomRatio) } ?: 72f
            },
            onRotationChanged = { imuManager?.rotationDegrees = it }
        )

        // Pass view dimensions to analyzer once the view is laid out
        overlayView.post {
            frameAnalyzer.setViewDimensions(overlayView.width, overlayView.height)
        }
        
        toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 90)
        
        setupStateObservers()
    }

    private fun setupStateObservers() {
        
            
        stateMachine.shouldAnnounce
            .onEach { shouldAnnounce ->
                if (shouldAnnounce) {
                    announceState(stateMachine.currentState.value)
                    stateMachine.acknowledgeAnnouncement()
                }
            }
            .launchIn(lifecycleScope)
    }

    private fun checkPermissions() {
        when {
            hasCameraPermission() && hasAudioPermission() && speedMonitor?.hasPermission == true -> {
                speedMonitor?.start()
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.ACCESS_FINE_LOCATION
                    )
                )
            }
        }
    }

    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        updateDebugText("開始啟動相機和 AI 模型...")
        lifecycleScope.launch {
            updateDebugText("正在初始化 AI 模型...")
            val initSuccess = inferenceEngine.initialize()
            if (!initSuccess) {
                updateDebugText("❌ AI 模型載入失敗!")
                Toast.makeText(this@MainActivity, "AI模型載入失敗", Toast.LENGTH_LONG).show()
                return@launch
            }
            updateDebugText("✅ AI 模型載入成功")
            
            updateDebugText("正在啟動相機...")
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this@MainActivity)
            cameraProvider = cameraProviderFuture.get()
            
            bindCameraUseCases()
            updateDebugText("✅ 相機啟動完成，開始檢測...")
        }
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return
        
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        
        imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, frameAnalyzer)
            }
        
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        try {
            cameraProvider.unbindAll()
            
            val camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )
            
            setupZoomControl(camera)
            
        } catch (exc: Exception) {
            Log.e("MainActivity", "Use case binding failed", exc)
            Toast.makeText(this, "相機啟動失敗", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupZoomControl(camera: Camera) {
        cameraInfo = camera.cameraInfo
        cameraControl = camera.cameraControl
        
        lifecycleScope.launch {
            try {
                val zoomState = cameraInfo?.zoomState?.value
                minZoomRatio = zoomState?.minZoomRatio ?: 1f
                maxZoomRatio = zoomState?.maxZoomRatio ?: 1f
                val hasZoom = maxZoomRatio > minZoomRatio
                if (hasZoom) {
                    currentZoomRatio = currentZoomRatio.coerceIn(minZoomRatio, maxZoomRatio)
                    cameraControl?.setZoomRatio(currentZoomRatio)
                    Log.d("MainActivity", "Zoom set to ${currentZoomRatio}x")
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "Failed to set zoom", e)
            }
        }
    }

    private fun adjustZoom(delta: Float) {
        val control = cameraControl ?: return
        if (maxZoomRatio <= minZoomRatio) return
        currentZoomRatio = (currentZoomRatio + delta).coerceIn(minZoomRatio, maxZoomRatio)
        control.setZoomRatio(currentZoomRatio)
        Toast.makeText(this, "Zoom ${String.format("%.1f", currentZoomRatio)}x", Toast.LENGTH_SHORT).show()
    }

    private fun onAnalysisResult(result: FrameAnalyzer.AnalysisResult) {
        runOnUiThread {
            statusText.text = result.currentState
            fpsText.text = "FPS: ${result.fps}"
            updateCalibStatus(result.cameraCalibration)
            
            // 更新所有檢測結果到 overlay
            overlayView.setResults(
                result.detections,
                result.imageWidth,
                result.imageHeight,
                result.imageRotation,
                result.pathPoints,
                result.shouldStop,
                result.shouldGo,
                result.cameraCalibration
            )
            
        }
    }

    private fun updateCalibStatus(cal: com.example.trafficlight.inference.CameraCalibrationEstimate) {
        when {
            cal.valid -> {
                calibStatusText.text = "✓ 校正完成,可用"
                calibStatusText.setTextColor(0xFF8CFC9A.toInt())
            }
            !cal.movingFastEnough -> {
                val kmh = com.example.trafficlight.core.calib.CalibrationFusion.MIN_CALIB_SPEED_KMH.toInt()
                calibStatusText.text = "待機中 ${cal.speedKmh.toInt()} km/h(時速 ≥$kmh 開始校正)"
                calibStatusText.setTextColor(0xFFBBBBBB.toInt())
            }
            else -> {
                val min = com.example.trafficlight.core.calib.CalibrationFusion.MIN_SAMPLES
                calibStatusText.text = "校正中 ${cal.sampleCount}/$min"
                calibStatusText.setTextColor(0xFFFFCC66.toInt())
            }
        }
    }

    private fun announceState(state: TrafficLightState) {
        when (state) {
            TrafficLightState.RED -> if (stopSoundEnabled) playStopTone()
            TrafficLightState.GREEN -> if (goSoundEnabled) playGoTone()
            else -> return
        }
    }

    private fun playStopTone() {
        toneGenerator?.startTone(ToneGenerator.TONE_PROP_NACK, 130)
        previewView.postDelayed({ toneGenerator?.startTone(ToneGenerator.TONE_PROP_NACK, 130) }, 180)
    }

    private fun playGoTone() {
        toneGenerator?.startTone(ToneGenerator.TONE_PROP_ACK, 180)
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        // configChanges 模式:旋轉不重建 Activity,自行換佈局並重綁 view
        setContentView(R.layout.activity_main)
        initViews()
        overlayView.post {
            frameAnalyzer.setViewDimensions(overlayView.width, overlayView.height)
        }
        cameraProvider?.let { bindCameraUseCases() }
        // 切換後靜音 1 秒,避免時序緩衝重置期間亂報
        stateMachine.muteFor(1000L)
    }

    override fun onResume() {
        super.onResume()
        imuManager?.start()
        speedMonitor?.start()
    }

    override fun onPause() {
        super.onPause()
        imuManager?.stop()
        speedMonitor?.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        
        toneGenerator?.release()
        
        inferenceEngine.release()
        cameraExecutor.shutdown()
        
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error unbinding camera", e)
        }
    }
}
