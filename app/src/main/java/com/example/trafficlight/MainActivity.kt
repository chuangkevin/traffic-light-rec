package com.example.trafficlight

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
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
import com.example.trafficlight.inference.InferenceEngine
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

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var fpsText: TextView
    private lateinit var debugText: TextView
    
    private lateinit var inferenceEngine: InferenceEngine
    private lateinit var stateMachine: StateMachine
    private lateinit var roiSelector: RoiSelector
    private lateinit var frameAnalyzer: FrameAnalyzer
    
    private var textToSpeech: TextToSpeech? = null
    private var isTtsReady = false
    
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] == true
        val audioGranted = permissions[Manifest.permission.RECORD_AUDIO] == true
        
        if (cameraGranted && audioGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "需要相機和音頻權限", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initViews()
        initComponents()
        checkPermissions()
        
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun initViews() {
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        fpsText = findViewById(R.id.fpsText)
        debugText = findViewById(R.id.debugText)
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
        
        frameAnalyzer = FrameAnalyzer(
            inferenceEngine = inferenceEngine,
            stateMachine = stateMachine,
            roiSelector = roiSelector,
            onResultCallback = ::onAnalysisResult,
            onDebugCallback = ::updateDebugText
        )
        
        textToSpeech = TextToSpeech(this, this)
        
        setupStateObservers()
    }

    private fun setupStateObservers() {
        stateMachine.currentState
            .onEach { state ->
                overlayView.updateState(state, stateMachine.getStateConfidence())
                overlayView.updateRoi(roiSelector.getCurrentRoi())
                overlayView.animateStateChange(state)
            }
            .launchIn(lifecycleScope)
            
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
            hasCameraPermission() && hasAudioPermission() -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.RECORD_AUDIO
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
        val cameraInfo = camera.cameraInfo
        val cameraControl = camera.cameraControl
        
        lifecycleScope.launch {
            try {
                val hasZoom = cameraInfo.zoomState.value?.maxZoomRatio ?: 1f > 1f
                if (hasZoom) {
                    cameraControl.setZoomRatio(2.0f)
                    Log.d("MainActivity", "Zoom set to 2x for better traffic light detection")
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "Failed to set zoom", e)
            }
        }
    }

    private fun onAnalysisResult(result: FrameAnalyzer.AnalysisResult) {
        runOnUiThread {
            statusText.text = "${result.currentState} (${(result.confidence * 100).toInt()}%)"
            fpsText.text = "FPS: ${result.fps}"
            
            Log.d("Analysis", "${result.debugInfo}")
        }
    }

    private fun announceState(state: TrafficLightState) {
        if (!isTtsReady) return
        
        val announcement = when (state) {
            TrafficLightState.RED -> "紅燈"
            TrafficLightState.GREEN -> "綠燈" 
            TrafficLightState.YELLOW -> "黃燈"
            else -> return
        }
        
        textToSpeech?.speak(
            announcement,
            TextToSpeech.QUEUE_FLUSH,
            null,
            "traffic_light_$state"
        )
        
        overlayView.showDetectionPulse()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = textToSpeech?.setLanguage(Locale.TRADITIONAL_CHINESE)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                textToSpeech?.setLanguage(Locale.CHINESE)
            }
            isTtsReady = true
            Log.d("MainActivity", "TTS initialized successfully")
        } else {
            Log.e("MainActivity", "TTS initialization failed")
        }
    }

    override fun onResume() {
        super.onResume()
        if (::frameAnalyzer.isInitialized) {
        }
    }

    override fun onPause() {
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        
        inferenceEngine.release()
        cameraExecutor.shutdown()
        
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error unbinding camera", e)
        }
    }
}