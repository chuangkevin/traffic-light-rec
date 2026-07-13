package com.example.trafficlight.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.example.trafficlight.core.calib.Tilt
import com.example.trafficlight.core.calib.tiltFromGravity

/** 訂閱重力感測器,轉成 core 的 Tilt 樣本。rotationDegrees 由外部(相機幀)提供。 */
class ImuManager(
    context: Context,
    private val onTilt: (Tilt, Long) -> Unit
) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gravitySensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    @Volatile var rotationDegrees: Int = 0

    val available: Boolean get() = gravitySensor != null

    fun start() {
        gravitySensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        val tilt = tiltFromGravity(event.values[0], event.values[1], event.values[2], rotationDegrees)
        onTilt(tilt, event.timestamp / 1_000_000L)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
