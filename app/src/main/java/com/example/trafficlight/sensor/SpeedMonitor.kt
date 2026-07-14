package com.example.trafficlight.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.core.content.ContextCompat

/** GPS 車速監聽,回報 m/s。無權限或無 GPS 時靜默不動作(速度維持 0 → 校正閘門不放行)。 */
class SpeedMonitor(
    private val context: Context,
    private val onSpeed: (Float) -> Unit
) : LocationListener {
    private val locationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    val hasPermission: Boolean
        get() = ContextCompat.checkSelfPermission(
            context, android.Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    fun start() {
        if (!hasPermission) return
        if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) return
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, this)
    }

    fun stop() {
        locationManager.removeUpdates(this)
    }

    override fun onLocationChanged(location: Location) {
        onSpeed(if (location.hasSpeed()) location.speed else 0f)
    }

    @Deprecated("Deprecated in API 29, still required for API 26-28")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {
        onSpeed(0f)
    }
}
