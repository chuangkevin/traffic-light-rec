package com.example.trafficlight.core.plan

enum class DrivingAction { STOP, GO, HOLD }

data class PlanPoint(val x: Float, val y: Float)

data class DrivingPlan(
    val shouldStop: Boolean,
    val shouldGo: Boolean,
    val confidence: Float,
    val nearVelocity: Float,
    val futureVelocity: Float,
    val desiredAcceleration: Float,
    val action: DrivingAction,
    val path: List<PlanPoint> = emptyList()
)

private const val PLAN_VALUES = 33 * 15
private const val PLAN_WIDTH = 15
private const val PLAN_ACCELERATION_X = 6
private const val PLAN_VELOCITY_X = 3
private const val MIN_STABLE_DELAY_S = 0.3f
private const val MODEL_ACTION_T_S = 0.075f
private const val STOPPING_VELOCITY_MPS = 0.3f
private const val GO_ACCELERATION_MPS2 = 0.45f
private const val GO_VELOCITY_DELTA_MPS = 0.35f

val EMPTY_PLAN = DrivingPlan(false, false, 0f, 0f, 0f, 0f, DrivingAction.HOLD)

fun parseDrivingPlan(policyData: FloatArray): DrivingPlan {
    if (policyData.size < PLAN_VALUES) return EMPTY_PLAN

    val nearVelocity = policyData[PLAN_VELOCITY_X]
    val futureVelocity = policyData[16 * PLAN_WIDTH + PLAN_VELOCITY_X]
    val path = (0 until 33).map { i ->
        PlanPoint(policyData[i * PLAN_WIDTH], policyData[i * PLAN_WIDTH + 1])
    }.filter { it.x.isFinite() && it.y.isFinite() && it.x >= 0f }
    val accelerationNow = policyData[PLAN_ACCELERATION_X]
    val desiredAcceleration = desiredAcceleration(policyData, nearVelocity, accelerationNow)
    val shouldStop = nearVelocity < STOPPING_VELOCITY_MPS && desiredAcceleration < 0.1f
    val velocityDelta = futureVelocity - nearVelocity
    val shouldGo = !shouldStop && nearVelocity < 1.5f &&
        desiredAcceleration > GO_ACCELERATION_MPS2 && velocityDelta > GO_VELOCITY_DELTA_MPS
    val action = when {
        shouldStop -> DrivingAction.STOP
        shouldGo -> DrivingAction.GO
        else -> DrivingAction.HOLD
    }
    val confidence = when (action) {
        DrivingAction.STOP -> ((0.1f - desiredAcceleration) / 1.6f).coerceIn(0.65f, 1f)
        DrivingAction.GO -> ((desiredAcceleration - GO_ACCELERATION_MPS2) / 1.8f).coerceIn(0.65f, 1f)
        DrivingAction.HOLD -> 0.2f
    }
    return DrivingPlan(shouldStop, shouldGo, confidence, nearVelocity, futureVelocity,
        desiredAcceleration, action, path)
}

private fun desiredAcceleration(policyData: FloatArray, vNow: Float, aNow: Float): Float {
    val stableTargetVelocity = interpolatePlanVelocity(policyData, MIN_STABLE_DELAY_S)
    val vTarget = vNow + (MODEL_ACTION_T_S / MIN_STABLE_DELAY_S) * (stableTargetVelocity - vNow)
    return 2f * (vTarget - vNow) / MODEL_ACTION_T_S - aNow
}

private fun interpolatePlanVelocity(policyData: FloatArray, targetTimeS: Float): Float {
    var previousTime = 0f
    var previousVelocity = policyData[PLAN_VELOCITY_X]
    for (i in 1 until 33) {
        val time = 10f * (i / 32f) * (i / 32f)
        val velocity = policyData[i * PLAN_WIDTH + PLAN_VELOCITY_X]
        if (targetTimeS <= time) {
            val ratio = ((targetTimeS - previousTime) / (time - previousTime)).coerceIn(0f, 1f)
            return previousVelocity + (velocity - previousVelocity) * ratio
        }
        previousTime = time
        previousVelocity = velocity
    }
    return previousVelocity
}
