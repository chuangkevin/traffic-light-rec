package com.example.trafficlight.core.plan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PlanParserTest {
    private fun policyVector(vNow: Float, vEverywhereElse: Float, aNow: Float = 0f): FloatArray {
        val data = FloatArray(33 * 15)
        for (i in 0 until 33) {
            data[i * 15 + 3] = if (i == 0) vNow else vEverywhereElse
            data[i * 15 + 6] = aNow
            data[i * 15] = i.toFloat()      // path x
            data[i * 15 + 1] = 0f           // path y
        }
        return data
    }

    @Test fun stoppedPlanYieldsStop() {
        val plan = parseDrivingPlan(policyVector(vNow = 0f, vEverywhereElse = 0f))
        assertTrue(plan.shouldStop)
        assertEquals(DrivingAction.STOP, plan.action)
    }

    @Test fun acceleratingFromStandstillYieldsGo() {
        val plan = parseDrivingPlan(policyVector(vNow = 0.5f, vEverywhereElse = 5f))
        assertTrue(plan.shouldGo)
        assertFalse(plan.shouldStop)
        assertEquals(DrivingAction.GO, plan.action)
    }

    @Test fun cruisingYieldsHold() {
        val plan = parseDrivingPlan(policyVector(vNow = 15f, vEverywhereElse = 15f))
        assertEquals(DrivingAction.HOLD, plan.action)
    }

    @Test fun tooShortVectorIsHold() {
        assertEquals(DrivingAction.HOLD, parseDrivingPlan(FloatArray(10)).action)
    }

    @Test fun pathFiltersNegativeAndNonFinite() {
        val data = policyVector(15f, 15f)
        data[0] = -1f                      // x < 0 → 濾掉
        data[15] = Float.NaN               // 非有限 → 濾掉
        val plan = parseDrivingPlan(data)
        assertEquals(31, plan.path.size)
    }
}
