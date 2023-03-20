package ai.getguru.androidsdk

import kotlin.math.PI
import kotlin.math.abs


class KeypointsFilter {
    private var filters: MutableMap<Int, OneEuroFilter2D?> = mutableMapOf()

    fun smooth(keypoints: Keypoints): Keypoints {
        val smoothed: Map<Int, Keypoint> = keypoints.mapIndexed { i, kpt ->
            val filter = filters[i] ?: OneEuroFilter2D()
            filters[i] = filter
            val (x, y) = filter.update(kpt.x, kpt.y)
            Pair(i, Keypoint(x, y, kpt.score))
        }.toMap()
        val sortedIndices = smoothed.keys.sorted()
        return Keypoints.of(sortedIndices.map { smoothed[it]!! }.toList())
    }
}

private fun smoothingFactor(te: Double, cutoff: Double): Double {
    val tau = 1.0 / (2 * PI * cutoff)
    return 1.0 / (1 + (tau / te))
}

private fun exponentialSmoothing(a: Double, x: Double, xPrev: Double): Double {
    return a * x + (1 - a) * xPrev
}

// https://hal.inria.fr/hal-00670496/document
private class OneEuroFilter2D(
    private var minCutoff: Double = 0.3,
    private var beta: Double = 20.0,
    private var dCutoff: Double = 1.0
) {

    private var xPrev: Double? = null
    private var yPrev: Double? = null
    private var dxPrev = 0.0
    private var dyPrev = 0.0
    private var tPrev = System.currentTimeMillis() / 1000.0

    fun update(x: Double, y: Double, timestamp: Double? = null): Pair<Double, Double> {
        val t = timestamp ?: (System.currentTimeMillis() / 1000.0)
        val tDelta = t - tPrev

        // Compute filtered derivative
        val aD = smoothingFactor(tDelta, dCutoff)
        val dx = if (tDelta > 0) (x - (xPrev ?: x)) / tDelta else 0.0
        val dy = if (tDelta > 0) (y - (yPrev ?: y)) / tDelta else 0.0
        val dxHat = exponentialSmoothing(aD, dx, dxPrev)
        val dyHat = exponentialSmoothing(aD, dy, dyPrev)

        // Compute filtered signal
        val cutoffX = minCutoff + beta * abs(dxHat)
        val cutoffY = minCutoff + beta * abs(dyHat)
        val alphaX = smoothingFactor(tDelta, cutoffX)
        val alphaY = smoothingFactor(tDelta, cutoffY)
        val xHat = exponentialSmoothing(alphaX, x, xPrev ?: x)
        val yHat = exponentialSmoothing(alphaY, y, yPrev ?: y)
        assert(!xHat.isNaN() && !yHat.isNaN())

        xPrev = xHat
        yPrev = yHat
        dxPrev = dxHat
        dyPrev = dyHat
        tPrev = t

        return Pair(xHat, yHat)
    }
}