package ai.getguru.androidsdk

import kotlin.math.min
import kotlin.math.max
import kotlin.math.roundToInt

class BoundingBox constructor(
    val x1: Int,
    val y1: Int,
    val x2: Int,
    val y2: Int
) {
    companion object {

        fun fromPreviousFrame(prevKeypoints: Keypoints, width: Int, height: Int, minScore: Float = 0.2f): BoundingBox? {
            val requiredLandmarks = listOf(
                InferenceLandmark.LEFT_SHOULDER,
                InferenceLandmark.RIGHT_SHOULDER,
                InferenceLandmark.LEFT_KNEE,
                InferenceLandmark.RIGHT_KNEE,
            )

            val isMostOfBodyVisible = requiredLandmarks.all {
                prevKeypoints[it]!!.score >= minScore
            }
            if (!isMostOfBodyVisible) {
                return null
            }

            val goodKeypoints = prevKeypoints.filter { it.score >= minScore }
            if (goodKeypoints.isEmpty()) {
                return null
            }
            val minX = goodKeypoints.minOf { it.x } * width
            val maxX = goodKeypoints.maxOf { it.x } * width
            val minY = goodKeypoints.minOf { it.y } * height
            val maxY = goodKeypoints.maxOf { it.y } * height

            val padTop = .1 * (maxY - minY)
            val padBottom = .2 * (maxY - minY)
            val padSides = .2 * (maxX - minX)

            fun clampX(a: Double): Int {
                return max(min(width.toDouble(), a), 0.0).roundToInt()
            }
            fun clampY(a: Double): Int {
                return max(min(height.toDouble(), a), 0.0).roundToInt()
            }
            return BoundingBox(
                clampX(minX - padSides),
                clampY(minY - padTop),
                clampX(maxX + padSides),
                clampY(maxY + padBottom),
            )
        }
    }
}