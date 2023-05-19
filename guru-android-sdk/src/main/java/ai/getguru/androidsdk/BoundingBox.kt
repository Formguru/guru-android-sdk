package ai.getguru.androidsdk

import android.graphics.RectF
import kotlin.math.min
import kotlin.math.max
import kotlin.math.roundToInt

class BoundingBox constructor(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    companion object {

        fun fromPreviousFrame(prevKeypoints: Keypoints, minScore: Float = 0.2f): BoundingBox? {
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
            val minX = goodKeypoints.minOf { it.x }
            val maxX = goodKeypoints.maxOf { it.x }
            val minY = goodKeypoints.minOf { it.y }
            val maxY = goodKeypoints.maxOf { it.y }

            val padTop = .1 * (maxY - minY)
            val padBottom = .2 * (maxY - minY)
            val padSides = .2 * (maxX - minX)

            return BoundingBox(
                clamp(minX - padSides),
                clamp(minY - padTop),
                clamp(maxX + padSides),
                clamp(maxY + padBottom),
            )
        }

        fun fromRect(rect: RectF, width: Int, height: Int): BoundingBox {
            return BoundingBox(
                clamp(rect.left.toDouble() / width),
                clamp(rect.top.toDouble() / height),
                clamp(rect.right.toDouble() / width),
                clamp(rect.bottom.toDouble() / height),
            )
        }

        private fun clamp(a: Double): Float {
            return max(min(1.0, a), 0.0).toFloat()
        }
    }

    override fun toString(): String {
        return "BoundingBox(x1=$x1, y1=$y1, x2=$x2, y2=$y2)"
    }
}