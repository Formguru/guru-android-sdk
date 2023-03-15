package ai.getguru.androidsdk

import kotlin.collections.HashMap

class FrameInference constructor(
    val keypoints: Map<Int, Keypoint>,
    val previousFrame: FrameInference?,
    val frameIndex: Int,
    val secondsSinceStart: Double,
    val analysis: Analysis,
) {

    val cocoPairs = arrayOf(
        arrayOf("left_shoulder", "right_shoulder"),
        arrayOf("left_shoulder", "left_hip"),
        arrayOf("left_hip", "left_knee"),
        arrayOf("left_knee", "left_ankle"),
        arrayOf("right_shoulder", "right_hip"),
        arrayOf("right_hip", "right_knee"),
        arrayOf("right_knee", "right_ankle"),
        arrayOf("left_hip", "right_hip"),
        arrayOf("left_shoulder", "left_elbow"),
        arrayOf("left_elbow", "left_wrist"),
        arrayOf("right_shoulder", "right_elbow"),
        arrayOf("right_elbow", "right_wrist"),
    )

    val smoothKeypoints: Map<Int, Keypoint> by lazy {
        if (previousFrame?.smoothKeypoints == null) {
            keypoints
        } else {
            buildSmoothedKeypoints()
        }
    }

    fun skeleton(useSmoothing: Boolean): Keypoints {
        val kpts = if (useSmoothing) smoothKeypoints else keypoints
        val orderedKeypoints = kpts.keys.toList().sorted().map { i -> kpts[i]!! }
        return Keypoints.of(orderedKeypoints.toList())
    }

    fun keypointForLandmark(landmark: InferenceLandmark): Keypoint? {
        return smoothKeypoints[landmark.cocoIndex()]
    }

    fun userFacing(): UserFacing {
        val nose = keypointForLandmark(InferenceLandmark.NOSE)
        if (nose == null) {
            return UserFacing.OTHER
        }
        else {
            val leftEar = keypointForLandmark(InferenceLandmark.LEFT_EAR)
            val rightEar = keypointForLandmark(InferenceLandmark.RIGHT_EAR)
            return if (leftEar != null && rightEar != null) {
                if ((nose.x < leftEar.x) && (nose.x > rightEar.x)) {
                    UserFacing.TOWARD
                } else if ((nose.x < leftEar.x) && (nose.x < rightEar.x)) {
                    UserFacing.RIGHT
                } else if ((nose.x > leftEar.x) && (nose.x > rightEar.x)) {
                    UserFacing.LEFT
                } else {
                    UserFacing.OTHER
                }
            } else {
                UserFacing.OTHER
            }
        }
    }

    private fun buildSmoothedKeypoints() : Map<Int, Keypoint> {
        val currentFrameWeight = 0.5

        val smoothedKeypoints = HashMap<Int, Keypoint>()
        for (nextLandmark in InferenceLandmark.values()) {
            val previousKeypoint = previousFrame!!.keypointForLandmark(nextLandmark)
            val landmarkIndex = nextLandmark.cocoIndex()
            val currentKeypoint = keypoints[landmarkIndex]

            val minScore = 0.01
            if (previousKeypoint == null || previousKeypoint.score < minScore) {
                smoothedKeypoints[landmarkIndex] = currentKeypoint!!
            } else if (currentKeypoint == null || currentKeypoint.score < minScore) {
                smoothedKeypoints[landmarkIndex] = previousKeypoint
            } else {
                smoothedKeypoints[landmarkIndex] = Keypoint(
                    x = (1 - currentFrameWeight) * previousKeypoint.x + currentFrameWeight * currentKeypoint.x,
                    y = (1 - currentFrameWeight) * previousKeypoint.y + currentFrameWeight * currentKeypoint.y,
                    score = (1 - currentFrameWeight) * previousKeypoint.score + currentFrameWeight * currentKeypoint.score
                )
            }
        }

        return smoothedKeypoints
    }
}