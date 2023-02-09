package ai.getguru.androidsdk

import kotlin.collections.HashMap

class FrameInference constructor(
    val keypoints: Map<Int, Keypoint>,
    val previousFrame: FrameInference?,
) {
    val smoothKeypoints: Map<Int, Keypoint>

    val cocoKeypoints = arrayOf(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    val cocoLabelToIdx = cocoKeypoints.associateBy ({ it }, { cocoKeypoints.indexOf(it) } )

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

    init {
        smoothKeypoints = if (previousFrame?.smoothKeypoints == null) {
            keypoints
        } else {
            buildSmoothedKeypoints()
        }
    }

    fun keypointForLandmark(landmark: InferenceLandmark): Keypoint? {
        return smoothKeypoints[
            cocoLabelToIdx[landmark.value]
        ]
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
        val currentFrameWeight = 0.25

        val smoothedKeypoints = HashMap<Int, Keypoint>()
        for (nextLandmark in InferenceLandmark.values()) {
            val previousKeypoint = previousFrame!!.keypointForLandmark(nextLandmark)
            val landmarkIndex = cocoLabelToIdx[nextLandmark.value]!!
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