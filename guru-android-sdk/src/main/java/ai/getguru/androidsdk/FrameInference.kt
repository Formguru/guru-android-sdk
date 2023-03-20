package ai.getguru.androidsdk

class FrameInference constructor(
    val keypoints: Map<Int, Keypoint>,
    val previousFrame: FrameInference?,
    val frameIndex: Int,
    val secondsSinceStart: Double,
    val analysis: Analysis,
    smoother: KeypointsFilter? = null,
) {

    val smoothedKeypoints: Keypoints? = smoother?.smooth(this.skeleton())

    @Deprecated("Use skeleton().getPairs()")
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

    @Deprecated("Use skeleton() to access keypoints")
    val smoothKeypoints: Map<Int, Keypoint> by lazy {
        smoothedKeypoints?.mapIndexed{ i, k -> i to k }?.toMap() ?: keypoints
    }

    fun skeleton(disableSmoothing: Boolean = false): Keypoints {
        return if (disableSmoothing || smoothedKeypoints == null) {
            val orderedKeypoints = keypoints.keys.toList().sorted().map { i -> keypoints[i]!! }
            Keypoints.of(orderedKeypoints.toList())
        } else {
            smoothedKeypoints
        }
    }

    fun keypointForLandmark(landmark: InferenceLandmark, disableSmoothing: Boolean = false): Keypoint? {
        return if (disableSmoothing || smoothedKeypoints == null) {
            keypoints[landmark.cocoIndex()]
        } else {
            smoothedKeypoints[landmark]
        }
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
}