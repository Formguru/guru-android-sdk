package ai.getguru.androidsdk

enum class InferenceLandmark(val value: String) {
    LEFT_EYE("left_eye"),
    RIGHT_EYE("right_eye"),
    LEFT_EAR("left_ear"),
    RIGHT_EAR("right_ear"),
    NOSE("nose"),
    LEFT_SHOULDER("left_shoulder"),
    RIGHT_SHOULDER("right_shoulder"),
    LEFT_ELBOW("left_elbow"),
    RIGHT_ELBOW("right_elbow"),
    LEFT_WRIST("left_wrist"),
    RIGHT_WRIST("right_wrist"),
    LEFT_HIP("left_hip"),
    RIGHT_HIP("right_hip"),
    LEFT_KNEE("left_knee"),
    RIGHT_KNEE("right_knee"),
    LEFT_ANKLE("left_ankle"),
    RIGHT_ANKLE("right_ankle");

    private val cocoKeypoints = arrayOf(
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
    private val cocoLabelToIdx = cocoKeypoints.associateBy ({ it }, { cocoKeypoints.indexOf(it) } )

    private val snakeCaseRegex = "_[a-zA-Z]".toRegex()
    fun camelCase(): String {
        return snakeCaseRegex.replace(value) {
            it.value.replace("_","").uppercase()
        }
    }

    fun cocoIndex(): Int {
        return cocoLabelToIdx[value]!!
    }
}
