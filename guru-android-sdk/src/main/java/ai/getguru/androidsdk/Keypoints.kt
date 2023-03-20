package ai.getguru.androidsdk

import android.util.Pair
import com.google.common.collect.ForwardingList
import org.pytorch.IValue
import java.util.stream.Collectors

class Keypoints private constructor(private val keypoints: List<Keypoint>) :
    ForwardingList<Keypoint>() {

    override fun delegate(): List<Keypoint> {
        return keypoints
    }
    fun getPairs(): List<Pair<Keypoint, Keypoint>> {
        return pairIndices.stream()
            .map { indices: Pair<Int, Int> ->
                Pair.create(
                    keypoints[indices.first], keypoints[indices.second]
                )
            }
            .collect(Collectors.toList())
    }

    operator fun get(landmark: InferenceLandmark): Keypoint? {
       return keypoints[landmark.cocoIndex()]
    }

    companion object {
        private val keypointNames = listOf(
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
            "right_ankle"
        )
        private val keypointPairs = arrayOf(
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
            arrayOf("right_elbow", "right_wrist")
        )
        private val pairIndices = keypointPairs.map { ab: Array<String> ->
            Pair.create(
                keypointNames.indexOf(ab[0]),
                keypointNames.indexOf(ab[1])
            )
        }.toList();

        fun of(keypoints: List<Keypoint>): Keypoints {
            return Keypoints(keypoints)
        }

        fun parse(rawModelOutput: IValue): Keypoints {
            val locationsAndScores = rawModelOutput.toTuple()
            val locations = locationsAndScores[0].toTensor().dataAsLongArray
            val scores = locationsAndScores[1].toTensor().dataAsFloatArray
            val output: MutableList<Keypoint> = ArrayList()
            var i = 0
            while (i < locations.size) {
                val x = locations[i].toInt()
                val y = locations[i + 1].toInt()
                val score = scores[Math.floorDiv(i, 2)]
                output.add(Keypoint(x = x.toDouble(), y = y.toDouble(), score = score.toDouble()))
                i += 2
            }
            return Keypoints(output)
        }
    }
}