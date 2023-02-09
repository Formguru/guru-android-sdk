package ai.getguru.androidsdk

import com.google.common.collect.ForwardingList
import org.pytorch.IValue

class Keypoints private constructor(private val keypoints: List<Keypoint>) :
    ForwardingList<Keypoint?>() {
    override fun delegate(): List<Keypoint?> {
        return keypoints
    }

    companion object {
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