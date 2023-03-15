package ai.getguru.androidsdk

import android.graphics.Bitmap
import java.io.File

class NcnnPoseEstimator(paramFile: File, binFile: File) : IPoseEstimator {

    companion object {
        init {
            System.loadLibrary("guru_pose_inference")
        }
    }

    init {
        initModel(paramFile.absolutePath, binFile.absolutePath)
    }

    external fun initModel(paramPath: String, binPath: String)
    external fun detectPose(bitmap: Bitmap): FloatArray

    override fun estimatePose(bitmap: Bitmap): Keypoints {
        return estimatePose(bitmap, null)
    }

    override fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints {
        // TODO: we don't actually use the bounding-box, yet
        val keypoints: FloatArray = detectPose(bitmap)
        val results: MutableList<Keypoint> = mutableListOf()
        for (i in 0 until keypoints.size / 3) {
            val x = keypoints[3 * i]
            val y = keypoints[3 * i + 1]
            val score = keypoints[3 * i + 2]
            results.add(Keypoint(x.toDouble(), y.toDouble(), score.toDouble()))
        }
        return Keypoints.of(results)
    }

}
