package ai.getguru.androidsdk

import ai.getguru.androidsdk.ImageUtils.toNv21Buffer
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.media.Image
import android.os.Build
import java.nio.file.Path
import kotlin.io.path.absolutePathString

class OnnxPoseEstimator : IPoseEstimator {

    companion object {
        init {
            System.loadLibrary("guru_pose_inference")
        }

        private fun getNumberOfCores(): Int {
            return if (Build.VERSION.SDK_INT >= 17) {
                Runtime.getRuntime().availableProcessors()
            } else 4
        }

        fun withModel(modelFile: Path): OnnxPoseEstimator {
            val estimator = OnnxPoseEstimator()
            estimator.initFromFile(modelFile.absolutePathString(), getNumberOfCores())
            return estimator
        }

        fun withModel(modelPath: String, assetManager: AssetManager): OnnxPoseEstimator {
            val estimator = OnnxPoseEstimator()
            estimator.initFromAsset(modelPath, assetManager, getNumberOfCores())
            return estimator
        }
    }

    external fun initFromFile(path: String, numProcessors: Int)
    external fun initFromAsset(path: String, mgr: AssetManager, numProcessors: Int)
    external fun detectPose(bitmap: Bitmap, boundingBox: BoundingBox?): FloatArray
    external fun detectPoseFromNv21(nv21: ByteArray, width: Int, height: Int, rotation: Int, boundingBox: BoundingBox?): FloatArray


    override fun estimatePose(bitmap: Bitmap): Keypoints {
        val defaultBbox = BoundingBox(0f, 0f, 1f, 1f)
        return estimatePose(bitmap, defaultBbox)
    }

    override fun estimatePose(img: Image, rotation: Int, bbox: BoundingBox?): Keypoints {
        val nv21Bytes = img.toNv21Buffer()
        val result: FloatArray = detectPoseFromNv21(
            nv21Bytes,
            img.width,
            img.height,
            rotation,
            bbox ?: BoundingBox(0f, 0f, 1f, 1f)
        )
        val numKeypoints = result.size / 3
        val keypoints = mutableListOf<Keypoint>()
        for (k in 0 until numKeypoints) {
            val x = result[k * 3]
            val y = result[k * 3 + 1]
            val score = result[k * 3 + 2]
            keypoints.add(Keypoint(x.toDouble(), y.toDouble(), score.toDouble()))
        }
        return Keypoints.of(keypoints)
    }

    override fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints {
        val bbox = boundingBox ?: BoundingBox(0f, 0f, 1f, 1f)
        val result: FloatArray = detectPose(bitmap, bbox)
        val K = result.size / 3
        val keypoints = mutableListOf<Keypoint>()
        for (k in 0 until K) {
            val x = result[k * 3]
            val y = result[k * 3 + 1]
            val score = result[k * 3 + 2]
            keypoints.add(Keypoint(x.toDouble(), y.toDouble(), score.toDouble()))
        }
        return Keypoints.of(keypoints)
    }
}
