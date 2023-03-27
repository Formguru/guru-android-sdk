package ai.getguru.androidsdk

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.media.Image
import java.util.concurrent.locks.ReentrantLock

class PoseTracker constructor(
    private val apiKey: String,
    private val context: Context,
    private val smoother: KeypointsFilter? = null,
) {

    private var poseEstimator: IPoseEstimator? = null
    private var previousKeypoints: Keypoints? = null
    private val inferenceLock = ReentrantLock()

    companion object {
        suspend fun create(apiKey: String, context: Context, isSmoothingEnabled: Boolean = true): PoseTracker {
            val smoother: KeypointsFilter? = if (isSmoothingEnabled) KeypointsFilter() else null
            return PoseTracker(apiKey, context, smoother).also {
                it.init()
            }
        }

        fun createWithModel(apiKey: String, context: Context, model: String, assetManager: AssetManager, isSmoothingEnabled: Boolean = true): PoseTracker {
            val smoother: KeypointsFilter? = if (isSmoothingEnabled) KeypointsFilter() else null
            return PoseTracker(apiKey, context, smoother).also {
                it.init(model, assetManager)
            }
        }
    }

    private suspend fun init() {
        val modelStore = ModelStore(apiKey, context)
        poseEstimator = modelStore.getPoseEstimator()
    }

    private fun init(modelPath: String, assetManager: AssetManager) {
        val modelStore = ModelStore(apiKey, context)
        poseEstimator = modelStore.getPoseEstimator(modelPath, assetManager)
    }

    fun newFrame(frame: Image, rotationDegrees: Int): Keypoints? {
        if (!inferenceLock.tryLock()) {
            return previousKeypoints
        }
        try {
            val bbox: BoundingBox? = if (previousKeypoints == null) {
                null
            } else {
                BoundingBox.fromPreviousFrame(previousKeypoints!!)
            }

            val keypoints = poseEstimator!!.estimatePose(frame, rotationDegrees, bbox)
            previousKeypoints = smoother?.smooth(keypoints) ?: keypoints
        } finally {
            inferenceLock.unlock()
        }

        return previousKeypoints!!
    }

    fun newFrame(frame: Bitmap): Keypoints? {
        if (frame.width != GuruVideoImpl.INPUT_RESOLUTION_WIDTH || frame.height != GuruVideoImpl.INPUT_RESOLUTION_HEIGHT) {
            throw IllegalArgumentException("Camera input must have resolution of width=${GuruVideoImpl.INPUT_RESOLUTION_WIDTH}, height=${GuruVideoImpl.INPUT_RESOLUTION_HEIGHT}")
        }
        if (!inferenceLock.tryLock()) {
            return previousKeypoints
        }

        return runInference(frame)
    }

    private fun runInference(frame: Bitmap): Keypoints {
        try {
            val bbox: BoundingBox? = if (previousKeypoints == null) {
                null
            } else {
                BoundingBox.fromPreviousFrame(previousKeypoints!!)
            }

            val keypoints = poseEstimator!!.estimatePose(frame, bbox)
            previousKeypoints = smoother?.smooth(keypoints) ?: keypoints
        } finally {
            inferenceLock.unlock()
        }

        return previousKeypoints!!
    }
}