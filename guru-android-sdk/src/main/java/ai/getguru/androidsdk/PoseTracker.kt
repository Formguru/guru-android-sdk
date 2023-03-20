package ai.getguru.androidsdk

import ai.getguru.androidsdk.ImageUtils.toBitmap
import android.content.Context
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
    }

    private suspend fun init() {
        val modelStore = ModelStore(apiKey, context)
        poseEstimator = modelStore.getPoseEstimator()
    }

    fun newFrame(frame: Image, rotationDegrees: Int): Keypoints? {
        return newFrame(frame.toBitmap(rotationDegrees))
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
                BoundingBox.fromPreviousFrame(previousKeypoints!!, frame.width, frame.height)
            }

            val keypoints = poseEstimator!!.estimatePose(frame, bbox)
            previousKeypoints = smoother?.smooth(keypoints) ?: keypoints
        } finally {
            inferenceLock.unlock()
        }

        return previousKeypoints!!
    }
}