package ai.getguru.androidsdk

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import java.util.concurrent.locks.ReentrantLock

class GuruVideoImpl constructor(
    val apiKey: String,
    val context: Context,
) : GuruVideo {

    private val LOG_TAG = "GuruVideoImpl"
    var previousFrameInference: FrameInference? = null
    val inferenceLock = ReentrantLock()
    var poseEstimator: PoseEstimator? = null

    companion object {
        suspend fun create(apiKey: String, context: Context): GuruVideoImpl {
            return GuruVideoImpl(apiKey, context).also {
                it.init()
            }
        }
    }

    override fun newFrame(frame: Image): FrameInference {
        if (!inferenceLock.tryLock()) {
            return previousFrameInference ?: FrameInference(
                keypoints = HashMap<Int, Keypoint>(),
                previousFrame = null
            )
        }

        try {
            val bitmapFrame = imageToBitmap(frame)

            val keypoints = poseEstimator!!.estimatePose(bitmapFrame)
            val keypointMap = HashMap<Int, Keypoint>()
            keypoints.forEachIndexed { index: Int, keypoint: Keypoint? ->
                keypointMap[index] = keypoint!!
            }

            previousFrameInference = FrameInference(
                keypoints = keypointMap,
                previousFrame = previousFrameInference
            )
        }
        finally {
            inferenceLock.unlock()
        }

        return previousFrameInference!!
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yuvBytes = arrayOfNulls<ByteArray>(3)
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]!!]
        }

        val yRowStride = planes[0].rowStride
        val uvRowStride = planes[1].rowStride
        val uvPixelStride = planes[1].pixelStride
        val rgbBytes = IntArray(image.width * image.height)
        ImageUtils.convertYUV420ToARGB8888(
            yuvBytes[0]!!,
            yuvBytes[1]!!,
            yuvBytes[2]!!,
            image.width,
            image.height,
            yRowStride,
            uvRowStride,
            uvPixelStride,
            rgbBytes
        )

        val rgbFrameBitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        rgbFrameBitmap?.setPixels(rgbBytes, 0, image.width, 0, 0, image.width, image.height)

        return rgbFrameBitmap
    }

    private suspend fun init() {
        val modelStore = ModelStore(apiKey, context)
        poseEstimator = PoseEstimator.withTorchModel(
            modelStore.getModel()
        )
    }
}