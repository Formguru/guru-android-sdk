package ai.getguru.androidsdk

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.time.Instant
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock

class GuruVideoImpl constructor(
    private val apiKey: String,
    private val context: Context,
    private val domain: String,
    private val activity: String,
    private val analysisPerSecond: Int = 8,
) : GuruVideo {

    private val LOG_TAG = "GuruVideoImpl"
    private val JSON_MEDIA_TYPE: MediaType = "application/json; charset=utf-8".toMediaType()
    private var previousFrameInference: FrameInference? = null
    private var latestAnalysis: Analysis? = null
    private var frameIndex: AtomicInteger = AtomicInteger(-1)
    private val inferenceLock = ReentrantLock()
    private var poseEstimator: PoseEstimator? = null
    private var videoId: String? = null
    private var startedAt: Instant? = null
    private var analysisClient: AnalysisClient? = null
    private val httpClient = OkHttpClient()
    private val analysisScope = CoroutineScope(context = Dispatchers.IO)

    companion object {
        suspend fun create(domain: String, activity: String, apiKey: String, context: Context): GuruVideoImpl {
            return GuruVideoImpl(apiKey, context, domain, activity).also {
                it.init()
            }
        }
    }

    override suspend fun newFrame(frame: Image): FrameInference {
        val newFrameIndex = frameIndex.incrementAndGet()
        if (!inferenceLock.tryLock()) {
            return previousFrameInference(newFrameIndex)
        }

        return runInference(imageToBitmap(frame), newFrameIndex)
    }

    override suspend fun newFrame(frame: Bitmap): FrameInference {
        val newFrameIndex = frameIndex.incrementAndGet()
        if (!inferenceLock.tryLock()) {
            return previousFrameInference(newFrameIndex)
        }

        return runInference(frame, newFrameIndex)
    }

    override suspend fun finish(): Analysis {
        if (analysisClient == null) {
            return emptyAnalysis()
        }
        else {
            analysisClient!!.waitUntilQuiet()
            return analysisClient!!.flush() ?: emptyAnalysis()
        }
    }

    private suspend fun createVideo(): String {
        val gson = Gson()
        val requestJson = gson.toJson(mapOf(
            "domain" to domain,
            "activity" to activity,
            "inference" to "local",
            "resolutionWidth" to 480,
            "resolutionHeight" to 640,
        ))

        val request: Request = Request.Builder()
            .url("https://api.getguru.fitness/videos")
            .header("Content-Type", "application/json")
            .header("x-api-key", apiKey)
            .post(requestJson.toRequestBody(JSON_MEDIA_TYPE))
            .build()

        return withContext(Dispatchers.IO) {
            val createVideoResponse = httpClient.newCall(request).execute().use { response ->
                response.body!!.string()
            }

            val mapType = object : TypeToken<Map<String, Any>>() {}.type
            val createVideoJson: Map<String, Any> = gson.fromJson(createVideoResponse, mapType)
            createVideoJson["id"] as String
        }
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

    private fun previousFrameInference(newFrameIndex: Int, analysis: Analysis? = null): FrameInference {
        val analysis = analysis ?: latestAnalysis ?: emptyAnalysis()
        if (previousFrameInference == null) {
            return FrameInference(
                keypoints = HashMap<Int, Keypoint>(),
                previousFrame = null,
                frameIndex = newFrameIndex,
                secondsSinceStart = 0.0,
                analysis = analysis,
            )
        }
        else {
            return FrameInference(
                keypoints = previousFrameInference!!.keypoints,
                previousFrame = previousFrameInference!!.previousFrame,
                frameIndex = newFrameIndex,
                secondsSinceStart = previousFrameInference!!.secondsSinceStart,
                analysis = analysis,
            )
        }
    }

    private suspend fun runInference(frame: Bitmap, newFrameIndex: Int): FrameInference {
        if (startedAt == null) {
            startedAt = Instant.now()
            videoId = createVideo()
            analysisClient = AnalysisClient(videoId!!, apiKey, analysisPerSecond)
        }

        val frameTimestamp = Instant.now()
        try {
            val keypoints = poseEstimator!!.estimatePose(frame)
            val keypointMap = HashMap<Int, Keypoint>()
            keypoints.forEachIndexed { index: Int, keypoint: Keypoint? ->
                keypointMap[index] = keypoint!!
            }

            val newInference = FrameInference(
                keypoints = keypointMap,
                previousFrame = previousFrameInference,
                frameIndex = newFrameIndex,
                secondsSinceStart = (
                        frameTimestamp.toEpochMilli() - startedAt!!.toEpochMilli()
                        ) / 1000.0,
                analysis = latestAnalysis ?: emptyAnalysis()
            )

            analysisScope.launch {
                val newAnalysis = analysisClient!!.add(newInference)
                if (newAnalysis != null) {
                    latestAnalysis = newAnalysis
                }
            }

            previousFrameInference = newInference
        } finally {
            inferenceLock.unlock()
        }

        return previousFrameInference!!
    }

    private fun emptyAnalysis() = Analysis(null, emptyList())
}