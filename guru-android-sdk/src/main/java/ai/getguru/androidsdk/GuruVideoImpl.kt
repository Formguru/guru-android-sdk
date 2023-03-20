package ai.getguru.androidsdk

import ai.getguru.androidsdk.ImageUtils.toBitmap
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
    private val smoother: KeypointsFilter? = null, // TODO: make private
) : GuruVideo {

    private val LOG_TAG = "GuruVideoImpl"
    private val JSON_MEDIA_TYPE: MediaType = "application/json; charset=utf-8".toMediaType()
    private var previousFrameInference: FrameInference? = null
    private var latestAnalysis: Analysis? = null
    private var frameIndex: AtomicInteger = AtomicInteger(-1)
    private val inferenceLock = ReentrantLock()
    private var poseEstimator: IPoseEstimator? = null
    private var videoId: String? = null
    private var startedAt: Instant? = null
    private var analysisClient: AnalysisClient? = null
    private val httpClient = OkHttpClient()
    private val analysisScope = CoroutineScope(context = Dispatchers.IO)


    companion object {
        suspend fun create(domain: String, activity: String, apiKey: String, context: Context): GuruVideoImpl {
            return GuruVideoImpl(apiKey, context, domain, activity, smoother = KeypointsFilter()).also {
                it.init()
            }
        }

        const val INPUT_RESOLUTION_WIDTH = 480
        const val INPUT_RESOLUTION_HEIGHT = 640
    }

    override suspend fun newFrame(frame: Image, rotationDegrees: Int): FrameInference {
        return newFrame(frame.toBitmap(rotationDegrees))
    }

    override suspend fun newFrame(frame: Bitmap): FrameInference {
        if (frame.width != INPUT_RESOLUTION_WIDTH || frame.height != INPUT_RESOLUTION_HEIGHT) {
            throw IllegalArgumentException("Camera input must have resolution of width=${INPUT_RESOLUTION_WIDTH}, height=${INPUT_RESOLUTION_HEIGHT}")
        }
        val newFrameIndex = frameIndex.incrementAndGet()
        if (!inferenceLock.tryLock()) {
            return previousFrameInference(newFrameIndex)
        }

        return runInference(frame, newFrameIndex)
    }

    override suspend fun finish(): Analysis {
        return if (analysisClient == null) {
            emptyAnalysis()
        } else {
            analysisClient!!.waitUntilQuiet()
            analysisClient!!.flush() ?: emptyAnalysis()
        }
    }

    private suspend fun createVideo(): String {
        val gson = Gson()
        val requestJson = gson.toJson(mapOf(
            "domain" to domain,
            "activity" to activity,
            "inference" to "local",
            "resolutionWidth" to INPUT_RESOLUTION_WIDTH,
            "resolutionHeight" to INPUT_RESOLUTION_HEIGHT,
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


    private suspend fun init() {
        val modelStore = ModelStore(apiKey, context)
        poseEstimator = modelStore.getPoseEstimator()
    }

    private fun previousFrameInference(newFrameIndex: Int, analysis: Analysis? = null): FrameInference {
        val analysis = analysis ?: latestAnalysis ?: emptyAnalysis()
        if (previousFrameInference == null) {
            return FrameInference(
                keypoints = HashMap(),
                previousFrame = null,
                frameIndex = newFrameIndex,
                secondsSinceStart = 0.0,
                analysis = analysis,
                smoother = smoother,
            )
        }
        else {
            return FrameInference(
                keypoints = previousFrameInference!!.keypoints,
                previousFrame = previousFrameInference!!.previousFrame,
                frameIndex = newFrameIndex,
                secondsSinceStart = previousFrameInference!!.secondsSinceStart,
                analysis = analysis,
                smoother = smoother,
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
            val bbox: BoundingBox? = if (previousFrameInference?.keypoints == null) {
                null
            } else {
                BoundingBox.fromPreviousFrame(previousFrameInference!!.skeleton(), frame.width, frame.height)
            }

            val keypoints = poseEstimator!!.estimatePose(frame, bbox)
            val newInference = FrameInference(
                keypoints = keypoints.mapIndexed { i, k -> i to k}.toMap(),
                previousFrame = previousFrameInference,
                frameIndex = newFrameIndex,
                secondsSinceStart = if (startedAt == null) -1.0 else (
                        (frameTimestamp.toEpochMilli() - startedAt!!.toEpochMilli()) / 1000.0
                ),
                analysis = latestAnalysis ?: emptyAnalysis(),
                smoother = smoother,
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