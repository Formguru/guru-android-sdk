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
    private val beginRecordingImmediately: Boolean = true,
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
    private var isRecording = beginRecordingImmediately

    companion object {
        suspend fun create(domain: String, activity: String, apiKey: String, context: Context, beginRecordingImmediately: Boolean = true): GuruVideoImpl {
            return GuruVideoImpl(apiKey, context, domain, activity, beginRecordingImmediately = beginRecordingImmediately).also {
                it.init()
            }
        }
    }

    override fun beginRecording() {
        if (isRecording) {
            throw IllegalStateException("Received a call to beginRecording(), but recording is already in progress!")
        }
        isRecording = true
    }

    override suspend fun newFrame(frame: Image, rotationDegrees: Int): FrameInference {
        return newFrame(frame.toBitmap(rotationDegrees))
    }

    override suspend fun newFrame(frame: Bitmap): FrameInference {
        val newFrameIndex = frameIndex.incrementAndGet()
        if (!inferenceLock.tryLock()) {
            return previousFrameInference(newFrameIndex)
        }

        return runInference(frame, newFrameIndex)
    }

    override suspend fun finish(): Analysis {
        isRecording = false
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
        if (isRecording && startedAt == null) {
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
                secondsSinceStart = if (startedAt == null) -1.0 else (
                        (frameTimestamp.toEpochMilli() - startedAt!!.toEpochMilli()) / 1000.0
                ),
                analysis = latestAnalysis ?: emptyAnalysis()
            )

            if (isRecording && analysisClient != null) {
                analysisScope.launch {
                    val newAnalysis = analysisClient!!.add(newInference)
                    if (newAnalysis != null) {
                        latestAnalysis = newAnalysis
                    }
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