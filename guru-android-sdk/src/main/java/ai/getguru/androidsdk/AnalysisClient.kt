package ai.getguru.androidsdk

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.time.Instant
import java.util.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock


class AnalysisClient(
    private val videoId: String,
    private val apiKey: String,
    private val maxPerSecond: Int = 8,
    private val apiServer: String = "https://api.getguru.fitness"
) {

    private val JSON_MEDIA_TYPE: MediaType = "application/json; charset=utf-8".toMediaType()

    private var buffer = Collections.synchronizedList(mutableListOf<FrameInference>())
    private val bufferLock = ReentrantLock()
    private val buildLock = ReentrantLock()
    private val maxBufferSize = 1000
    private var numTokens: Double = 0.0
    private var tokensLastReplenished = Instant.now()
    private val httpClient = OkHttpClient()

    init {
        this.numTokens = this.maxPerSecond.toDouble()
    }

    suspend fun add(inference: FrameInference): Analysis? {
        if (bufferLock.tryLock(10, TimeUnit.SECONDS)) {
            if (readyToBuffer()) {
                buffer.add(inference)

                while (buffer.size > maxBufferSize) {
                    buffer.removeFirst()
                }
            }

            bufferLock.unlock()
        }

        return if (buffer.isNotEmpty()) {
            flush()
        } else {
            null
        }
    }

    suspend fun flush(): Analysis? {
        if (buildLock.tryLock()) {
            try {
                val bufferCopy = buffer.toList()
                val analysis = patchAnalysis(bufferCopy)

                if (bufferLock.tryLock(10, TimeUnit.SECONDS)) {
                    buffer.subList(0, bufferCopy.size).clear()
                    bufferLock.unlock()
                }

                return analysis
            }
            finally {
                buildLock.unlock()
            }
        } else {
            return null
        }
    }

    fun waitUntilQuiet() {
        if (bufferLock.tryLock(30, TimeUnit.SECONDS)) {
            bufferLock.unlock()
        }

        if (buildLock.tryLock(30, TimeUnit.SECONDS)) {
            buildLock.unlock()
        }
    }

    private suspend fun patchAnalysis(frames: List<FrameInference>): Analysis {
        val gson = Gson()
        val requestJson = gson.toJson(frames.map { frameInferenceToJson(it) })

        val request: Request = Request.Builder()
            .url("$apiServer/videos/$videoId/j2p")
            .header("Content-Type", "application/json")
            .header("x-api-key", apiKey)
            .patch(requestJson.toRequestBody(JSON_MEDIA_TYPE))
            .build()

        return withContext(Dispatchers.IO) {
            val analysisResponse = httpClient.newCall(request).execute().use { response ->
                response.body!!.string()
            }

            val mapType = object : TypeToken<Map<String, Any>>() {}.type
            jsonToAnalysis(gson.fromJson(analysisResponse, mapType))
        }
    }

    private fun frameInferenceToJson(inference: FrameInference): Map<String, Any> {
        val json = mutableMapOf<String, Any>()
        json["frameIndex"] = inference.frameIndex
        json["timestamp"] = inference.secondsSinceStart

        for (nextLandmark in InferenceLandmark.values()) {
            inference.keypointForLandmark(nextLandmark)?.let { keypoint ->
                json[nextLandmark.value] = mapOf(
                    "x" to keypoint.x,
                    "y" to keypoint.y,
                    "score" to keypoint.score
                )
            }
        }
        return json
    }

    private fun jsonToAnalysis(json: Map<String, Any>): Analysis {
        val reps = mutableListOf<Rep>()
        (json["reps"] as? List<*>)?.forEach { rep ->
            (rep as? Map<*, *>)?.let { repMap ->
                reps.add(
                    Rep(
                        startTimestamp = repMap["startTimestampMs"] as? Long ?: 0,
                        midTimestamp = repMap["midTimestampMs"] as? Long ?: 0,
                        endTimestamp = repMap["endTimestampMs"] as? Long ?: 0,
                        analyses = (repMap["analyses"] as? List<*>)?.associate { analysis ->
                            (analysis as Map<*, *>).let { analysisMap ->
                                analysisMap["analysisType"] as String to analysisMap["analysisScalar"]!!
                            }
                        } ?: emptyMap()
                    )
                )
            }
        }
        return Analysis(
            movement = json["liftType"] as? String,
            reps = reps
        )
    }

    private fun readyToBuffer(): Boolean {
        replenishTokens()

        return if (numTokens >= 1.0) {
            numTokens -= 1.0
            true
        } else {
            false
        }
    }

    private fun replenishTokens() {
        val now = Instant.now()
        if (numTokens < maxPerSecond) {
            numTokens += (
                now.toEpochMilli() - tokensLastReplenished.toEpochMilli()
            ) / 1000.0 * maxPerSecond

            if (numTokens > maxPerSecond) {
                numTokens = maxPerSecond.toDouble()
            }
        }
        tokensLastReplenished = now
    }
}