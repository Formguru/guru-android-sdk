package ai.getguru.androidsdk

import com.google.gson.Gson
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.assertTrue
import org.junit.Test


class AnalysisClientTest {
    @Test
    fun frameIsSentToServer() {
        val server = MockWebServer()
        val activity = "squat"
        val videoId = "abc-123"
        val analysisResponse = mapOf(
            "liftType" to activity,
            "reps" to listOf(
                mapOf(
                    "startTimestampMs" to 0,
                    "midTimestampMs" to 100,
                    "endTimestampMs" to 200,
                )
            )
        )
        val gson = Gson()
        server.enqueue(MockResponse().setBody(gson.toJson(analysisResponse)))
        server.start()
        val baseUrl = server.url("")

        val analysisClient = AnalysisClient(videoId, "def-456", apiServer = baseUrl.toString())
        runBlocking {
            analysisClient.add(
                FrameInference(
                    emptyMap(),
                    null,
                    0,
                    0.0,
                    Analysis(null, emptyList())
                )
            )

            val request = server.takeRequest()
            assertTrue(request.path!!.endsWith("/videos/$videoId/j2p"))
        }
    }
}