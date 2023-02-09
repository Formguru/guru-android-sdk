package ai.getguru.androidsdk

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.*
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import java.io.File
import java.net.URL

data class OnDeviceModel(
    val modelId: String,
    val file: File
)

data class ModelMetadata(
    val modelId: String,
    val modelType: String,
    val modelUri: URL
)

data class ListModelsResponse(
    val android: List<ModelMetadata>,
)

enum class ModelType {
    POSE,
}

class ModelDownloadFailedException(message: String, cause: java.lang.Exception): Exception(message, cause)

class ModelStore(
    private val apiKey: String,
    private val context: Context
) : CoroutineScope {

    private val LOG_TAG = "ModelStore"
    private var model: Module? = null
    private val useLocalModel = false

    override val coroutineContext
        get() = Dispatchers.IO

    suspend fun getModel(): Module {
        if (model == null) {
            model = loadModel()
        }

        return model!!
    }

    @Throws(Exception::class)
    private fun downloadFile(modelMetadata: ModelMetadata): File {
        Log.i(LOG_TAG, "Downloading model: $modelMetadata")

        val url = modelMetadata.modelUri
        url.openConnection().getInputStream().use { input ->
            val outputFile = File(getModelStoreRoot(), "${modelMetadata.modelId}.ptl")
            if (outputFile.exists()) {
                outputFile.delete()
            }
            else {
                outputFile.mkdirs()
            }
            outputFile.outputStream().use { output ->
                input.copyTo(output)
            }
            return outputFile
        }
    }

    private fun getModelStoreRoot(): File {
        val fm = context.getExternalFilesDir(null)
        val rootUrl = File(fm, "GuruOnDeviceModel")
        if (!rootUrl.exists()) {
            rootUrl.mkdir()
        }
        return rootUrl
    }

    private suspend fun fetchLatestModelMetadata(): ModelMetadata {
        val models = listRemoteModels()
        return models.first { it.modelType.lowercase() == ModelType.POSE.name.lowercase() }
    }

    private suspend fun fetchModel(): File {
        val localModels = listLocalModels()
        val latestModelMetadata = fetchLatestModelMetadata()

        val localModel = localModels.firstOrNull { it.modelId == latestModelMetadata.modelId }
        if (localModel != null) {
            return localModel.file
        }

        try {
            return downloadFile(latestModelMetadata)
        } catch (e: Exception) {
            throw ModelDownloadFailedException("Failed to download ${latestModelMetadata.modelUri}", e)
        }
    }

    private suspend fun listRemoteModels(): List<ModelMetadata> {
        getOnDeviceModelOverride()?.let {
            return listOf(it)
        }

        return withContext(Dispatchers.IO) {
            val listModelsUrl = URL("https://api.getguru.fitness/mlmodels/ondevice")
            val urlConnection = listModelsUrl.openConnection()
            urlConnection.setRequestProperty("x-api-key", apiKey)
            urlConnection.getInputStream().use {
                val jsonResponse = it.bufferedReader().readText()
                val modelList = Gson().fromJson(jsonResponse, ListModelsResponse::class.java)
                modelList.android
            }
        }
    }

    private fun listLocalModels(): List<OnDeviceModel> {
        val root = getModelStoreRoot()
        val results: MutableList<OnDeviceModel> = mutableListOf()
        for (file in root.walkBottomUp()) {
            if (file.name.endsWith(".ptl")) {
                val modelId = file.nameWithoutExtension
                results.add(OnDeviceModel(modelId, root.resolve(file.name)))
            }
        }
        return results
    }

    private suspend fun loadModel(): Module {
        val modelFile = fetchModel()
        return LiteModuleLoader.load(modelFile.absolutePath)
    }

    private fun getOnDeviceModelOverride(): ModelMetadata? {
        return if (useLocalModel) {
            ModelMetadata(
                modelId = "ADDME",
                modelType = ModelType.POSE.name,
                modelUri = URL("ADDME")
            )
        } else {
            null
        }
    }
}
