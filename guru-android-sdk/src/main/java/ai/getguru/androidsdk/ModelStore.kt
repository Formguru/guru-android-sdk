package ai.getguru.androidsdk

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.*
import org.pytorch.LiteModuleLoader
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipFile

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
    private val useLocalModel = false

    override val coroutineContext
        get() = Dispatchers.IO

    suspend fun getPoseEstimator(): IPoseEstimator {
        val modelFiles = getModelFiles()
        val isNcnn = modelFiles.size == 2 && listOf(".param", ".bin").all { suffix -> modelFiles.any { it.name.endsWith(suffix) } }
        val isTorchLite = modelFiles.size == 1 && modelFiles[0].name.endsWith(".ptl")
        if (isNcnn) {
            val paramFile = modelFiles.first() { it.name.endsWith(".param") }
            val binFile = modelFiles.first() { it.name.endsWith(".bin") }
            return NcnnPoseEstimator(paramFile, binFile)
        } else if (isTorchLite) {
            val ptlFile = modelFiles.first()
            val model = LiteModuleLoader.load(ptlFile.absolutePath)
            return TorchLitePoseEstimator.withTorchModel(model)
        } else {
            throw Exception("Unexpected model type: ${modelFiles[0].name.substringAfterLast(".")}")
        }
    }

    private suspend fun getModelFiles(): List<File> {
        val modelFile = fetchModel()
        if (modelFile.name.endsWith(".zip")) {
            val modelDir = File(modelFile.parentFile, modelFile.nameWithoutExtension)
            if (!modelDir.exists()) {
                modelDir.mkdir()
            }
            return unzip(modelFile, modelDir)
        } else {
            return listOf(modelFile)
        }
    }

    private fun unzip(zipFile: File, outputDir: File): List<File> {
        val files = mutableListOf<File>()
        val zip = ZipFile(zipFile)
        val entries = zip.entries()
        while (entries.hasMoreElements()) {
            val entry = entries.nextElement()
            val entryFile = File(outputDir, entry.name)
            if (entry.isDirectory) {
                entryFile.mkdirs()
            } else {
                entryFile.parentFile.mkdirs()
                if (!entryFile.exists()) {
                    zip.getInputStream(entry).use { input ->
                        entryFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                files.add(entryFile)
            }
        }
        return files
    }

    @Throws(Exception::class)
    private fun downloadFile(modelMetadata: ModelMetadata): File {
        Log.i(LOG_TAG, "Downloading model: $modelMetadata")

        val fileRoot = getModelStoreRoot()
        fileRoot.mkdirs()

        val objectName = modelMetadata.modelUri.toString().split("/").last()
        val extension = objectName.substringAfter(".", missingDelimiterValue = "")
        if (extension.isEmpty()) {
            throw IOException("Invalid model file")
        }
        val fileName = "${modelMetadata.modelId}.${extension}"
        val outputFile = File(fileRoot, fileName)
        if (outputFile.exists()) {
            return outputFile
        }

        val url = modelMetadata.modelUri
        val connection = url.openConnection() as HttpURLConnection
        connection.requestMethod = "GET"
        outputFile.outputStream().use { output ->
            connection.inputStream.use { input ->
                input.copyTo(output)
            }
        }

        if (!outputFile.exists()) {
            throw IOException("Failed to download file")
        }
        return outputFile
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
    private fun getListModelsUrl(): URL {
        val params = mapOf(
            "platform" to "android",
            "sdkVersion" to Version.ANDROID_SDK_VERSION,
        )
        val queryString = params.map { (k, v) -> "${k}=${v}" }.joinToString("&")
        return URL("https://api.getguru.fitness/mlmodels/ondevice?${queryString}")
    }

    private suspend fun listRemoteModels(): List<ModelMetadata> {
        getOnDeviceModelOverride()?.let {
            return listOf(it)
        }

        return withContext(Dispatchers.IO) {
            val url = getListModelsUrl()
            val urlConnection = url.openConnection()
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
            } else if (file.name.endsWith(".ncnn.zip")) {
                val modelId = file.name.substringBeforeLast(".ncnn.zip")
                results.add(OnDeviceModel(modelId, root.resolve(file.name)))
            }
        }
        return results
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
