package ai.getguru.gradle

import okhttp3.OkHttpClient
import okhttp3.Request
import okio.BufferedSink
import okio.buffer
import okio.sink
import org.gradle.api.DefaultTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.TaskAction
import java.io.*
import java.net.URI
import java.nio.file.*
import java.util.zip.ZipFile
import kotlin.io.path.*

/**
 * Download ONNXRuntime and OpenCV native libraries
 */
open class InstallNativeLibsTask : DefaultTask() {

    @Input
    lateinit var destDir: String
    @Input
    lateinit var onnxUrl: String
    @Input
    lateinit var openCvUrl: String

    companion object {
        val ABI_LIST = listOf("x86", "x86_64", "armeabi-v7a", "arm64-v8a")
    }

    @TaskAction
    fun run() {
        installOnnx()
        installOpenCv()
    }

    private fun installOpenCv() {
        val zip = downloadFile(openCvUrl)
        extractZipTargets(
            zip.toPath(),
            mapOf("OpenCV-android-sdk/sdk" to "sdk"),
            Path.of(destDir).resolve("opencv")
        )
    }

    private fun installOnnx() {
        val aar: File = downloadFile(onnxUrl)
        extractZipTargets(
            aar.toPath(),
            ABI_LIST.associateBy { abi -> "jni/${abi}" } +
                    mapOf("headers" to "headers"),
            Path.of(destDir).resolve("onnxruntime"),
        )
    }

    private fun downloadFile(url: String): File {
        val filename = URI(url).rawPath.substringAfterLast("/")
        val tmpdir = System.getProperty("java.io.tmpdir") ?: "/tmp"
        val downloadedFile = File(tmpdir, filename)
        if (!downloadedFile.exists()) {
            val request = Request.Builder().url(url).build()
            val response = OkHttpClient().newCall(request).execute()
            val sink: BufferedSink = downloadedFile.sink().buffer()
            sink.writeAll(response.body!!.source())
            sink.close()
        }
        return downloadedFile

    }

    private fun unzip(zipFilePath: String, destinationDirPath: String) {
        val zipFile = File(zipFilePath)
        val destDir = File(destinationDirPath)

        if (!destDir.exists()) {
            destDir.mkdirs()
        }

        val zip = ZipFile(zipFile)
        val entries = zip.entries()

        while (entries.hasMoreElements()) {
            val entry = entries.nextElement()
            val entryPath = Paths.get(destDir.absolutePath, entry.name)
            if (entry.isDirectory) {
                Files.createDirectories(entryPath)
            } else {
                val parentDirPath = entryPath.parent
                if (!Files.exists(parentDirPath)) {
                    Files.createDirectories(parentDirPath)
                }
                Files.copy(zip.getInputStream(entry), entryPath, StandardCopyOption.REPLACE_EXISTING)
            }
        }

        zip.close()
    }

    private fun remove(path: Path) {
        if (path.isDirectory()) {
            path.forEachDirectoryEntry {
                remove(it)
            }
        }
        path.deleteExisting()
    }

    private fun extractZipTargets(zipFile: Path, entries: Map<String, String>, destination: Path) {
        unzip(zipFile.absolutePathString(), destination.absolutePathString())
        for ((src, dst) in entries) {
            if (destination.resolve(src) == destination.resolve(dst)) {
                continue
            }
            destination.resolve(src).toFile().copyRecursively(
                destination.resolve(dst).toFile(), overwrite = true
            )
        }
        destination.forEachDirectoryEntry {dir ->
            val canDiscard = entries.values.none { target ->
                val rel = dir.relativeTo(destination).toString()
                target.startsWith(rel)
            }
            if (canDiscard) {
                remove(dir)
            }
        }
    }
}