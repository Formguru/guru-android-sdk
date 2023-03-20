package ai.getguru.androidsdk

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.graphics.Rect
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.Image
import java.io.ByteArrayOutputStream
import kotlin.math.roundToInt


object ImageUtils {

    // Adapted from: https://stackoverflow.com/a/56812799/895769
    fun Image.toBitmap(rotationDegrees: Int = 0): Bitmap {
        if (this.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Image must be in YUV format (i..e., ImageFormat.YUV_420_888 format")
        }
        val yBuffer = planes[0].buffer // Y
        val vuBuffer = planes[2].buffer // VU

        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()

        val nv21 = ByteArray(ySize + vuSize)

        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        return if (rotationDegrees == 0) {
            bitmap
        } else {
            val rotate = Matrix()
            rotate.postRotate(rotationDegrees.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, rotate, true)
        }
    }

    fun Bitmap.crop(box: BoundingBox): Bitmap {
        val width = box.x2 - box.x1
        val height = box.y2 - box.y1
        return Bitmap.createBitmap(this, box.x1, box.y1, width, height)
    }
}
