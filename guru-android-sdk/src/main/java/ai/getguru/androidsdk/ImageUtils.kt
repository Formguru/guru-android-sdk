package ai.getguru.androidsdk

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.graphics.Rect
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.Image
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ReadOnlyBufferException
import kotlin.experimental.inv
import kotlin.math.roundToInt


object ImageUtils {

    /**
     * Convert the image to a bitmap.
     *
     * Note that the Image must be YUV_420_8888 - RGBA_8888 is not supported.
     *
     * Adapted from: https://stackoverflow.com/a/56812799/895769
    */
    fun Image.toBitmap(rotationDegrees: Int = 0): Bitmap {
        if (this.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Image must be in ImageFormat.YUV_420_888 format")
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

    /**
     * Return the raw, NV21-encoded bytes.
     *
     * Note that the Image must be YUV_420_8888 - RGBA_8888 is not supported.
     *
     * Based on https://stackoverflow.com/a/52740776/895769
     */
    fun Image.toNv21Buffer(): ByteArray {
        if (this.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Image must be in ImageFormat.YUV_420_888 format")
        }
        val width: Int = this.width
        val height: Int = this.height
        val ySize = width * height
        val uvSize = width * height / 4

        val nv21 = ByteArray(ySize + uvSize * 2)
        val yBuffer: ByteBuffer = this.planes[0].buffer // Y
        val uBuffer: ByteBuffer = this.planes[1].buffer // U
        val vBuffer: ByteBuffer = this.planes[2].buffer // V

        var rowStride: Int = this.planes[0].rowStride
        assert(this.planes[0].pixelStride == 1)

        var pos = 0
        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize)
            pos += ySize
        } else {
            var yBufferPos = -rowStride // not an actual position
            while (pos < ySize) {
                yBufferPos += rowStride
                yBuffer.position(yBufferPos)
                yBuffer.get(nv21, pos, width)
                pos += width
            }
        }

        rowStride = this.planes[2].rowStride
        val pixelStride: Int = this.planes[2].pixelStride

        assert(rowStride == this.planes[1].rowStride)
        assert(pixelStride == this.planes[1].pixelStride)

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            val savePixel: Byte = vBuffer.get(1)
            try {
                vBuffer.put(1, savePixel.inv())
                if (uBuffer.get(0) == savePixel.inv()) {
                    vBuffer.put(1, savePixel)
                    vBuffer.position(0)
                    uBuffer.position(0)
                    vBuffer.get(nv21, ySize, 1)
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining())
                    return nv21 // shortcut
                }
            } catch (ex: ReadOnlyBufferException) {
                // unfortunately, we cannot check if vBuffer and uBuffer overlap
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel)
        }

        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                val vuPos = col * pixelStride + row * rowStride
                nv21[pos++] = vBuffer.get(vuPos)
                nv21[pos++] = uBuffer.get(vuPos)
            }
        }

        return nv21
    }

    fun Bitmap.crop(box: BoundingBox): Bitmap {
        val width = box.x2 - box.x1
        val height = box.y2 - box.y1
        return Bitmap.createBitmap(
            this,
            (box.x1 * this.width).roundToInt(),
            (box.y1 * this.width).roundToInt(),
            (width * this.width).roundToInt(),
            (height * this.height).roundToInt()
        )
    }
}
