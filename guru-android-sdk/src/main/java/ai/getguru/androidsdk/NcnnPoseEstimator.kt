package ai.getguru.androidsdk

import ai.getguru.androidsdk.ImageUtils.crop
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import java.io.File
import kotlin.math.roundToInt

class NcnnPoseEstimator(paramFile: File, binFile: File) : IPoseEstimator {

    companion object {
        init {
            System.loadLibrary("guru_pose_inference")
        }
        const val INPUT_WIDTH = 192;
        const val INPUT_HEIGHT = 256;

        private fun zeroPadToSize(
            image: Bitmap,
            destWidth: Int,
            destHeight: Int,
            boundingBox: BoundingBox?,
        ): PreprocessedImage {
            /**
             * Scale the image to destWidth x destHeight with zero-padding as-needed.
             *
             * Note: the larger of image.width/destWidth and image.height/destHeight determines the scale.
             * The other dimension is the one that will be zero-padded.
             *
             * Source: https://stackoverflow.com/a/35598907/895769
             */
            if (boundingBox != null) {
                Log.i("NcnnPoseEstimator", "Bounding box: x1=${boundingBox.x1},y1=${boundingBox.y1},x2=${boundingBox.x2}, y2=${boundingBox.y2}")
            }
            val hasGoodBbox = (
                    boundingBox != null
                            && (boundingBox.x2 - boundingBox.x1) >= .2 * image.width
                            && (boundingBox.y2 - boundingBox.y1) >= .2 * image.height
                    )

            val cropped = if (hasGoodBbox) image.crop(boundingBox!!) else image
            val background = Bitmap.createBitmap(destWidth, destHeight, Bitmap.Config.ARGB_8888)
            val originalWidth = cropped.width.toFloat()
            val originalHeight = cropped.height.toFloat()
            val canvas = Canvas(background)

            val scaleX = destWidth.toFloat() / originalWidth
            val scaleY = destHeight.toFloat() / originalHeight

            var xTranslation = 0.0f
            var yTranslation = 0.0f
            val scale: Float

            if (scaleX < scaleY) { // Scale on X, translate on Y
                scale = scaleX
                yTranslation = (destHeight.toFloat() - originalHeight * scale) / 2.0f
            } else { // Scale on Y, translate on X
                scale = scaleY
                xTranslation = (destWidth.toFloat() - originalWidth * scale) / 2.0f
            }

            val transformation = Matrix()
            transformation.postTranslate(xTranslation, yTranslation)
            transformation.preScale(scale, scale)
            val paint = Paint()
            paint.isFilterBitmap = true
            // paint.alpha = 255
            canvas.drawBitmap(cropped, transformation, paint)

            return PreprocessedImage(
                background,
                scale,
                xTranslation,
                yTranslation,
                if (hasGoodBbox) boundingBox!!.x1 else 0,
                if (hasGoodBbox) boundingBox!!.y1 else 0,
                image.width,
                image.height,
            )
        }


    }

    init {
        initModel(paramFile.absolutePath, binFile.absolutePath)
    }

    external fun initModel(paramPath: String, binPath: String)
    external fun detectPose(bitmap: Bitmap): FloatArray

    override fun estimatePose(bitmap: Bitmap): Keypoints {
        return estimatePose(bitmap, null)
    }

    override fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints {
        val input = zeroPadToSize(bitmap, INPUT_WIDTH, INPUT_HEIGHT, boundingBox)
        val keypoints: FloatArray = detectPose(input.bitmap)
        val results: MutableList<Keypoint> = mutableListOf()
        for (i in 0 until keypoints.size / 3) {
            val x = keypoints[3 * i]
            val y = keypoints[3 * i + 1]
            val score = keypoints[3 * i + 2]
            results.add(Keypoint(x.toDouble(), y.toDouble(), score.toDouble()))
        }
        return postProcess(Keypoints.of(results), input)
    }

    private class PreprocessedImage(
        val bitmap: Bitmap,
        val scale: Float,
        val xPad: Float,
        val yPad: Float,
        val xOffset: Int,
        val yOffset: Int,
        val originalWidth: Int,
        val originalHeight: Int,
    )

    private fun postProcess(keypoints: Keypoints, input: PreprocessedImage): Keypoints {
        val shifted: MutableList<Keypoint> = ArrayList()
        for (i in keypoints.indices) {
            val k = keypoints[i]
            var x = k.x * INPUT_WIDTH
            x -= input.xPad
            x /= input.scale
            x += input.xOffset

            var y = k.y * INPUT_HEIGHT
            y -= input.yPad
            y /= input.scale
            y += input.yOffset

            val normalizedX = x / input.originalWidth
            val normalizedY = y / input.originalHeight
            shifted.add(
                Keypoint(
                    normalizedX,
                    normalizedY,
                    k.score
                )
            )
        }
        return Keypoints.of(shifted)
    }
}
