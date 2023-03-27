package ai.getguru.androidsdk

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import ai.getguru.androidsdk.Keypoints.Companion.of
import ai.getguru.androidsdk.Keypoints.Companion.parse
import org.pytorch.Module
import org.pytorch.IValue
import org.pytorch.Tensor
import org.pytorch.MemoryFormat
import org.pytorch.torchvision.TensorImageUtils
import java.io.IOException

class TorchLitePoseEstimator private constructor(private val module: Module) : IPoseEstimator {

    override fun estimatePose(bitmap: Bitmap): Keypoints {
        val defaultBbox = BoundingBox(0f, 0f, 1f, 1f)
        return estimatePose(bitmap, defaultBbox)
    }

    override fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints {
        // TODO: pre-crop using the bounding-box and scale the person height to 200px w/ 25% padding
        // (for now the bounding-box is completely ignored)
        val preprocessed = zeroPadToSize(
            bitmap,
            INPUT_WIDTH,
            INPUT_HEIGHT
        )
        val rawOutput = module.forward(IValue.from(preprocessed.feats))
        val keypoints = parse(rawOutput)
        return postProcess(keypoints, preprocessed)
    }

    private class ImageFeatures {
        var feats // chw, rgb-normalized
                : Tensor? = null
        var scale = 0f
        var xPad = 0f
        var yPad = 0f
        var originalHeight = 0f
        var originalWidth = 0f
    }

    companion object {
        private const val INPUT_WIDTH = 192
        private const val INPUT_HEIGHT = 256
        @Throws(IOException::class)
        fun withTorchModel(model: Module): TorchLitePoseEstimator {
            return TorchLitePoseEstimator(model)
        }

        private fun zeroPadToSize(image: Bitmap, destWidth: Int, destHeight: Int): ImageFeatures {
            /**
             * Scale the image to destWidth x destHeight with zero-padding as-needed.
             *
             * Note: the larger of image.width/destWidth and image.height/destHeight determines the scale.
             * The other dimension is the one that will be zero-padded.
             *
             * Source: https://stackoverflow.com/a/35598907/895769
             */
            val background = Bitmap.createBitmap(destWidth, destHeight, Bitmap.Config.ARGB_8888)
            val originalWidth = image.width.toFloat()
            val originalHeight = image.height.toFloat()
            val canvas = Canvas(background)

            val scaleX = destWidth.toFloat() / originalWidth
            val scaleY = destHeight.toFloat() / originalHeight

            var xTranslation = 0.0f
            var yTranslation = 0.0f
            var scale = 1f

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
            canvas.drawBitmap(image, transformation, paint)

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                background,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                MemoryFormat.CONTIGUOUS // NCHW, i.e., (batch, channels, height, width)
            )

            val result = ImageFeatures()
            result.feats = inputTensor
            result.scale = scale
            result.xPad = xTranslation
            result.yPad = yTranslation
            result.originalHeight = originalHeight
            result.originalWidth = originalWidth
            return result
        }

        private fun postProcess(keypoints: Keypoints, feats: ImageFeatures): Keypoints {
            /**
             * Note: this translation logic will need to be updated if and when we start center-cropping
             * based on the bounding-box.
             */
            val shifted: MutableList<Keypoint> = ArrayList()
            val heatmapHeight = 64.0f
            val heatmapWidth = 48.0f
            for (i in keypoints.indices) {
                val k = keypoints[i]
                var x = (k!!.x * INPUT_WIDTH / heatmapWidth).toFloat()
                x -= feats.xPad
                x /= feats.scale

                var y = (k.y * INPUT_HEIGHT / heatmapHeight).toFloat()
                y -= feats.yPad
                y /= feats.scale

                val normalizedX = x / feats.originalWidth
                val normalizedY = y / feats.originalHeight
                shifted.add(
                    Keypoint(
                        normalizedX.toDouble(),
                        normalizedY.toDouble(),
                        k.score
                    )
                )
            }
            return of(shifted)
        }
    }
}