package ai.getguru.androidsdk

import ai.getguru.androidsdk.ImageUtils.toBitmap
import android.graphics.Bitmap
import android.media.Image

interface IPoseEstimator {

    fun estimatePose(img: Image, rotation: Int): Keypoints {
        return estimatePose(img, rotation, null)
    }

    /**
     * Note: this default implementation uses toBitmap(), which is slow
     */
    fun estimatePose(img: Image, rotation: Int, bbox: BoundingBox?): Keypoints {
        return estimatePose(img.toBitmap(rotation), bbox)
    }

    fun estimatePose(bitmap: Bitmap): Keypoints
    fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints
}