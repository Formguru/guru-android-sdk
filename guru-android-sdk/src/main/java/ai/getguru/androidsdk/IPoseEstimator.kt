package ai.getguru.androidsdk

import android.graphics.Bitmap

interface IPoseEstimator {
    fun estimatePose(bitmap: Bitmap): Keypoints
    fun estimatePose(bitmap: Bitmap, boundingBox: BoundingBox?): Keypoints
}