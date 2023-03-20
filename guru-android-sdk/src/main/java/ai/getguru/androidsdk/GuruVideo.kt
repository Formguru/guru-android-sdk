package ai.getguru.androidsdk

import android.graphics.Bitmap
import android.media.Image

interface GuruVideo {

    suspend fun newFrame(frame: Image): FrameInference {
        return newFrame(frame, 0)
    }
    suspend fun newFrame(frame: Image, rotationDegrees: Int): FrameInference

    suspend fun newFrame(frame: Bitmap): FrameInference

    suspend fun finish(): Analysis
}