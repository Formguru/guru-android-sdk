package ai.getguru.androidsdk

import android.graphics.Bitmap
import android.media.Image

interface GuruVideo {

    suspend fun newFrame(frame: Image): FrameInference

    suspend fun newFrame(frame: Bitmap): FrameInference

    suspend fun finish(): Analysis
}