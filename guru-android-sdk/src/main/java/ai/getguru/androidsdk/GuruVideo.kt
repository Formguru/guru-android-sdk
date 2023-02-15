package ai.getguru.androidsdk

import android.media.Image

interface GuruVideo {

    suspend fun newFrame(frame: Image): FrameInference

    suspend fun finish(): Analysis
}