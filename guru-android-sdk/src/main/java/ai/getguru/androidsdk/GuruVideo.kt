package ai.getguru.androidsdk

import android.media.Image

interface GuruVideo {

    fun newFrame(frame: Image): FrameInference
}