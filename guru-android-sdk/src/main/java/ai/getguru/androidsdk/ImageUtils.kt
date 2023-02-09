package ai.getguru.androidsdk


object ImageUtils {
    // This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their ranges
    // are normalized to eight bits.
    private const val kMaxChannelValue = 262143

    fun convertYUV420ToARGB8888(
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        out: IntArray
    ) {
        var yp = 0
        for (j in 0 until height) {
            val pY = yRowStride * j
            val pUV = uvRowStride * (j shr 1)
            for (i in 0 until width) {
                val uv_offset = pUV + (i shr 1) * uvPixelStride
                out[yp++] = yuv2rgb(
                    0xff and yData[pY + i].toInt(),
                    0xff and uData[uv_offset].toInt(),
                    0xff and vData[uv_offset].toInt()
                )
            }
        }
    }

    private fun yuv2rgb(y: Int, u: Int, v: Int): Int {
        // Adjust and check YUV values
        var y = y
        var u = u
        var v = v
        y = if (y - 16 < 0) 0 else y - 16
        u -= 128
        v -= 128

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        val y1192 = 1192 * y
        var r = y1192 + 1634 * v
        var g = y1192 - 833 * v - 400 * u
        var b = y1192 + 2066 * u

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r =
            if (r > kMaxChannelValue) kMaxChannelValue else if (r < 0) 0 else r
        g =
            if (g > kMaxChannelValue) kMaxChannelValue else if (g < 0) 0 else g
        b =
            if (b > kMaxChannelValue) kMaxChannelValue else if (b < 0) 0 else b
        return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
    }
}
