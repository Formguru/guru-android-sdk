package ai.getguru.androidsdk

class Rep constructor(
    val startTimestamp: Long,
    val midTimestamp: Long,
    val endTimestamp: Long,
    val analyses: Map<String, Any>,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Rep

        if (startTimestamp != other.startTimestamp) return false
        if (midTimestamp != other.midTimestamp) return false
        if (endTimestamp != other.endTimestamp) return false
        if (analyses != other.analyses) return false

        return true
    }

    override fun hashCode(): Int {
        var result = startTimestamp.hashCode()
        result = 31 * result + midTimestamp.hashCode()
        result = 31 * result + endTimestamp.hashCode()
        result = 31 * result + analyses.hashCode()
        return result
    }
}