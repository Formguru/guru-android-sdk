package ai.getguru.androidsdk

class Analysis constructor(
    val movement: String?,
    val reps: List<Rep>,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Analysis

        if (movement != other.movement) return false
        if (reps != other.reps) return false

        return true
    }

    override fun hashCode(): Int {
        var result = movement?.hashCode() ?: 0
        result = 31 * result + reps.hashCode()
        return result
    }
}