package ai.getguru.androidsdk

class Keypoint constructor(
    val x: Double,
    val y: Double,
    val score: Double,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        val otherKeypoint = other as Keypoint
        return x == otherKeypoint.x &&
                y == otherKeypoint.y &&
                score == otherKeypoint.score
    }

    override fun hashCode(): Int {
        return x.hashCode() xor y.hashCode() xor score.hashCode()
    }

    override fun toString(): String {
        return "Keypoint(x=$x, y=$y, score=$score)"
    }
}