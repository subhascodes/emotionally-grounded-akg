from scripts.emotion_detector import detect_batch
from akg.emotion_schema import EMOTION_SET


def test_emotion_outputs():
    samples = [
        "I feel amazing today!",
        "I am scared of what might happen",
        "I can't believe I messed this up",
        "They betrayed me and I am furious",
        "Maybe things will improve",
        "I feel proud of my work",
        "Thank you for everything you've done",
        "Everything is falling apart",
    ]

    results = detect_batch(samples)

    for r in results:
        emotion = r["emotion"]
        assert emotion in EMOTION_SET, f"Invalid emotion detected: {emotion}"

    print("✅ All outputs valid OCC emotions")

if __name__ == "__main__":
    test_emotion_outputs()