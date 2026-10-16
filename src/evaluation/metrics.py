import re
import jiwer


_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

_CHAR_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])

WER_CLEAN_THRESHOLD = 0.20
WER_BORDERLINE_HIGH = 0.50


def calculate_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0
    try:
        return jiwer.wer(
            reference, hypothesis,
            reference_transform=_TRANSFORM,
            hypothesis_transform=_TRANSFORM,
        )
    except Exception:
        return 1.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0
    try:
        return jiwer.cer(
            reference, hypothesis,
            reference_transform=_CHAR_TRANSFORM,
            hypothesis_transform=_CHAR_TRANSFORM,
        )
    except Exception:
        return 1.0


def exact_match(reference: str, hypothesis: str) -> bool:
    def normalise(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    return normalise(reference) == normalise(hypothesis)


def classify_result(wer: float) -> str:
    if wer <= WER_CLEAN_THRESHOLD:
        return "clean"
    if wer <= WER_BORDERLINE_HIGH:
        return "borderline"
    return "failed"
