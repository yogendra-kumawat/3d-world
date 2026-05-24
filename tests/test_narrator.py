from tactile_vision.audio.narrator import Narrator


class FakeEngine:
    def __init__(self):
        self.spoken = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def test_first_utterance_speaks_immediately():
    eng = FakeEngine()
    n = Narrator(engine=eng, min_interval_s=3.0, now_fn=lambda: 0.0)
    n.maybe_speak("obstacle 2 meters ahead")
    assert eng.spoken == ["obstacle 2 meters ahead"]


def test_dedup_within_interval():
    eng = FakeEngine()
    t = [0.0]
    n = Narrator(engine=eng, min_interval_s=3.0, now_fn=lambda: t[0])
    n.maybe_speak("obstacle 2 meters ahead")
    t[0] = 1.0
    n.maybe_speak("obstacle 2 meters ahead")
    assert eng.spoken == ["obstacle 2 meters ahead"]


def test_speaks_again_after_interval():
    eng = FakeEngine()
    t = [0.0]
    n = Narrator(engine=eng, min_interval_s=3.0, now_fn=lambda: t[0])
    n.maybe_speak("obstacle 2 meters ahead")
    t[0] = 4.0
    n.maybe_speak("obstacle 2 meters ahead")
    assert eng.spoken == ["obstacle 2 meters ahead", "obstacle 2 meters ahead"]


def test_different_text_speaks_immediately_even_within_interval():
    eng = FakeEngine()
    t = [0.0]
    n = Narrator(engine=eng, min_interval_s=3.0, now_fn=lambda: t[0])
    n.maybe_speak("obstacle 2 meters ahead")
    t[0] = 1.0
    n.maybe_speak("clear ahead")
    assert eng.spoken == ["obstacle 2 meters ahead", "clear ahead"]


def test_disabled_narrator_speaks_nothing():
    eng = FakeEngine()
    n = Narrator(engine=eng, min_interval_s=3.0, now_fn=lambda: 0.0, enabled=False)
    n.maybe_speak("anything")
    assert eng.spoken == []
