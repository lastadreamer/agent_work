from xiangqi_engine.progress import eta_seconds, format_hms, progress_line


def test_format_hms():
    assert format_hms(None) == "?"
    assert format_hms(9) == "9s"
    assert format_hms(75) == "1m15s"
    assert format_hms(3661) == "1h01m01s"


def test_eta_seconds():
    assert eta_seconds(0, 10, 5) is None
    assert eta_seconds(5, 10, 100) == 100
    assert eta_seconds(10, 10, 50) == 0.0


def test_progress_line_includes_counts():
    line = progress_line("self-play", 8, 32, started_at=0.0, extra="avg 40 plies/game")
    assert "8/32" in line
    assert "self-play" in line
    assert "avg 40 plies/game" in line
