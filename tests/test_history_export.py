"""Tests for CSV history export utility."""

from utils.history_export import history_to_csv_text


def test_history_to_csv_aligns_columns() -> None:
  csv_text = history_to_csv_text({
      "time_s": [0.0, 1.0],
      "power_mw": [100.0, 101.0],
  })
  lines = csv_text.strip().splitlines()
  assert lines[0] == "time_s,power_mw"
  assert lines[1] == "0.0,100.0"
  assert lines[2] == "1.0,101.0"


def test_history_to_csv_empty_returns_empty_string() -> None:
  assert history_to_csv_text({}) == ""
