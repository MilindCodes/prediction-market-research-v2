"""Invariants on the research configuration.

These constants drive every downstream estimate, so a typo here silently
changes results rather than raising. Each assertion below encodes an
assumption the pipeline already relies on.
"""

from __future__ import annotations

import datetime as dt

import pytest

import config


class TestCredentials:
    """Credentials must come from the environment, never from source."""

    @pytest.mark.parametrize(
        "name",
        ["POLYMARKET_API_KEY", "KALSHI_API_KEY", "KALSHI_KEY_ID"],
    )
    def test_credential_defaults_to_empty(self, name, monkeypatch):
        monkeypatch.delenv(name, raising=False)
        import importlib

        reloaded = importlib.reload(config)
        assert getattr(reloaded, name) == "", (
            f"{name} must default to an empty string. A non-empty default "
            "means a credential is hardcoded in config.py."
        )

    def test_key_file_is_a_path_not_key_material(self):
        assert "PRIVATE KEY" not in str(config.KALSHI_KEY_FILE), (
            "KALSHI_KEY_FILE must hold a filesystem path, never key material."
        )


class TestLogOddsClipping:
    def test_bounds_are_ordered_and_inside_the_unit_interval(self):
        assert 0.0 < config.LOG_ODDS_CLIP_LO < config.LOG_ODDS_CLIP_HI < 1.0

    def test_bounds_are_symmetric(self):
        # An asymmetric clip would bias the log-odds transform.
        assert config.LOG_ODDS_CLIP_LO == pytest.approx(1.0 - config.LOG_ODDS_CLIP_HI)


class TestTimeSteps:
    def test_time_steps_are_positive(self):
        assert config.HOURLY_DT > 0
        assert config.DAILY_DT > 0

    def test_daily_is_exactly_twenty_four_hourly_steps(self):
        assert config.DAILY_DT == pytest.approx(24.0 * config.HOURLY_DT)

    def test_hourly_step_matches_a_365_25_day_year(self):
        assert config.HOURLY_DT == pytest.approx(1.0 / (365.25 * 24))


class TestDateRanges:
    @pytest.mark.parametrize(
        "start_attr,end_attr",
        [
            ("POLYMARKET_DATE_START", "POLYMARKET_DATE_END"),
            ("KALSHI_DATE_START", "KALSHI_DATE_END"),
            ("DATE_RANGE_START", "DATE_RANGE_END"),
        ],
    )
    def test_start_precedes_end(self, start_attr, end_attr):
        start = dt.date.fromisoformat(getattr(config, start_attr))
        end = dt.date.fromisoformat(getattr(config, end_attr))
        assert start < end, f"{start_attr} must precede {end_attr}"

    def test_legacy_aliases_track_the_polymarket_window(self):
        # config.py redefines these aliases after the initial assignment;
        # this pins the intended final value.
        assert config.DATE_RANGE_START == config.POLYMARKET_DATE_START
        assert config.DATE_RANGE_END == config.POLYMARKET_DATE_END


class TestFilters:
    def test_thresholds_are_positive(self):
        assert config.MIN_DURATION_DAYS > 0
        assert config.MIN_TRADE_COUNT > 0


class TestFomcDates:
    def test_every_entry_parses_as_utc(self):
        for raw in config.FOMC_DATES:
            assert raw.endswith("Z"), f"{raw} is not marked UTC"
            dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))

    def test_dates_are_strictly_increasing(self):
        parsed = [dt.datetime.fromisoformat(d.replace("Z", "+00:00")) for d in config.FOMC_DATES]
        assert parsed == sorted(parsed), "FOMC_DATES must be chronological"
        assert len(set(parsed)) == len(parsed), "FOMC_DATES contains duplicates"

    def test_eight_scheduled_meetings_per_year(self):
        years = {d[:4] for d in config.FOMC_DATES}
        for year in years:
            count = sum(1 for d in config.FOMC_DATES if d.startswith(year))
            assert count == 8, f"{year} has {count} FOMC dates; 8 are scheduled each year"

    def test_release_times_are_2pm_eastern(self):
        # Statements drop at 14:00 ET, i.e. 18:00 UTC in EDT and 19:00 UTC in EST.
        for raw in config.FOMC_DATES:
            hour = dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).hour
            assert hour in (18, 19), f"{raw} is not a 14:00 ET release time"

    def test_covers_the_polymarket_sample_window(self):
        start = dt.date.fromisoformat(config.POLYMARKET_DATE_START)
        end = dt.date.fromisoformat(config.POLYMARKET_DATE_END)
        for raw in config.FOMC_DATES:
            day = dt.date.fromisoformat(raw[:10])
            assert start <= day <= end, f"{raw} falls outside the sample window"


class TestKeywordLists:
    @pytest.mark.parametrize(
        "name", ["FED_KEYWORDS", "CPI_KEYWORDS", "ECONOMIC_KEYWORDS", "POLITICAL_KEYWORDS"]
    )
    def test_lists_are_populated_lowercase_and_unique(self, name):
        keywords = getattr(config, name)
        assert keywords, f"{name} is empty"
        assert all(k == k.lower() for k in keywords), f"{name} contains uppercase entries"
        assert all(k.strip() == k for k in keywords), f"{name} has untrimmed whitespace"
        assert len(set(keywords)) == len(keywords), f"{name} contains duplicates"

    def test_economic_keywords_is_fed_plus_cpi(self):
        assert config.ECONOMIC_KEYWORDS == config.FED_KEYWORDS + config.CPI_KEYWORDS

    def test_economic_and_political_keywords_do_not_overlap(self):
        # The catalog filters use these as mutually exclusive buckets.
        overlap = set(config.ECONOMIC_KEYWORDS) & set(config.POLITICAL_KEYWORDS)
        assert not overlap, f"keyword buckets overlap: {sorted(overlap)}"
