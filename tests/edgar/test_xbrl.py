from era.edgar.xbrl import facts_card, normalize_facts

COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "fy": 2024,
                            "fp": "FY",
                            "form": "10-K",
                            "val": 391035000000,
                            "accn": "0000320193-24-000123",
                        }
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "fy": 2024,
                            "fp": "FY",
                            "form": "10-K",
                            "val": 93736000000,
                            "accn": "0000320193-24-000123",
                        }
                    ]
                }
            },
        }
    }
}


def test_resolves_revenue_from_an_alternate_tag() -> None:
    facts = normalize_facts(COMPANYFACTS)

    assert facts.metrics["revenue"].value == 391_035_000_000
    assert facts.metrics["revenue"].tag == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_reports_metrics_it_could_not_resolve() -> None:
    facts = normalize_facts(COMPANYFACTS)

    assert "total_debt" in facts.missing
    assert "revenue" not in facts.missing


def test_facts_card_is_compact_and_names_its_period() -> None:
    card = facts_card(normalize_facts(COMPANYFACTS))

    assert "revenue" in card
    assert "391,035,000,000" in card
    assert "FY2024" in card
    assert len(card) < 1200


# --- Below: fixtures for the failure modes a single-entry-per-tag payload
# (like COMPANYFACTS above) can never exercise. In real companyfacts, fy/fp/
# form describe the *filing* a fact appeared in, not the period it covers --
# a single 10-K emits its whole multi-year comparative statement under one
# fy/fp/form. These fixtures give each tag several entries, distinguished
# only by start/end/filed, the way SEC actually reports them.


def test_three_year_comparatives_pick_the_current_year_not_the_oldest() -> None:
    # One FY2024 10-K's income statement tags all three comparative years
    # under the same fy=2024/fp=FY/form=10-K -- only start/end tell them
    # apart. The oldest ends up first in SEC's own ordering.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2021-09-26",
                                "end": "2022-09-24",
                                "filed": "2024-11-01",
                                "val": 394328000000,
                                "accn": "0000320193-24-000123",
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2022-09-25",
                                "end": "2023-09-30",
                                "filed": "2024-11-01",
                                "val": 383285000000,
                                "accn": "0000320193-24-000123",
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 391035000000,
                                "accn": "0000320193-24-000123",
                            },
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["revenue"].value == 391_035_000_000
    assert facts.metrics["revenue"].fiscal_period == "FY2024"


def test_a_same_period_restatement_prefers_the_later_filed_value() -> None:
    # Same fy/fp/form/start/end/accn -- SEC reprocessed the accession's
    # XBRL and corrected the tagged value. Only `filed` distinguishes the
    # original submission from the correction, so the tie-break must look
    # at it: picking "whichever entry appears first" (as a naive `max` on
    # a shared key does) would silently keep the uncorrected figure.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 93736000000,
                                "accn": "0000320193-24-000123",
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2025-02-10",
                                "val": 93800000000,
                                "accn": "0000320193-24-000123",
                            },
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["net_income"].value == 93_800_000_000


def test_instant_metric_picks_the_current_balance_not_the_comparative() -> None:
    # A comparative balance sheet: both the current and prior year-end are
    # tagged Assets under the same fy/fp/form, distinguished only by end
    # (instant facts carry no start at all).
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "end": "2023-09-30",
                                "filed": "2024-11-01",
                                "val": 352755000000,
                                "accn": "0000320193-24-000123",
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 364980000000,
                                "accn": "0000320193-24-000123",
                            },
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["total_assets"].value == 364_980_000_000
    assert facts.metrics["total_assets"].fiscal_period == "FY2024"


def test_a_q4_stub_does_not_override_the_full_year_duration() -> None:
    # A quarterly duration fact tagged within the annual filing, ending on
    # the same date as the full year, would tie on `end` alone. The
    # ~365-day span check is what tells them apart.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2024-07-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 94930000000,
                                "accn": "0000320193-24-000123",
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 391035000000,
                                "accn": "0000320193-24-000123",
                            },
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["revenue"].value == 391_035_000_000


def test_stale_priority_tag_does_not_shadow_a_live_lower_priority_tag() -> None:
    # The higher-priority tag exists but only through 2019; the
    # lower-priority tag is the one this filer currently reports under.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2019,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2018-09-30",
                                "end": "2019-09-28",
                                "filed": "2019-11-01",
                                "val": 100.0,
                                "accn": "0000320193-19-000001",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 391035000000,
                                "accn": "0000320193-24-000123",
                            }
                        ]
                    }
                },
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["revenue"].value == 391_035_000_000
    assert facts.metrics["revenue"].tag == "Revenues"


def test_a_malformed_entry_is_skipped_not_fatal() -> None:
    # Row 1 is missing "accn" and "fy" entirely -- both occur in real SEC
    # data. It must be skipped, not raise, and must not stop the good row
    # from resolving.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "val": 93736000000,
                            },
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 93736000000,
                                "accn": "0000320193-24-000123",
                            },
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert facts.metrics["net_income"].value == 93_736_000_000
    assert facts.metrics["net_income"].accession == "0000320193-24-000123"


def test_a_metric_with_only_malformed_entries_falls_through_to_missing() -> None:
    # `val: null` and `fy: null` both occur in real payloads. Neither may
    # raise -- the metric should simply end up unresolved.
    companyfacts = {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "fy": None,
                                "fp": "FY",
                                "form": "10-K",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "val": None,
                                "accn": "0000320193-24-000123",
                            }
                        ]
                    }
                }
            }
        }
    }

    facts = normalize_facts(companyfacts)

    assert "operating_income" in facts.missing


def test_facts_card_header_names_it_as_financial_facts() -> None:
    card = facts_card(normalize_facts(COMPANYFACTS))

    assert "financial facts" in card.lower()
