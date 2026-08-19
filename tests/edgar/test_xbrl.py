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
    assert facts.metrics["revenue"].tag == ("RevenueFromContractWithCustomerExcludingAssessedTax")


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
