"""Company travel policy tools for the LangGraph travel agent.

Design:
- Rules live in policy.json (data), not in the prompt, so they are precise,
  easy to change, and the exact rule applied is visible in each trace.
- The checks are deterministic Python, not LLM judgment: the LLM decides WHEN
  to check, the code decides WHETHER something complies.
- Every violation names the rule that was broken, so the agent can explain
  "why" to the user and evaluators can verify it.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

POLICY_PATH = Path(__file__).parent / "policy.json"


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


# ---------- Schemas ----------


class FlightToCheck(BaseModel):
    flight_number: str
    duration_minutes: int = Field(description="Total itinerary duration including stops")
    stops: int
    price_usd: float = Field(description="Total price for all passengers, as returned by search_flights")


class FlightPolicyInput(BaseModel):
    departure_date: str = Field(description="Departure date YYYY-MM-DD")
    international: bool = Field(description="Use the 'international' value returned by search_flights")
    cabin: Literal["economy", "premium_economy", "business", "first"] = "economy"
    adults: int = Field(1, ge=1, le=9)
    flights: list[FlightToCheck] = Field(description="Flight options returned by search_flights")


class Violation(BaseModel):
    rule: str
    detail: str


class FlightVerdict(BaseModel):
    flight_number: str
    compliant: bool
    violations: list[Violation] = []


class FlightPolicyResult(BaseModel):
    company: str
    verdicts: list[FlightVerdict]
    recommended_flight: Optional[str] = None
    recommendation_reason: Optional[str] = None


class HotelPolicyInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Lisbon'")
    nightly_rate_usd: float = Field(description="Nightly rate in USD")


class HotelPolicyResult(BaseModel):
    company: str
    city: str
    nightly_cap_usd: float
    compliant: bool
    violations: list[Violation] = []


# ---------- Flight check ----------


def _check_flight(f: FlightToCheck, q: FlightPolicyInput, rules: dict) -> FlightVerdict:
    violations = []

    # Cabin class by itinerary length
    hours = f.duration_minutes / 60
    limit = rules["economy_only_under_hours"]
    if hours < limit and q.cabin != "economy":
        violations.append(Violation(
            rule="cabin_class",
            detail=f"{q.cabin} not allowed: trip is {hours:.1f}h, economy only under {limit}h.",
        ))
    elif hours >= limit and q.cabin not in rules["allowed_cabins_long_haul"]:
        violations.append(Violation(
            rule="cabin_class",
            detail=f"{q.cabin} not allowed even on long-haul; allowed: {', '.join(rules['allowed_cabins_long_haul'])}.",
        ))

    # Price cap (per traveler)
    cap = rules["price_cap_usd"]["international" if q.international else "domestic"]
    per_person = f.price_usd / q.adults
    if per_person > cap:
        kind = "international" if q.international else "domestic"
        violations.append(Violation(
            rule="price_cap",
            detail=f"${per_person:,.0f} per traveler exceeds the {kind} cap of ${cap:,}.",
        ))

    # Advance booking
    days_ahead = (datetime.strptime(q.departure_date, "%Y-%m-%d").date() - date.today()).days
    if days_ahead < rules["min_advance_booking_days"]:
        violations.append(Violation(
            rule="advance_booking",
            detail=f"Departure is only {days_ahead} days away; policy requires booking at least {rules['min_advance_booking_days']} days ahead.",
        ))

    return FlightVerdict(flight_number=f.flight_number, compliant=not violations, violations=violations)


def _recommend(verdicts, flights, rules):
    by_num = {f.flight_number: f for f in flights}
    ok = [by_num[v.flight_number] for v in verdicts if v.compliant]
    if not ok:
        return None, "No option complies with policy."
    cheapest = min(ok, key=lambda f: f.price_usd)
    nonstops = [f for f in ok if f.stops == 0]
    if cheapest.stops and nonstops:
        best_nonstop = min(nonstops, key=lambda f: f.price_usd)
        diff = best_nonstop.price_usd - cheapest.price_usd
        if diff < rules["prefer_nonstop_if_price_diff_under_usd"]:
            return best_nonstop.flight_number, (
                f"Nonstop preferred: only ${diff:,.0f} more than the cheapest connecting option "
                f"(policy threshold ${rules['prefer_nonstop_if_price_diff_under_usd']})."
            )
    return cheapest.flight_number, "Cheapest compliant option."


@tool("check_flight_policy", args_schema=FlightPolicyInput)
def check_flight_policy(**kwargs) -> dict:
    """Check flight options against company travel policy.

    ALWAYS call this after search_flights and before recommending any flight.
    Pass the flights exactly as returned by search_flights, plus the trip's
    departure date, cabin, number of adults and the 'international' flag.
    Returns which options comply, the rule behind each violation, and the
    recommended compliant flight.
    """
    q = FlightPolicyInput(**kwargs)
    policy = load_policy()
    rules = policy["flights"]
    verdicts = [_check_flight(f, q, rules) for f in q.flights]
    rec, reason = _recommend(verdicts, q.flights, rules)
    return FlightPolicyResult(
        company=policy["company"], verdicts=verdicts,
        recommended_flight=rec, recommendation_reason=reason,
    ).model_dump()


# ---------- Hotel check ----------


@tool("check_hotel_policy", args_schema=HotelPolicyInput)
def check_hotel_policy(**kwargs) -> dict:
    """Check a hotel's nightly rate against the company's per-city cap.

    Call this before recommending a hotel or hotel area, and use the returned
    cap to tell the user their nightly budget.
    """
    q = HotelPolicyInput(**kwargs)
    policy = load_policy()
    caps = policy["hotels"]["nightly_cap_usd"]
    match = next((c for c in caps if c.lower() == q.city.strip().lower()), None)
    cap = caps[match] if match else caps["default"]
    violations = []
    if q.nightly_rate_usd > cap:
        violations.append(Violation(
            rule="hotel_nightly_cap",
            detail=f"${q.nightly_rate_usd:,.0f}/night exceeds the {match or 'default'} cap of ${cap:,}.",
        ))
    return HotelPolicyResult(
        company=policy["company"], city=match or q.city, nightly_cap_usd=cap,
        compliant=not violations, violations=violations,
    ).model_dump()
