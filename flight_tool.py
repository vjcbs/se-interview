"""Flight search tool for the LangGraph travel agent.

Design:
- Typed input schema (Pydantic) so the LLM must supply IATA codes + ISO dates.
- Typed output schema with an explicit `status` so failures are visible in traces
  and easy to evaluate (ok / no_results / error).
- Mock provider by default: deterministic (seeded by route + date) so demo runs
  are repeatable and never fail on a third-party API. A live provider can be
  plugged in behind FLIGHT_PROVIDER without changing the agent.
"""

import hashlib
import os
import random
from datetime import date, datetime, timedelta
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------- Schemas ----------


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin airport IATA code, e.g. 'JFK'")
    destination: str = Field(description="Destination airport IATA code, e.g. 'LIS'")
    departure_date: str = Field(description="Departure date, YYYY-MM-DD")
    return_date: Optional[str] = Field(None, description="Return date YYYY-MM-DD for round trips")
    adults: int = Field(1, ge=1, le=9, description="Number of adult passengers")
    cabin: Literal["economy", "premium_economy", "business", "first"] = "economy"
    max_stops: Optional[int] = Field(None, ge=0, le=2, description="Max stops; 0 = nonstop only")


class FlightOption(BaseModel):
    airline: str
    flight_number: str
    depart_time: str  # local time at origin
    duration_minutes: int
    stops: int
    price_usd: float  # total for all passengers, one way or round trip


class FlightSearchResult(BaseModel):
    status: Literal["ok", "no_results", "error"]
    source: Literal["mock", "live"]
    query: FlightSearchInput
    international: Optional[bool] = None
    options: list[FlightOption] = []
    message: Optional[str] = None


# ---------- Mock provider ----------

AIRLINES = {
    "AA": "American Airlines", "DL": "Delta", "UA": "United", "B6": "JetBlue",
    "BA": "British Airways", "LH": "Lufthansa", "AF": "Air France", "TP": "TAP Air Portugal",
}
# Realistic nonstop durations (minutes) and typical economy one-way fares (USD).
# Keyed by sorted airport pair so JFK-LIS and LIS-JFK match.
ROUTES = {
    ("JFK", "LIS"): (420, 520, True),
    ("JFK", "LHR"): (420, 600, True),
    ("JFK", "CDG"): (445, 580, True),
    ("LHR", "SFO"): (630, 750, True),
    ("JFK", "NRT"): (845, 1100, True),
    ("JFK", "SFO"): (385, 330, False),
    ("JFK", "LAX"): (370, 300, False),
    ("JFK", "ORD"): (165, 190, False),
    ("BOS", "JFK"): (75, 140, False),
}
US_AIRPORTS = {"JFK", "LGA", "EWR", "SFO", "LAX", "ORD", "BOS", "SEA", "ATL", "DFW", "MIA", "DCA", "IAD"}
CABIN_MULTIPLIER = {"economy": 1.0, "premium_economy": 1.6, "business": 3.5, "first": 6.0}


def _validate(q: FlightSearchInput) -> Optional[str]:
    for code in (q.origin, q.destination):
        if len(code) != 3 or not code.isalpha():
            return f"'{code}' is not a valid IATA airport code (expected 3 letters, e.g. JFK)."
    if q.origin.upper() == q.destination.upper():
        return "Origin and destination are the same airport."
    try:
        dep = datetime.strptime(q.departure_date, "%Y-%m-%d").date()
        ret = datetime.strptime(q.return_date, "%Y-%m-%d").date() if q.return_date else None
    except ValueError:
        return "Dates must be in YYYY-MM-DD format."
    if dep < date.today():
        return f"Departure date {q.departure_date} is in the past."
    if ret and ret < dep:
        return "Return date is before departure date."
    return None


def _mock_search(q: FlightSearchInput) -> FlightSearchResult:
    seed = int(hashlib.md5(f"{q.origin}{q.destination}{q.departure_date}".encode()).hexdigest(), 16)
    rng = random.Random(seed)

    route = ROUTES.get(tuple(sorted((q.origin, q.destination))))
    if route:
        base_minutes, base_price, international = route
    else:  # unknown route: plausible random values
        international = not (q.origin in US_AIRPORTS and q.destination in US_AIRPORTS)
        base_minutes = rng.randint(400, 800) if international else rng.randint(90, 360)
        base_price = 120 + base_minutes * 0.9
    trip_factor = 1.85 if q.return_date else 1.0

    options = []
    for _ in range(rng.randint(4, 6)):
        code = rng.choice(list(AIRLINES))
        stops = rng.choices([0, 1, 2], weights=[5, 4, 1])[0]
        duration = base_minutes + rng.randint(-15, 20) + stops * rng.randint(70, 180)
        depart = datetime.strptime(q.departure_date, "%Y-%m-%d") + timedelta(
            hours=rng.randint(6, 22), minutes=rng.choice([0, 15, 30, 45])
        )
        price = base_price * CABIN_MULTIPLIER[q.cabin] * trip_factor * q.adults
        price *= rng.uniform(0.8, 1.5) * (0.85 if stops else 1.0)
        options.append(
            FlightOption(
                airline=AIRLINES[code],
                flight_number=f"{code}{rng.randint(100, 9999)}",
                depart_time=depart.isoformat(timespec="minutes"),
                duration_minutes=duration,
                stops=stops,
                price_usd=round(price, 2),
            )
        )

    if q.max_stops is not None:
        options = [o for o in options if o.stops <= q.max_stops]
    options.sort(key=lambda o: o.price_usd)

    if not options:
        return FlightSearchResult(
            status="no_results", source="mock", query=q, international=international,
            message="No flights match these filters. Try allowing more stops or other dates.",
        )
    return FlightSearchResult(
        status="ok", source="mock", query=q, international=international, options=options[:5]
    )


# ---------- Tool ----------


@tool("search_flights", args_schema=FlightSearchInput)
def search_flights(**kwargs) -> dict:
    """Search for flights between two airports on given dates.

    Use this whenever the user asks about flights, airfare, or getting from one
    city to another by air. Convert city names to IATA codes first
    (New York -> JFK, Lisbon -> LIS). Returns up to 5 options sorted by price (total for all passengers),
    plus whether the route is international.
    """
    q = FlightSearchInput(**kwargs)
    q.origin, q.destination = q.origin.upper(), q.destination.upper()

    error = _validate(q)
    if error:
        return FlightSearchResult(status="error", source="mock", query=q, message=error).model_dump()

    provider = os.getenv("FLIGHT_PROVIDER", "mock")
    if provider == "mock":
        return _mock_search(q).model_dump()
    # Placeholder for a live provider (e.g. Duffel test mode or SerpApi Google Flights).
    return FlightSearchResult(
        status="error", source="live", query=q, message=f"Provider '{provider}' not configured."
    ).model_dump()
