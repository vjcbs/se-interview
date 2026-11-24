"""Send a set of test conversations to the travel agent so they show up as traces in Phoenix.

Usage (with the app running on localhost:8000 and Phoenix on localhost:6006):
    poetry run python scripts/run_queries.py

Each conversation runs in its own session, so multi-turn conversations are grouped
under one session in Phoenix. The scenarios are designed to cover:
  A. happy paths, B. policy violations, C. error handling, D. a frustrated user.
"""

import json
import time
import urllib.request

API_URL = "http://localhost:8000/chat"

CONVERSATIONS = [
    # A. Happy paths
    ("A1 flights in policy", [
        "I need to fly from New York to Lisbon for a client meeting Dec 15-18. What are my options?",
    ]),
    ("A2 hotel budget", [
        "What's my hotel budget in London, and which neighborhoods are good for business travelers?",
    ]),
    ("A3 attractions", [
        "What are some good things to do near Lisbon's business district in the evening?",
    ]),
    # B. Policy violations
    ("B1 business class", [
        "Book me business class from SFO to London on Dec 10.",
    ]),
    ("B2 too late to book", [
        "I need a flight from JFK to Chicago on Nov 27.",
    ]),
    ("B3 hotel over cap", [
        "Is a $260 per night hotel in Lisbon OK?",
    ]),
    # C. Error handling
    ("C1 same airport", [
        "Find me flights from JFK to JFK tomorrow.",
    ]),
    ("C2 missing details", [
        "Find me a flight to Tokyo.",
    ]),
    # D. Frustrated user (one multi-turn session)
    ("D1 frustrated traveler", [
        "Flights from NYC to Lisbon, Dec 15-18.",
        "Two stops and 11 hours? That's ridiculous for a client meeting. Give me the nonstop.",
        "I already said nonstop. Why do you keep recommending the United flight?",
        "Forget it. Just tell me if premium economy on the nonstop is allowed.",
    ]),
]


def ask(message: str, session_id: str | None) -> dict:
    body = json.dumps({"message": message, "session_id": session_id}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    total = 0
    for name, turns in CONVERSATIONS:
        print(f"\n=== {name} ===")
        session_id = None
        for message in turns:
            start = time.time()
            result = ask(message, session_id)
            session_id = result["session_id"]
            total += 1
            print(f"\nUSER: {message}")
            print(f"AGENT ({time.time() - start:.1f}s): {result['response']}")
        print(f"\n(session {session_id})")
    print(f"\nDone: {total} queries sent across {len(CONVERSATIONS)} conversations.")


if __name__ == "__main__":
    main()
