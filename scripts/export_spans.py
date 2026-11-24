"""Export every span of the travel-agent project from Phoenix to files.

Usage (with Phoenix running on localhost:6006):
    poetry run python scripts/export_spans.py

Writes:
    exports/spans.csv    - one row per span (LLM calls, tool calls, graph steps), easy to open
    exports/spans.jsonl  - same data, one JSON object per line, lossless
"""

from pathlib import Path

from phoenix.client import Client

PROJECT = "arize-travel-agent"
OUT_DIR = Path(__file__).resolve().parent.parent / "exports"


def main():
    client = Client()  # defaults to http://localhost:6006
    df = client.spans.get_spans_dataframe(project_identifier=PROJECT, limit=10000, timeout=30)

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "spans.csv")
    df.to_json(OUT_DIR / "spans.jsonl", orient="records", lines=True, date_format="iso")

    traces = df["context.trace_id"].nunique()
    print(f"Exported {len(df)} spans from {traces} traces to {OUT_DIR}/")
    print("\nSpans by kind:")
    print(df["span_kind"].value_counts().to_string())
    print("\nTool calls:")
    print(df.loc[df["span_kind"] == "TOOL", "name"].value_counts().to_string())


if __name__ == "__main__":
    main()
