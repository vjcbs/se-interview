"""Evaluate the travel agent's traces in Phoenix and build a dataset of frustrated interactions.

Usage (with Phoenix running on localhost:6006 and OPENAI_API_KEY in .env):
    poetry run python scripts/run_evals.py

What it does:
1. Pulls the root span of every conversation turn from Phoenix (one per user message).
2. Rebuilds each turn: conversation so far, the user's message, the agent's reply,
   and any policy-tool results from that turn.
3. Runs three LLM-as-judge evals:
     user_frustration   - Phoenix's built-in UserFrictionEvaluator: does this user message
                          express frustration/friction with the agent's previous behavior?
     frustration_cause  - (frustrated turns only) was it an agent error, a policy/design
                          problem, or the user's side?
     policy_faithfulness- (turns with a policy check only) does the reply describe the policy
                          tool's result accurately, or misstate it?
4. Attaches every result to its span as an annotation (visible in the Phoenix UI).
5. Creates a Phoenix dataset of the frustrated interactions.
6. Compares user_frustration with hand labels for the scripted test queries.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.evals import LLM, create_classifier, evaluate_dataframe
from phoenix.evals.metrics.user_friction import UserFrictionEvaluator
from phoenix.evals.utils import to_annotation_dataframe

load_dotenv()

PROJECT = "arize-travel-agent"
JUDGE_MODEL = "gpt-4o"
OUT_DIR = Path(__file__).resolve().parent.parent / "exports"
POLICY_TOOLS = {"check_flight_policy", "check_hotel_policy"}

# Hand labels for the scripted test queries (scripts/run_queries.py), used to sanity-check the judge.
EXPECTED_FRUSTRATION = {
    "Two stops and 11 hours? That's ridiculous for a client meeting. Give me the nonstop.": "friction",
    "I already said nonstop. Why do you keep recommending the United flight?": "friction",
    "Forget it. Just tell me if premium economy on the nonstop is allowed.": "friction",
}

CAUSE_PROMPT = """You are reviewing a conversation between an employee and a company travel assistant
that must only recommend options compliant with company travel policy.

The user's latest message expresses frustration. Decide what caused it, based on the
assistant's behavior BEFORE that message.

- agent_error: the assistant misunderstood the request, ignored an instruction or constraint
  the user gave, stated something wrong or made up, or failed to do what was asked.
- policy_or_design: the assistant followed the policy and its tools correctly, but the
  outcome was unreasonable for the user (e.g. a technically compliant but impractical option).
- user_side: the assistant did nothing wrong; the user misremembers, changed their mind,
  or is reacting to something the assistant did not do.

<conversation_before_user_message>
{{conversation}}
</conversation_before_user_message>

<latest_user_message>
{{user_message}}
</latest_user_message>

Reason step by step, then choose one label."""

FAITHFULNESS_PROMPT = """You are checking whether a travel assistant's reply accurately reflects the
results returned by its company-policy tool.

<policy_tool_results>
{{tool_results}}
</policy_tool_results>

<assistant_reply>
{{agent_response}}
</assistant_reply>

- faithful: every compliance verdict, violated rule, number and reason the reply states about
  policy matches the tool results (omitting details is fine).
- unfaithful: the reply contradicts the tool results, inverts or misstates a rule or reason,
  or claims compliance/non-compliance the tool did not return.

Reason step by step, then choose one label."""


# ---------- Rebuild turns from root spans ----------


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        content = m["data"].get("content") or ""
        if m["type"] == "human":
            lines.append(f"User: {content}")
        elif m["type"] == "ai" and content:
            lines.append(f"Assistant: {content}")
    return "\n\n".join(lines)


def parse_turn(output_value: str) -> dict:
    """Split the graph's output (full conversation state) into the parts of the latest turn."""
    messages = json.loads(output_value)["messages"]
    last_human = max(i for i, m in enumerate(messages) if m["type"] == "human")
    turn = messages[last_human + 1:]

    tool_names = {}
    for m in turn:
        for call in m["data"].get("tool_calls") or []:
            tool_names[call["id"]] = call["name"]
    tools_used = [tool_names.get(m["data"].get("tool_call_id"), "?") for m in turn if m["type"] == "tool"]
    policy_results = [
        m["data"]["content"] for m in turn
        if m["type"] == "tool" and tool_names.get(m["data"].get("tool_call_id")) in POLICY_TOOLS
    ]
    final = next((m["data"]["content"] for m in reversed(turn) if m["type"] == "ai" and m["data"]["content"]), "")

    return {
        "conversation": _render(messages[:last_human]) or "(start of conversation)",
        "user_message": messages[last_human]["data"]["content"],
        "agent_response": final,
        "tools_used": ", ".join(tools_used),
        "tool_results": "\n".join(policy_results),
    }


def load_turns(client: Client) -> pd.DataFrame:
    spans = client.spans.get_spans_dataframe(
        query=SpanQuery().where("parent_id is None"), project_identifier=PROJECT, limit=10000, timeout=30
    )
    spans = spans[spans["name"] == "LangGraph"]
    rows = []
    for span_id, s in spans.iterrows():
        try:
            turn = parse_turn(s["attributes.output.value"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue  # skip spans without a parsable conversation (e.g. failed requests)
        turn["span_id"] = s.get("context.span_id", span_id)
        turn["session_id"] = s.get("attributes.session.id")
        turn["start_time"] = s["start_time"]
        rows.append(turn)
    return pd.DataFrame(rows).sort_values("start_time").set_index("span_id")


# ---------- Main ----------


def main():
    client = Client()
    judge = LLM(provider="openai", model=JUDGE_MODEL)

    turns = load_turns(client)
    print(f"Loaded {len(turns)} conversation turns from Phoenix project '{PROJECT}'.")

    # 1. User frustration on every turn (Phoenix built-in evaluator)
    frustration = evaluate_dataframe(dataframe=turns, evaluators=[UserFrictionEvaluator(llm=judge)], hide_tqdm_bar=True)
    turns["frustration_label"] = frustration["user_friction_score"].map(lambda s: s["label"] if s else None)
    turns["frustration_explanation"] = frustration["user_friction_score"].map(lambda s: s.get("explanation") if s else None)
    annotations = [to_annotation_dataframe(dataframe=frustration, score_names=["user_friction"]).assign(annotation_name="user_frustration")]

    # 2. Cause of frustration (only where frustration was found)
    frustrated = turns[turns["frustration_label"] == "friction"]
    if len(frustrated):
        cause_eval = create_classifier(
            name="frustration_cause", prompt_template=CAUSE_PROMPT, llm=judge,
            choices={"agent_error": 1.0, "policy_or_design": 0.5, "user_side": 0.0}, direction="minimize",
        )
        causes = evaluate_dataframe(dataframe=frustrated, evaluators=[cause_eval], hide_tqdm_bar=True)
        turns.loc[causes.index, "frustration_cause"] = causes["frustration_cause_score"].map(lambda s: s["label"] if s else None)
        annotations.append(to_annotation_dataframe(dataframe=causes, score_names=["frustration_cause"]))

    # 3. Policy faithfulness (only turns where a policy tool ran)
    checked = turns[turns["tool_results"] != ""]
    if len(checked):
        faith_eval = create_classifier(
            name="policy_faithfulness", prompt_template=FAITHFULNESS_PROMPT, llm=judge,
            choices={"faithful": 1.0, "unfaithful": 0.0}, direction="maximize",
        )
        faith = evaluate_dataframe(dataframe=checked, evaluators=[faith_eval], hide_tqdm_bar=True)
        turns.loc[faith.index, "policy_faithfulness"] = faith["policy_faithfulness_score"].map(lambda s: s["label"] if s else None)
        annotations.append(to_annotation_dataframe(dataframe=faith, score_names=["policy_faithfulness"]))

    # 4. Attach all results to the spans in Phoenix
    for ann in annotations:
        ann = ann.dropna(subset=["label"])
        if len(ann):
            client.spans.log_span_annotations_dataframe(dataframe=ann, annotator_kind="LLM")
    print("Logged annotations to Phoenix: user_frustration, frustration_cause, policy_faithfulness.")

    # 5. Dataset of frustrated interactions
    frustrated = turns[turns["frustration_label"] == "friction"].reset_index()
    if len(frustrated):
        name = f"frustrated-interactions-{datetime.now():%Y%m%d-%H%M}"
        client.datasets.create_dataset(
            name=name,
            dataframe=frustrated,
            input_keys=["conversation", "user_message"],
            output_keys=["agent_response"],
            metadata_keys=["session_id", "frustration_explanation", "frustration_cause"],
            span_id_key="span_id",
            dataset_description="User turns the user_frustration eval labeled as frustrated.",
        )
        print(f"Created Phoenix dataset '{name}' with {len(frustrated)} examples.")

    # 6. Results summary and agreement with hand labels
    OUT_DIR.mkdir(exist_ok=True)
    turns.to_csv(OUT_DIR / "eval_results.csv")
    print(f"\nSaved per-turn results to {OUT_DIR / 'eval_results.csv'}\n")
    print("user_frustration:", turns["frustration_label"].value_counts().to_dict())
    if "frustration_cause" in turns:
        print("frustration_cause:", turns["frustration_cause"].dropna().value_counts().to_dict())
    if "policy_faithfulness" in turns:
        print("policy_faithfulness:", turns["policy_faithfulness"].dropna().value_counts().to_dict())

    labeled = turns[turns["user_message"].isin(EXPECTED_FRUSTRATION) | turns["conversation"].eq("(start of conversation)")]
    if len(labeled):
        expected = labeled["user_message"].map(EXPECTED_FRUSTRATION).fillna("no_friction")
        agree = (expected == labeled["frustration_label"]).mean()
        print(f"\nJudge vs hand labels: {agree:.0%} agreement on {len(labeled)} scripted turns.")
        for msg, exp, got in zip(labeled["user_message"], expected, labeled["frustration_label"]):
            if exp != got:
                print(f"  MISMATCH expected={exp} got={got}: {msg[:80]}")

    print("\nFrustrated / unfaithful turns:")
    flagged = turns[(turns["frustration_label"] == "friction") | (turns.get("policy_faithfulness") == "unfaithful")]
    for _, r in flagged.iterrows():
        print(f"- {r['user_message'][:90]}")
        print(f"    frustration={r['frustration_label']} cause={r.get('frustration_cause')} "
              f"faithfulness={r.get('policy_faithfulness')}")


if __name__ == "__main__":
    main()
