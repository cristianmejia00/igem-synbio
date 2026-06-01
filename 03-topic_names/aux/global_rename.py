"""Function-calling pass that assigns globally unique topic names (part 2).

Per-cluster naming in part 1 works locally: each topic is named in isolation,
which can produce duplicate or ambiguous names. Here the LLM is given a global
view of all topics at once and returns a structured array of ``(topic_id, name)``
pairs via OpenAI function calling, guaranteeing distinct names.
"""
import json

import pandas as pd

from .paths import OPENAI_MODEL


def build_rename_tool(n_topics: int) -> dict:
    """Build the OpenAI function-calling tool schema for renaming n topics."""
    return {
        "type": "function",
        "function": {
            "name": "rename_topics",
            "description": (
                f"Assign a unique, concise, publication-ready name to each of "
                f"the {n_topics} topic clusters. Every name must be distinct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "description": f"Exactly {n_topics} renamed topics.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic_id": {
                                    "type": "integer",
                                    "description": "The original topic ID.",
                                },
                                "name": {
                                    "type": "string",
                                    "description": (
                                        "A short, unique name for this topic "
                                        "Must be distinct from "
                                        "every other topic name."
                                    ),
                                },
                            },
                            "required": ["topic_id", "name"],
                        },
                    },
                },
                "required": ["topics"],
            },
        },
    }


def build_user_message(names_df: pd.DataFrame) -> str:
    """Format all topics into a single user message for the LLM."""
    lines = []
    for _, row in names_df.iterrows():
        lines.append(
            f"Topic {int(row['topic'])}:\n"
            f"  Current name: {row['name']}\n"
            f"  Description: {row['description']}\n"
        )
    return "\n".join(lines)


def build_system_prompt(theme: str, theme_description: str) -> str:
    """Build the system prompt instructing the model to produce unique names."""
    return (
        f"You are a research consultant with expertise on {theme}, meaning that "
        f"you know about {theme_description}\n\n"
        f"You will receive a list of topic clusters, each with an ID, a current "
        f"name, and a description. Some current names may be duplicated or "
        f"ambiguous. Your task is to assign a NEW, UNIQUE, short name to each "
        f"topic that:\n"
        f"1. Clearly distinguishes it from every other topic in the list.\n"
        f"2. Faithfully reflects the description.\n"
        f"3. Is concise and suitable for use in an academic publication.\n"
        f"4. No two names should overlap or be paraphrases of each other.\n\n"
        f"Return ALL topics, even if the current name is already good — confirm "
        f"or improve each one."
    )


def rename_topics_global(names_df, client, prompts, model=OPENAI_MODEL) -> pd.DataFrame:
    """Call OpenAI with function calling to get globally distinct topic names.

    Returns a copy of ``names_df`` with a new ``global_name`` column.
    """
    n = len(names_df)
    tool = build_rename_tool(n)
    user_msg = build_user_message(names_df)
    system_prompt = build_system_prompt(prompts["theme"], prompts["theme_description"])

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "rename_topics"}},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )

    # Parse the function call arguments
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    renamed = {item["topic_id"]: item["name"] for item in args["topics"]}

    result = names_df.copy()
    result["global_name"] = result["topic"].map(renamed)

    # Verify all topics got a name
    missing = result["global_name"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} topic(s) did not receive a global name")

    # Verify uniqueness
    dupes = result["global_name"].duplicated().sum()
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate global name(s) found")

    return result
