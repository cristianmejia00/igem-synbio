"""LLM naming of the mid / high hierarchy groups via OpenAI function calling.

Each group is shown to the model with the low-level sub-topics (name +
description) it contains; the model returns one unique name per group.
"""
import json

import pandas as pd
import yaml
from openai import OpenAI

from .paths import MODELS_DIR, OPENAI_KEY_PATH, OPENAI_MODEL, PROMPTS_PATH, REPORTS_DIR


def load_prompts(path=PROMPTS_PATH) -> dict:
    """Load the theme definition and hierarchy-naming system prompt template."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_client(key_path=OPENAI_KEY_PATH) -> OpenAI:
    """Read the API key file and return an authenticated OpenAI client."""
    return OpenAI(api_key=open(key_path).read().strip())


def build_system_prompt(prompts: dict) -> str:
    """Fill the hierarchy-naming system template with the theme definition."""
    return prompts["hierarchy_naming"]["system"].format(
        theme=prompts["theme"], theme_description=prompts["theme_description"],
    )


def build_rename_tool(n_groups: int) -> dict:
    """OpenAI function-calling tool schema for naming hierarchy groups."""
    return {
        "type": "function",
        "function": {
            "name": "name_groups",
            "description": (
                f"Assign a unique, concise, publication-ready name to each of "
                f"the {n_groups} topic groups. Every name must be distinct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "description": f"Exactly {n_groups} named groups.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "group_id": {
                                    "type": "integer",
                                    "description": "The numeric group ID.",
                                },
                                "name": {
                                    "type": "string",
                                    "description": (
                                        "A short, unique name for this group. "
                                        "Must be distinct from every other group name."
                                    ),
                                },
                            },
                            "required": ["group_id", "name"],
                        },
                    },
                },
                "required": ["groups"],
            },
        },
    }


def build_user_message(hierarchy_df: pd.DataFrame, topic_names: pd.DataFrame, level_col: str) -> str:
    """Format groups for the LLM: each group lists its sub-topic names & descriptions."""
    merged = hierarchy_df.merge(
        topic_names[["low", "global_name", "description"]], on="low", how="left",
    )
    lines = []
    for gid, grp in merged.groupby(level_col):
        if int(gid) < 0:
            continue
        lines.append(f"Group {int(gid)}:")
        for _, row in grp.iterrows():
            name = row.get("global_name", row.get("name", "unknown"))
            desc = row.get("description", "")
            lines.append(f"  - {name}: {desc}")
        lines.append("")
    return "\n".join(lines)


def name_hierarchy_level(hierarchy_df, topic_names, level_col, label, client,
                         system_prompt, model=OPENAI_MODEL) -> dict[int, str]:
    """Call OpenAI to name all groups at one hierarchy level. Returns {group_id: name}."""
    n_groups = hierarchy_df[hierarchy_df[level_col] >= 0][level_col].nunique()
    print(f"  Naming {n_groups} {label} groups via {model} …")

    tool = build_rename_tool(n_groups)
    user_msg = build_user_message(hierarchy_df, topic_names, level_col)

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "name_groups"}},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )

    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    mapping = {item["group_id"]: item["name"] for item in args["groups"]}

    expected_ids = set(hierarchy_df[hierarchy_df[level_col] >= 0][level_col].unique())
    missing = expected_ids - set(mapping.keys())
    if missing:
        print(f"  WARNING: {len(missing)} group(s) did not receive a name: {missing}")
    dupes = len(mapping.values()) - len(set(mapping.values()))
    if dupes:
        print(f"  WARNING: {dupes} duplicate name(s) found")

    print(f"  ✓ {len(mapping)} {label} names assigned")
    return mapping


def load_naming_inputs(prefix, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR):
    """Load the low-level topic names and the hierarchy table for one corpus."""
    topic_names = pd.read_csv(models_dir / f"{prefix}_topic_names.txt", sep="\t").rename(columns={"topic": "low"})
    hierarchy = pd.read_csv(reports_dir / f"{prefix}_topic_name_hierarchy.tsv", sep="\t")
    return topic_names, hierarchy


def save_named_hierarchy(hierarchy, prefix, reports_dir=REPORTS_DIR):
    """Overwrite the hierarchy table with the added mid_name / high_name columns."""
    hierarchy.to_csv(reports_dir / f"{prefix}_topic_name_hierarchy.tsv", sep="\t", index=False)
