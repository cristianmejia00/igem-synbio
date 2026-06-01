"""Per-cluster description and name generation (part 1)."""
import pandas as pd

from .openai_client import ask_gpt, fmt_prompt
from .paths import OPENAI_MODEL, TOP_N_DOCS


def get_representative_texts(df, topic_id, top_n=TOP_N_DOCS, max_chars=14_000):
    """Return the top-N documents of a topic joined with ##### separators.

    The text is truncated to ``max_chars`` (~3 500 tokens) to stay within
    context limits.
    """
    subset = df[df["topic"] == topic_id]
    texts = subset["text"].head(top_n).tolist()
    bulk = " ##### ".join(texts)
    return bulk[:max_chars]


def name_topics(df, topic_info, client, prompts, model=OPENAI_MODEL, top_n=TOP_N_DOCS):
    """For each non-outlier topic, get description, enhanced description, and name.

    Returns a DataFrame with columns: topic, name, description, raw_description.
    """
    theme = prompts["theme"]
    theme_description = prompts["theme_description"]
    p_desc = prompts["cluster_description"]
    p_enh = prompts["cluster_description_enhanced"]
    p_name = prompts["cluster_name"]

    rows = []
    topic_ids = sorted(topic_info[topic_info["Topic"] != -1]["Topic"].unique())

    for tid in topic_ids:
        print(f"  Topic {tid} ...", end=" ", flush=True)
        cluster_text = get_representative_texts(df, tid, top_n=top_n)

        # Step 1: cluster description
        description = ask_gpt(
            client,
            fmt_prompt(p_desc["system"], theme=theme, theme_description=theme_description),
            fmt_prompt(p_desc["user"], cluster_text=cluster_text),
            model=model,
            temperature=0.2,
        )

        # Step 2: enhanced (synthesised) description
        enhanced = ask_gpt(
            client,
            fmt_prompt(p_enh["system"], theme=theme),
            fmt_prompt(p_enh["user"], cluster_description=description),
            model=model,
            temperature=0.1,
        )

        # Step 3: short name
        name = ask_gpt(
            client,
            fmt_prompt(p_name["system"], theme=theme, theme_description=theme_description),
            fmt_prompt(p_name["user"], cluster_description=description),
            model=model,
            max_tokens=60,
            temperature=0.3,
        )
        name = name.strip().strip('"').strip("'")

        print(name)
        rows.append({
            "topic": tid,
            "name": name,
            "description": enhanced,
            "raw_description": description,
        })

    return pd.DataFrame(rows)
