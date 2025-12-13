from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from datasets import load_dataset, Dataset, DatasetDict


TLDR_FORMAT = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
CNNDM_FORMAT = "Article:\n{article}\n\nTL;DR:"
CNNDM_BATCHES = {"batch0_cnndm", "cnndm0", "cnndm2"}


@dataclass
class TLDRDatasetConfig:
    include_cnndm: bool = False
    pref_split: Literal["train", "validation"] = "train"
    sft_split: Literal["train", "validation", "test"] = "train"
    sft_from_axis: bool = False
    axis_split: Literal["validation", "test"] = "validation"


def _format_prompt_from_info(info: dict, batch: Optional[str]) -> str:
    if batch in CNNDM_BATCHES:
        return CNNDM_FORMAT.format(**info)
    return TLDR_FORMAT.format(**info)


def load_tldr_preferences(cfg: TLDRDatasetConfig = TLDRDatasetConfig()) -> Dataset:
    split_name = "validation" if cfg.pref_split == "validation" else "train"
    ds = load_dataset("trl-lib/tldr-preference", split=split_name)
    def process(row):
        return {
            "prompt_text": row["prompt"],
            "chosen_text": row["chosen"],
            "rejected_text": row["rejected"],
        }
    ds = ds.map(process, remove_columns=ds.column_names)
    return ds


def load_tldr_sft(cfg: TLDRDatasetConfig = TLDRDatasetConfig()) -> Dataset:
    if not cfg.sft_from_axis:
        ds = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered", split=cfg.sft_split)
        def process(row):
            prompt = (
                f"SUBREDDIT: r/{row['subreddit']}\n\n"
                f"TITLE: {row['title']}\n\n"
                f"POST: {row['post']}\n\n"
                f"TL;DR:"
            )
            completion = "\n" + row["summary"].strip()
            return {"prompt_text": prompt, "target_text": completion}
        ds = ds.map(process, remove_columns=ds.column_names)
        return ds
    ds = load_dataset("openai/summarize_from_feedback", "axis", split=cfg.axis_split)
    def process(row):
        batch = row.get("batch")
        if (not cfg.include_cnndm) and (batch in CNNDM_BATCHES):
            return {"_drop": True}
        prompt = _format_prompt_from_info(row["info"], batch)
        summary_text = row["summary"]["text"]
        return {"_drop": False, "prompt_text": prompt, "target_text": summary_text}
    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: not x["_drop"])
    ds = ds.remove_columns(["_drop"])
    return ds

def load_tldr_preferences_for_trainer(pref_split="train", include_cnndm=False):
    cfg = TLDRDatasetConfig(pref_split=pref_split, include_cnndm=include_cnndm)
    ds = load_tldr_preferences(cfg)
    return ds.rename_columns({
        "prompt_text": "prompt",
        "chosen_text": "chosen",
        "rejected_text": "rejected",
    })

def load_tldr_sft_for_trainer(sft_split="train", include_cnndm=False, **kwargs):
    cfg = TLDRDatasetConfig(sft_split=sft_split, include_cnndm=include_cnndm)
    ds = load_tldr_sft(cfg)
    return ds.rename_columns({
        "prompt_text": "prompt",
        "target_text": "completion", 
    })

def load_all(cfg: TLDRDatasetConfig = TLDRDatasetConfig()) -> DatasetDict:
    return DatasetDict(
        {
            "preference": load_tldr_preferences(cfg),
            "sft": load_tldr_sft(cfg),
        }
    )
