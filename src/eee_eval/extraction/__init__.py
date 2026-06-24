"""
eee_eval.extraction — EEE evaluation data extraction package.

Modules
-------
constants           Canonical benchmark names, developer maps, shared inference.
add_leaderboard_records   Fetch live leaderboard scores and write EEE records.
hf_model_card_fetcher     Extract benchmark results from HuggingFace model cards.
pwc_fetcher               Fetch Papers With Code evaluation results.
extract_paper               LLM-assisted extraction from arXiv PDFs (main CLI).
convert_eval_logs         Convert raw evaluation logs to EEE format.
fetch_arxiv_papers        Download arXiv PDFs.
run_hf_pipeline           Orchestrate HF-based extraction stages.
"""

from eee_eval.extraction.constants import infer_developer

__all__ = ["infer_developer"]
