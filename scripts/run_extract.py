"""Wrapper to run extract_paper.py on all papers from arxiv_ids_full.txt,
one at a time, capturing output cleanly."""
import subprocess
import sys
import os
import re

def main():
    ids_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/arxiv_ids_full.txt"
    model = sys.argv[2] if len(sys.argv) > 2 else "anthropic/claude-haiku-4.5"
    
    # Parse arxiv IDs
    arxiv_ids = []
    with open(ids_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            aid = line.split()[0]  # take first token (ID before comment)
            arxiv_ids.append(aid)

    print(f"=== Extracting {len(arxiv_ids)} papers with {model} ===\n")
    
    results = {}
    for i, aid in enumerate(arxiv_ids, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(arxiv_ids)}] Processing {aid}")
        print(f"{'='*60}")
        
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [
            sys.executable, "-u", "scripts/extract_paper.py",
            "--arxiv_id", aid,
            "--llm-fallback",
            "--llm-model", model,
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=False, text=True, env=env,
                timeout=1200,  # 20 min max per paper
            )
            results[aid] = "OK" if result.returncode == 0 else f"exit={result.returncode}"
        except subprocess.TimeoutExpired:
            results[aid] = "TIMEOUT"
            print(f"  *** TIMEOUT after 600s for {aid}")
        except Exception as e:
            results[aid] = f"ERROR: {e}"
            print(f"  *** ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"=== EXTRACTION SUMMARY ===")
    print(f"{'='*60}")
    for aid, status in results.items():
        print(f"  {aid}: {status}")
    
    # Count files
    import glob
    files = glob.glob("data/**/*.json", recursive=True)
    print(f"\nTotal JSON files: {len(files)}")

if __name__ == "__main__":
    main()
