#!/usr/bin/env python3
"""Collect the four good RESULT_JSON records (0.4.3 from the orchestrate combined
file, 0.4.11 from the two native-stack canary logs) into final_runs.jsonl."""
import json, os, glob

keep = []
seen = set()

def add(d):
    k = (d["lm_eval_version"], d["model_id"])
    if d.get("status") == "ok" and d.get("score") is not None and k not in seen:
        seen.add(k); keep.append(d)

# 0.4.3 results live in the orchestrate combined file
p = "results/all_version_runs.jsonl"
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if line:
            add(json.loads(line))

# 0.4.11 results live in the native canary logs (RESULT_JSON: lines)
for log in glob.glob("results/c411*.log"):
    for line in open(log):
        if line.startswith("RESULT_JSON:"):
            add(json.loads(line[len("RESULT_JSON:"):]))

with open("results/final_runs.jsonl", "w") as f:
    for d in keep:
        f.write(json.dumps(d) + "\n")
print(f"wrote results/final_runs.jsonl with {len(keep)} rows")
for d in keep:
    print(f"  {d['lm_eval_version']:>7} | {d['model_id']:<30} | {d['score']} | {d['metric']}")
