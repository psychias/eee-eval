"""Check eval_library.version coverage across all data sources."""
import json, os
from pathlib import Path
from collections import defaultdict

data_dir = Path('data')

for source_dir in sorted(data_dir.iterdir()):
    if not source_dir.is_dir() or source_dir.name == 'aggregated':
        continue
    
    # Find all JSON files recursively
    jsons = list(source_dir.rglob('*.json'))
    if not jsons:
        continue
    
    versions_found = defaultdict(int)
    names_found = defaultdict(int)
    total = 0
    
    for jf in jsons[:500]:  # sample up to 500 per source
        try:
            d = json.loads(jf.read_text(encoding='utf-8'))
        except:
            continue
        
        if not isinstance(d, dict):
            continue
        
        el = d.get('eval_library', {})
        if isinstance(el, dict):
            total += 1
            ver = el.get('version', 'FIELD_MISSING')
            name = el.get('name', 'FIELD_MISSING')
            versions_found[ver] += 1
            names_found[name] += 1
    
    print(f"\n{source_dir.name} (sampled {total} files):")
    print(f"  Harness names: {dict(names_found)}")
    print(f"  Versions: {dict(versions_found)}")

