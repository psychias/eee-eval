"""Check Tulu first-party direction."""
import pandas as pd

cp = pd.read_csv('analysis_output/collision_pairs.csv')
tulu = cp[cp['model_id_norm'].str.contains('tulu-3-8b', case=False, na=False)]
tulu = tulu[~tulu['model_id_norm'].str.contains('70b', case=False, na=False)]

print("Tulu-3-8B collision pairs:")
for _, row in tulu.iterrows():
    d = row['delta']
    fp = 'FP higher' if d > 0 else 'FP lower' if d < 0 else 'equal'
    print(f"  {row['benchmark_norm']:12s}: delta={d:+.2f} "
          f"score_a({row['source_a']})={row['score_a']:.2f} "
          f"score_b({row['source_b']})={row['score_b']:.2f} -> {fp}")

print(f"\nMean delta (signed): {tulu['delta'].mean():.3f}")
print(f"  -> This is {'+' if tulu['delta'].mean() > 0 else ''}{tulu['delta'].mean():.3f}")
print(f"  -> If source_a = hf_model_card (first-party), positive = FP advantage")
print(f"Mean |delta|: {tulu['delta'].abs().mean():.2f}")

fp_higher = (tulu['delta'] > 0).sum()
fp_lower = (tulu['delta'] < 0).sum()
print(f"FP higher: {fp_higher}/5, FP lower: {fp_lower}/5")
print(f"Delta range: [{tulu['delta'].min():.2f}, {tulu['delta'].max():.2f}]")

# Paper says: "Mean first-party advantage is +0.032 pp (range [-0.19, +0.23]), with 3/5 higher"
# Check: which source is first-party?
print(f"\nsource_a values: {tulu['source_a'].unique()}")
print(f"source_b values: {tulu['source_b'].unique()}")
