"""Quick check: do upgrade04 and upgrade08 have different upgrade.hvac_heating_efficiency values?"""
import pandas as pd

base = 'cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv'

# Read just the column we need, with chunked reading to avoid OOM
col = 'upgrade.hvac_heating_efficiency'

def get_value_counts(path):
    chunks = pd.read_csv(path, low_memory=True, usecols=[col], chunksize=20000)
    counts = pd.Series(dtype=int)
    for chunk in chunks:
        c = chunk[col].value_counts(dropna=False)
        counts = counts.add(c, fill_value=0)
    return counts.astype(int).sort_index()

print("Reading upgrade04...")
vc04 = get_value_counts(f'{base}/upgrade04_metadata_and_annual_results.csv')
print("upgrade04 value_counts:")
print(vc04)
print()

print("Reading upgrade08...")
vc08 = get_value_counts(f'{base}/upgrade08_metadata_and_annual_results.csv')
print("upgrade08 value_counts:")
print(vc08)
print()

# Compare
if vc04.equals(vc08):
    print("RESULT: IDENTICAL value distributions")
else:
    print("RESULT: DIFFERENT value distributions")
    all_vals = sorted(set(vc04.index) | set(vc08.index))
    print(f"{'Value':<60} {'upgrade04':>10} {'upgrade08':>10}")
    print("-" * 82)
    for v in all_vals:
        c04 = vc04.get(v, 0)
        c08 = vc08.get(v, 0)
        marker = " <---" if c04 != c08 else ""
        print(f"{str(v):<60} {c04:>10} {c08:>10}{marker}")
