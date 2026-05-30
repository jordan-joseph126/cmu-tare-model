import json
path = r'c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\calculate_postTARE_am_kpis_demand_bill_savings.ipynb'
nb = json.load(open(path, encoding='utf-8'))
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for c in cells[-4:]:
    src = c['source']
    src_str = src if isinstance(src, str) else ''.join(src)
    cell_id = c.get('id', 'no-id')
    print(f'  id={cell_id} type={c["cell_type"]} start={src_str[:100]!r}')
