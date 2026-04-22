"""One-shot fixup for remaining Phase 1 items in the preTARE script."""
import re

path = r'C:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py'
content = open(path, encoding='utf-8').read()

# -----------------------------------------------------------------------
# Fix 1: n_total = len(jenkins_ref) → len(JENKINS_BREAKEVEN_REF_90)
# -----------------------------------------------------------------------
assert 'n_total = len(jenkins_ref)' in content, "FAIL: n_total target not found"
content = content.replace(
    'n_total = len(jenkins_ref)',
    'n_total = len(JENKINS_BREAKEVEN_REF_90)',
)

# -----------------------------------------------------------------------
# Fix 2: ± in strict/relaxed match print lines (Task D block)
# -----------------------------------------------------------------------
content = content.replace(
    'print(f"\\nStrict match (\u00b10.05): {n_strict}/{n_total}")\n'
    'print(f"Relaxed match (\u00b10.50): {n_relaxed}/{n_total}")',
    'print(f"\\nStrict match (+/-0.05): {n_strict}/{n_total}")\n'
    'print(f"Relaxed match (+/-0.50): {n_relaxed}/{n_total}")',
)

# -----------------------------------------------------------------------
# Fix 3: ✓ / ✗ glyphs in the Task D row-print line
# -----------------------------------------------------------------------
content = content.replace(
    "f\"{ref_val:>8.2f} {'✓' if strict else '✗':>7} \"\n"
    "          f\"{'✓' if relaxed else '✗':>8} {'Yes' if cop_exceeds else 'No':>7}\")",
    "f\"{ref_val:>8.2f} {'[OK]' if strict else '[FAIL]':>7} \"\n"
    "          f\"{'[OK]' if relaxed else '[FAIL]':>8} {'Yes' if cop_exceeds else 'No':>7}\")",
)

# -----------------------------------------------------------------------
# Fix 4: Remaining ✓ / ⚠ / ✗ glyphs in lines we touched
# (only in sections already modified — don't mass-replace untouched code)
# -----------------------------------------------------------------------
content = content.replace(
    "print(f\"\\n✓ TASK D COMPLETE\")",
    "print(\"\\n[OK] TASK D COMPLETE\")",
)

open(path, 'w', encoding='utf-8').write(content)
print("[OK] Phase 1 fixup complete")
