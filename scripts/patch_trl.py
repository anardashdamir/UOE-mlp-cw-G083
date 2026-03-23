"""Patch TRL to disable eager imports of optional dependencies (mergekit, llm_blender, vllm, weave)."""

import site
import os

sp = site.getsitepackages()[0]

patches = {
    os.path.join(sp, "trl", "mergekit_utils.py"): [
        ("if is_mergekit_available():", "if False:  # patched"),
    ],
    os.path.join(sp, "trl", "trainer", "judges.py"): [
        ("if is_llm_blender_available():", "if False:  # patched"),
    ],
    os.path.join(sp, "trl", "trainer", "callbacks.py"): [
        ("if is_weave_available():", "if False:  # patched"),
    ],
}

for filepath, replacements in patches.items():
    if not os.path.exists(filepath):
        print(f"  SKIP {filepath} (not found)")
        continue
    code = open(filepath).read()
    changed = False
    for old, new in replacements:
        if old in code and new not in code:
            code = code.replace(old, new)
            changed = True
    if changed:
        open(filepath, "w").write(code)
        print(f"  PATCHED {filepath}")
    else:
        print(f"  OK {filepath} (already patched or not needed)")

print("\nDone. TRL optional imports disabled.")
