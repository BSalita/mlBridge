# Canonical Source — `src/mlBridge/`

This directory is the **single canonical source** for the `mlBridge` Python package
used by every active project in this workspace (ACBL pipeline, Elo Ratings app,
FFBridge app, BBO bidding tools, etc.).

## How sharing works

Other subprojects do not vendor copies of these files. The package ROOT (this
directory's parent, i.e. `src/`) is put on `sys.path`, either by the venv's
`site-packages/monorepo_src.pth` or by a small resolver at the top of each app
entry point. All imports are package-style:

```python
from mlBridge.mlBridgeAugmentLib import AllAugmentations
from mlBridge import mlBridgeFFLib
```

Never put THIS directory itself on `sys.path` and never use flat imports
(`import mlBridgeFFLib`). The package `__init__` is lazy (PEP 562), so package
imports are cheap; heavy dependencies (sklearn, torch, sqlalchemy) load only
when a symbol from their owning module is first touched.

Windows junctions (`mklink /J mlBridge ...`) are obsolete and should not be
recreated.

## "Duplicate" copies you may see in the workspace

You will find files with the same name under at least two other paths:

- `src/postmortem-ffbridge/src/mlbridgelib/mlBridge/`
- `src/_archive/github-tests/ffbridge-postmortem/mlBridge/`

Both of these directories live **inside other, independent git repositories**
(`src/postmortem-ffbridge/.git`, `src/_archive/github-tests/ffbridge-postmortem/.git`).
They are vendored snapshots owned by those external repos, not by this workspace.

**Do not edit them.** They are out-of-date and not imported by any active code
path here — `ffbridge_streamlit.py` (the live one) imports the `mlBridge`
package from the `src/` root on `sys.path`, which resolves to *this* directory.

If you need to refresh those vendored copies, do it inside their own git repos
as a separate exercise.

## Rule of thumb

If you are editing a file in `mlBridge/...`, the file you should be editing is
the one under `src/mlBridge/`. Nothing else.
