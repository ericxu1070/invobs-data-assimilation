"""Extract figures + clean text digests from the project notebooks.

Produces, under paper/:
  - figures/<shortname>__cellNN__outMM.png    : every embedded PNG output
  - _digests/<shortname>.md                    : markdown + code + text outputs,
                                                 with figures referenced inline
  - _manifest.json                             : one record per extracted figure
                                                 (notebook, cell idx, fig path,
                                                  producing code, preceding markdown)
"""
import base64
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figures")
DIG_DIR = os.path.join(HERE, "_digests")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DIG_DIR, exist_ok=True)

# notebook -> short slug used in filenames
NOTEBOOKS = {
    "PyTorch_InvObs_DA_v2_PaperFaithful.ipynb": "L96_PaperFaithful",
    "PyTorch_InvObs_DA_v2_Integrator.ipynb": "L96_Integrator",
    "PyTorch_InvObs_DA_v2_Kolmogorov.ipynb": "KFlow_Full4DVar",
    "SlidingWindow_PyTorch.ipynb": "L96_SlidingWindow",
    "SlidingWindow_Kolmogorov_PyTorch.ipynb": "KFlow_SlidingWindow",
    "PyTorch_InvObs_DA_WindowSweep_Corrected.ipynb": "L96_WindowSweep",
    "PyTorch_InvObs_DA.ipynb": "L96_Original_Port",
}

TEXT_TRUNC = 3000


def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def output_text(out):
    t = out.get("output_type")
    if t == "stream":
        s = out.get("text", "")
        return "".join(s) if isinstance(s, list) else s
    if t in ("execute_result", "display_data"):
        data = out.get("data", {})
        if "text/plain" in data:
            s = data["text/plain"]
            return "".join(s) if isinstance(s, list) else s
    if t == "error":
        return "ERROR: " + out.get("ename", "") + ": " + out.get("evalue", "")
    return ""


def main():
    manifest = []
    for nb_name, slug in NOTEBOOKS.items():
        path = os.path.join(ROOT, nb_name)
        if not os.path.exists(path):
            print("MISSING:", nb_name)
            continue
        nb = json.load(open(path, encoding="utf-8"))
        cells = nb["cells"]
        lines = [f"# DIGEST: {nb_name}  (slug: {slug})", ""]
        # rolling record of the most recent markdown text, for caption context
        prev_md = ""
        for ci, cell in enumerate(cells):
            ctype = cell.get("cell_type")
            code = src(cell)
            if ctype == "markdown":
                prev_md = code.strip()
                lines.append(f"## [md cell {ci}]")
                lines.append(code.rstrip())
                lines.append("")
            elif ctype == "code":
                lines.append(f"## [code cell {ci}]")
                lines.append("```python")
                lines.append(code.rstrip())
                lines.append("```")
                outs = cell.get("outputs", [])
                txt_parts = []
                img_idx = 0
                for out in outs:
                    # text
                    tx = output_text(out)
                    if tx and tx.strip():
                        txt_parts.append(tx)
                    # image
                    data = out.get("data", {})
                    if "image/png" in data:
                        b64 = data["image/png"]
                        if isinstance(b64, list):
                            b64 = "".join(b64)
                        fig_name = f"{slug}__cell{ci:02d}__out{img_idx:02d}.png"
                        fig_path = os.path.join(FIG_DIR, fig_name)
                        with open(fig_path, "wb") as fh:
                            fh.write(base64.b64decode(b64))
                        lines.append(f"\n>>> FIGURE EMBEDDED: figures/{fig_name}\n")
                        manifest.append({
                            "notebook": nb_name,
                            "slug": slug,
                            "cell_index": ci,
                            "fig_name": fig_name,
                            "fig_relpath": f"figures/{fig_name}",
                            "producing_code": code.strip()[:1500],
                            "preceding_markdown": prev_md[:1500],
                        })
                        img_idx += 1
                if txt_parts:
                    joined = "".join(txt_parts)
                    if len(joined) > TEXT_TRUNC:
                        joined = joined[:TEXT_TRUNC] + "\n...[truncated]..."
                    lines.append("--- output ---")
                    lines.append(joined.rstrip())
                lines.append("")
        digest = "\n".join(lines)
        with open(os.path.join(DIG_DIR, f"{slug}.md"), "w", encoding="utf-8") as fh:
            fh.write(digest)
        n_imgs = sum(1 for m in manifest if m["slug"] == slug)
        print(f"{slug:24s} cells={len(cells):3d} figs={n_imgs:3d} digest_chars={len(digest)}")

    with open(os.path.join(HERE, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nTotal figures extracted: {len(manifest)}")
    print(f"Manifest: {os.path.join(HERE, '_manifest.json')}")


if __name__ == "__main__":
    main()
