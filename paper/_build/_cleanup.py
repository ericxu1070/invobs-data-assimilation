"""Tidy the paper/ folder: keep only used figures; move reproducibility helpers to _build/."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BUILD = os.path.join(HERE, "_build")
UNUSED = os.path.join(BUILD, "extracted_unused")
os.makedirs(UNUSED, exist_ok=True)

USED = {
    "noise_distributions.png",
    "L96_Integrator__cell08__out00.png",
    "L96_Integrator__cell10__out00.png",
    "L96_PaperFaithful__cell10__out00.png",
    "L96_PaperFaithful__cell19__out00.png",
    "L96_PaperFaithful__cell20__out00.png",
    "L96_PaperFaithful__cell28__out00.png",
    "L96_SlidingWindow__cell27__out00.png",
    "L96_SlidingWindow__cell30__out00.png",
    "L96_SlidingWindow__cell31__out00.png",
    "L96_SlidingWindow__cell34__out00.png",
    "L96_SlidingWindow__cell36__out00.png",
    "KFlow_Full4DVar__cell06__out00.png",
    "KFlow_SlidingWindow__cell19__out00.png",
    "KFlow_SlidingWindow__cell24__out00.png",
    "KFlow_SlidingWindow__cell27__out00.png",
    "KFlow_SlidingWindow__cell31__out00.png",
    "KFlow_SlidingWindow__cell33__out00.png",
}

moved_figs = 0
for name in os.listdir(FIG):
    if name.lower().endswith(".png") and name not in USED:
        shutil.move(os.path.join(FIG, name), os.path.join(UNUSED, name))
        moved_figs += 1

# Move reproducibility helpers into _build/ (keep main.tex, references.bib, _sections/, figures/).
for item in ["_digests", "_manifest.json", "_extract_notebooks.py",
             "_author_guide.md", "_preamble.tex", "_build.log", "_build2.log"]:
    src = os.path.join(HERE, item)
    if os.path.exists(src):
        dst = os.path.join(BUILD, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
        shutil.move(src, dst)

kept = sorted(n for n in os.listdir(FIG) if n.lower().endswith(".png"))
print(f"figures kept: {len(kept)}  moved to _build/extracted_unused: {moved_figs}")
assert set(kept) == USED, "mismatch between kept figures and USED set!"
print("All 18 used figures present; cleanup OK.")
