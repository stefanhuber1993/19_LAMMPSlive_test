# Spheres Without Geometry

A short textbook on the deferred impostor renderer in `lammps_live/ui/gl3d.py`,
written for reading on an e-reader. Fifteen chapters, fourteen figures, every
worked number taken from the MesoMem sheet playground as it actually runs.

## Building

```
./venv/bin/python docs/impostor-book/figures.py    # regenerate the PNGs
./venv/bin/python docs/impostor-book/build.py      # zip the EPUB
```

Output: `docs/impostor-renderer.epub`. No third-party dependencies beyond
matplotlib (figures only); the EPUB is assembled with `zipfile` from the stdlib.

## Layout

```
figures.py        every figure, as matplotlib line art -> src/images/*.png
build.py          EPUB 3 packaging (manifest, nav, NCX, OCF zip)
src/*.xhtml       one file per chapter, hand-written XHTML
src/style.css     deliberately minimal: what survives an e-ink reader
src/images/       generated, not hand-edited
```

## Conventions

* **PNG, not SVG.** E-ink readers all render PNG; SVG support is patchy.
* **Black on white line art.** No colour, heavy strokes, large type.
* Chapters are XHTML, so they must stay **well-formed XML** — named entities
  other than the five XML built-ins are not available; use numeric ones
  (`&#8212;`, `&#160;`, `&#963;`).
* Figure captions live in the XHTML `<figcaption>`, not baked into the image, so
  they reflow and stay readable at any font size.

Checks worth re-running after an edit:

```
./venv/bin/python -c "
import glob; from xml.dom.minidom import parse
[parse(f) for f in glob.glob('docs/impostor-book/src/*.xhtml')]; print('well-formed')"
```

## Reference numbers

The book's worked examples use the sheet at `periodic_images=(0, 0, 0)` in an
820x900 sim viewport: focal 444.4 px, bead radius 0.5 sigma, depth span
11.911-28.077 sigma. Chapter 13 gives the tiled figures (19,966 instances,
3.5-77.7 sigma). If the scenario's camera or box changes, those numbers move and
the affected passages need re-deriving.
