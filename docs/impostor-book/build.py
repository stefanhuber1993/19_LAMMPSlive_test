"""Assemble the impostor-renderer book into a single .epub.

No third-party dependencies: an EPUB is a zip with a prescribed layout, and the
stdlib has everything needed to write one. Produces EPUB 3 with a legacy NCX
table of contents alongside the nav document, because e-ink readers vary in
which one they honour.

    ./venv/bin/python docs/impostor-book/figures.py    # PNGs first
    ./venv/bin/python docs/impostor-book/build.py
"""
import os
import zipfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(os.path.dirname(HERE), "impostor-renderer.epub")

TITLE = "Spheres Without Geometry"
SUBTITLE = "How the LAMMPS-live impostor renderer draws 1500 beads in one draw call"
AUTHOR = "LAMMPS-live"
UID = "urn:uuid:8c8a1f42-6b1e-4d7a-9f3a-lammpslive-gl3d"
LANG = "en"

# (file, title in the table of contents, include in the TOC?)
CHAPTERS = [
    ("title.xhtml", "Title page and how to read this", True),
    ("ch01.xhtml", "1. The problem with spheres", True),
    ("ch02.xhtml", "2. Where a bead lives", True),
    ("ch03.xhtml", "3. The billboard that is exactly right", True),
    ("ch04.xhtml", "4. Ray-casting the sphere", True),
    ("ch05.xhtml", "5. The G-buffer: shade it later", True),
    ("ch06.xhtml", "6. The screen-space trick", True),
    ("ch07.xhtml", "7. Occlusion: ambient and the sun", True),
    ("ch08.xhtml", "8. The blur that must not leak", True),
    ("ch09.xhtml", "9. The composite", True),
    ("ch10.xhtml", "10. Depth of field", True),
    ("ch11.xhtml", "11. Antialiasing after the fact", True),
    ("ch12.xhtml", "12. Lines, and the depth buffer that outlives its pass", True),
    ("ch13.xhtml", "13. Drawing an infinite membrane", True),
    ("ch14.xhtml", "14. The cost model", True),
    ("ch15.xhtml", "15. Things to go and break", True),
]

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="EPUB/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def images():
    d = os.path.join(SRC, "images")
    return sorted(f for f in os.listdir(d) if f.endswith(".png"))


def content_opf():
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = [
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '    <item id="css" href="style.css" media-type="text/css"/>',
    ]
    spine = []
    for i, (fn, _, _) in enumerate(CHAPTERS):
        ident = "c%02d" % i
        items.append('    <item id="%s" href="%s" media-type="application/xhtml+xml"/>'
                     % (ident, fn))
        spine.append('    <itemref idref="%s"/>' % ident)
    for i, img in enumerate(images()):
        items.append('    <item id="img%02d" href="images/%s" media-type="image/png"/>'
                     % (i, img))
    return """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id"
         xml:lang="{lang}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">{uid}</dc:identifier>
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:language>{lang}</dc:language>
    <dc:description>{subtitle}</dc:description>
    <meta property="dcterms:modified">{modified}</meta>
  </metadata>
  <manifest>
{items}
  </manifest>
  <spine toc="ncx">
{spine}
  </spine>
</package>
""".format(lang=LANG, uid=UID, title=TITLE, author=AUTHOR, subtitle=SUBTITLE,
           modified=modified, items="\n".join(items), spine="\n".join(spine))


def nav_xhtml():
    lis = "\n".join(
        '      <li><a href="%s">%s</a></li>' % (fn, title)
        for fn, title, show in CHAPTERS if show)
    return """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}">
<head>
  <meta charset="utf-8"/>
  <title>Contents</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Contents</h1>
    <ol>
{lis}
    </ol>
  </nav>
</body>
</html>
""".format(lang=LANG, lis=lis)


def toc_ncx():
    points = []
    n = 0
    for fn, title, show in CHAPTERS:
        if not show:
            continue
        n += 1
        points.append(
            '    <navPoint id="np%d" playOrder="%d">\n'
            '      <navLabel><text>%s</text></navLabel>\n'
            '      <content src="%s"/>\n'
            '    </navPoint>' % (n, n, title, fn))
    return """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{uid}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{title}</text></docTitle>
  <navMap>
{points}
  </navMap>
</ncx>
""".format(uid=UID, title=TITLE, points="\n".join(points))


def build():
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w") as z:
        # The mimetype entry must be first and STORED, per the OCF spec.
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER, zipfile.ZIP_DEFLATED)
        z.writestr("EPUB/content.opf", content_opf(), zipfile.ZIP_DEFLATED)
        z.writestr("EPUB/nav.xhtml", nav_xhtml(), zipfile.ZIP_DEFLATED)
        z.writestr("EPUB/toc.ncx", toc_ncx(), zipfile.ZIP_DEFLATED)
        z.write(os.path.join(SRC, "style.css"), "EPUB/style.css",
                zipfile.ZIP_DEFLATED)
        for fn, _, _ in CHAPTERS:
            z.write(os.path.join(SRC, fn), "EPUB/" + fn, zipfile.ZIP_DEFLATED)
        for img in images():
            z.write(os.path.join(SRC, "images", img), "EPUB/images/" + img,
                    zipfile.ZIP_DEFLATED)
    print("wrote %s (%.1f kB)" % (OUT, os.path.getsize(OUT) / 1024.0))


if __name__ == "__main__":
    build()
