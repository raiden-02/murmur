"""Small SVG drawing kit shared by the diagram generators (light/dark aware)."""
import os

# Ramp: (fill, stroke, title, subtitle) for light, then dark.
RAMPS = {
    "purple": (("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"), ("#3C3489", "#AFA9EC", "#EEEDFE", "#CECBF6")),
    "teal":   (("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"), ("#085041", "#5DCAA5", "#E1F5EE", "#9FE1CB")),
    "gray":   (("#F1EFE8", "#5F5E5A", "#444441", "#5F5E5A"), ("#444441", "#B4B2A9", "#F1EFE8", "#D3D1C7")),
    "amber":  (("#FAEEDA", "#854F0B", "#633806", "#854F0B"), ("#633806", "#EF9F27", "#FAEEDA", "#FAC775")),
    "coral":  (("#FAECE7", "#993C1D", "#712B13", "#993C1D"), ("#712B13", "#F0997B", "#FAECE7", "#F5C4B3")),
    "red":    (("#FCEBEB", "#A32D2D", "#791F1F", "#A32D2D"), ("#791F1F", "#F09595", "#FCEBEB", "#F7C1C1")),
    "green":  (("#EAF3DE", "#3B6D11", "#27500A", "#3B6D11"), ("#27500A", "#97C459", "#EAF3DE", "#C0DD97")),
}

# Saturated accents that read on both light and dark backgrounds.
ACCENT = {"purple": "#7F77DD", "teal": "#1D9E75", "amber": "#E0921C", "coral": "#D85A30", "red": "#D94848",
          "green": "#5E9E25", "blue": "#378ADD", "gray": "#8A887F"}

WIDTH = 680


def style():
    light, dark = [], []
    for name, (l, d) in RAMPS.items():
        light.append(f".b-{name}{{fill:{l[0]};stroke:{l[1]}}}.k-{name}{{fill:{l[2]}}}.s-{name}{{fill:{l[3]}}}")
        dark.append(f".b-{name}{{fill:{d[0]};stroke:{d[1]}}}.k-{name}{{fill:{d[2]}}}.s-{name}{{fill:{d[3]}}}")
    return (
        "<style>"
        "text{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}"
        ".th{font-size:14px;font-weight:600;fill:#2C2C2A}.ts{font-size:12px;fill:#5F5E5A}"
        ".tx{font-size:11px;fill:#888780}.mono{font-family:Consolas,'SF Mono',Menlo,monospace}"
        ".ln{stroke:#888780}.dash{stroke:#B4B2A9}.bg{fill:#FFFFFF}.grid{stroke:#E4E2DA}"
        + "".join(light)
        + "@media (prefers-color-scheme:dark){"
        ".th{fill:#F1EFE8}.ts{fill:#B4B2A9}.tx{fill:#8A887F}.ln{stroke:#B4B2A9}.dash{stroke:#5F5E5A}"
        ".bg{fill:#1F1E1C}.grid{stroke:#34332F}"
        + "".join(dark)
        + "}</style>"
    )


ARROW = ('<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
         'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
         'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def svg(out_dir, name, height, title, body):
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'role="img"><title>{esc(title)}</title>{style()}{ARROW}'
        f'<rect class="bg" x="0" y="0" width="{WIDTH}" height="{height}" rx="12"/>{body}</svg>\n'
    )
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(content)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="ts", anchor="start", extra=""):
    return (f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="central"{extra}>'
            f'{esc(s)}</text>')


def rect(x, y, w, h, ramp, rx=3, extra=""):
    return f'<rect class="b-{ramp}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" stroke-width="0.6"{extra}/>'


def label_in(x, y, w, h, ramp, s, cls="ts", mono=False):
    m = " mono" if mono else ""
    return (f'<text class="{cls} k-{ramp}{m}" x="{x + w / 2:.1f}" y="{y + h / 2:.1f}" text-anchor="middle" '
            f'dominant-baseline="central">{esc(s)}</text>')


def chip(x, y, w, h, ramp, s="", cls="ts", mono=False, rx=3):
    return rect(x, y, w, h, ramp, rx) + (label_in(x, y, w, h, ramp, s, cls, mono) if s != "" else "")


def fill_rect(x, y, w, h, color, opacity=1.0, rx=0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{color}" '
            f'fill-opacity="{opacity:.3f}"/>')


def line(x1, y1, x2, y2, color=None, width=1.0, dash=False, arrow_end=False):
    stroke = f'stroke="{color}"' if color else 'class="ln"'
    d = ' stroke-dasharray="4 3"' if dash else ""
    m = ' marker-end="url(#a)"' if arrow_end else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {stroke} stroke-width="{width}"{d}{m}/>'


def arrow(x1, y1, x2, y2, color=None, width=1.4):
    return line(x1, y1, x2, y2, color, width, arrow_end=True)


def path(d, color=None, width=1.4, dash=False, arrow_end=True, fill="none", opacity=1.0):
    stroke = f'stroke="{color}"' if color else 'class="ln"'
    da = ' stroke-dasharray="4 4"' if dash else ""
    m = ' marker-end="url(#a)"' if arrow_end else ""
    return f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" {stroke} stroke-width="{width}"{da}{m}/>'


def circle(cx, cy, r, color, opacity=1.0, stroke=None):
    s = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" fill-opacity="{opacity}"{s}/>'


def polygon(points, color, opacity=1.0):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{color}" fill-opacity="{opacity}"/>'
