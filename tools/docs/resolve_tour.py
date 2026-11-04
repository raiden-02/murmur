"""Turns docs/code_tour.template.md into docs/CODE_TOUR.md: every
({path|text}) link becomes (../src/path#Lnnn), the line where `text`
appears. A \\n in the text spans lines. Fails loudly when the text is
missing or appears more than once, so no link can point at a stale line.

Run from anywhere: python tools/docs/resolve_tour.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "docs", "code_tour.template.md")
OUT = os.path.join(ROOT, "docs", "CODE_TOUR.md")
LINK = re.compile(r"\(\{([^|}]+)\|((?:[^}]|\}(?!\)))+)\}\)")


def main():
    template = open(TEMPLATE, encoding="utf-8").read()
    errors = []
    cache = {}

    def resolve(match):
        path, needle = match.group(1), match.group(2).replace("\\n", "\n").replace("\\t", "\t")
        if path not in cache:
            cache[path] = open(os.path.join(ROOT, "src", path), encoding="utf-8").read().replace("\r\n", "\n")
        source = cache[path]
        count = source.count(needle)
        if count != 1:
            errors.append(f"{path}: {needle!r} found {count} times")
            return match.group(0)
        line = source[: source.index(needle)].count("\n") + 1
        return f"(../src/{path}#L{line})"

    header = "<!-- Generated from docs/code_tour.template.md by tools/docs/resolve_tour.py: edit the template. -->\n"
    out = header + LINK.sub(resolve, template)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    open(OUT, "w", encoding="utf-8", newline="\n").write(out)
    print("tour written,", len(re.findall(r"#L\d+\)", out)), "line links")


if __name__ == "__main__":
    main()
