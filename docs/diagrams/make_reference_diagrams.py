"""Diagram for docs/correctness.md: every checkpoint's measured error against
the Hugging Face reference, in network order. Real data:
data/reference_errors.json, taken from the reference tests' report.

Refresh the data, then draw:
  out/build/x64-release/murmur-tests.exe --gtest_filter=*TransformerReferenceTest* --gtest_output=json:report.json
  python docs/diagrams/make_reference_diagrams.py report.json
Draw from the saved data only:
  python docs/diagrams/make_reference_diagrams.py
"""
import json
import math
import os
import sys

from svg_kit import ACCENT, circle, line, svg, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "reference_errors.json")
PROMPTS = {"0": "plain", "1": "chat"}   # the order of INSTANTIATE_TEST_SUITE_P in transformer_tests.cpp
MATHS = ("fp32", "f16", "bf16")


def from_test_report(path):
    """Checkpoint errors as the tests recorded them (ReferenceComparer::RecordErrors)."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    data = {"source": "murmur-tests, TransformerReferenceTest (Release, RTX 4080 SUPER); relative error = "
                      "largest absolute difference / largest reference magnitude", "prompts": {}}
    for suite in report["testsuites"]:
        for test in suite["testsuite"]:
            prompt = PROMPTS[test["name"].rsplit("/", 1)[1]]
            for key, value in test.items():
                math_name, _, checkpoint = key.partition(".")
                if math_name in MATHS and checkpoint:
                    data["prompts"].setdefault(prompt, {}).setdefault(math_name, []).append([checkpoint, float(value)])
    return data


def reference_depth(data):
    """Error per checkpoint from the embedding to the logits, log scale, for
    the chat prompt: exact FP32 math as a flat band far below its tolerance,
    tensor-core math (activations rounded to 16 bits) as a higher band below
    its own. Neither climbs with depth: rounding doesn't pile up."""
    runs = data["prompts"]["chat"]
    x0, x1, y0, y1 = 70, 560, 46, 256
    low, high = -8, 0
    count = len(runs["fp32"])

    def px(i):
        return x0 + i / (count - 1) * (x1 - x0)

    def py(v):
        return y1 - (math.log10(max(v, 10 ** low)) - low) / (high - low) * (y1 - y0)

    b = text(40, 22, "Every checkpoint of Mei's prompt against the reference, embedding to logits", "th")
    for p in range(low, high + 1, 2):
        b += line(x0, py(10 ** p), x1, py(10 ** p), "#8A887F33", 0.6) + text(x0 - 8, py(10 ** p), f"1e{p}", "tx", "end")
    for math_name, color, tolerance in (("bf16", ACCENT["amber"], 1e-1), ("fp32", ACCENT["teal"], 1e-3)):
        b += line(x0, py(tolerance), x1, py(tolerance), color, 1.2, dash=True)
        for i, (_, value) in enumerate(runs[math_name]):
            if value > 0:
                b += circle(px(i), py(value), 2.2, color, 0.85)
    b += text(x1 + 8, py(1e-1), "BF16 limit", "tx") + text(x1 + 8, py(1e-3), "FP32 limit", "tx")
    for i, label in ((0, "embedding"), (count // 2, "layer 14"), (count - 1, "logits")):
        b += text(px(i), y1 + 16, label, "tx", "middle")
    b += circle(x0 + 6, y1 + 38, 4, ACCENT["teal"]) + text(x0 + 16, y1 + 42, "FP32 math", "tx")
    b += circle(x0 + 120, y1 + 38, 4, ACCENT["amber"]) + text(x0 + 130, y1 + 42, "tensor cores (BF16 inputs)", "tx")
    svg(HERE, "reference_depth.svg", y1 + 58, "Error per checkpoint against the reference, embedding to logits", b)


def main():
    if len(sys.argv) > 1:
        with open(DATA, "w", encoding="utf-8") as f:
            json.dump(from_test_report(sys.argv[1]), f, indent=1)
    with open(DATA, encoding="utf-8") as f:
        reference_depth(json.load(f))


if __name__ == "__main__":
    main()
