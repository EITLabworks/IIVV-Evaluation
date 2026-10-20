"""Localized marked replacements with a reversible source ledger."""

from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]


def protect_deleted(text):
    def end(start):
        depth = 1
        i = start
        while depth:
            if text[i] in "{}" and text[i - 1] != "\\":
                depth += 1 if text[i] == "{" else -1
            i += 1
        return i

    parts = []
    last = 0
    for match in re.finditer(r"\\(num|SI|SIrange|ref|eqref|cite)\{", text):
        if match.start() < last:
            continue
        j = match.end() - 1
        for _ in range(
            {"num": 1, "SI": 2, "SIrange": 3, "ref": 1, "eqref": 1, "cite": 1}[match[1]]
        ):
            j = end(j + 1)
        parts.append(
            text[last : match.start()] + r"\mbox{" + text[match.start() : j] + "}"
        )
        last = j
    return "".join(parts) + text[last:]


class MarkedManuscript:
    def __init__(self):
        self.path = ROOT / "manuscript/Paper/Revision-1.tex"
        self.ledger = ROOT / "manuscript/revision-baseline.json"
        self.baseline = json.loads(self.ledger.read_text())
        self.text = self.path.read_text()
        self.changes = []
        assert (
            hashlib.sha256(self.path.with_name("main.tex").read_bytes()).hexdigest()
            == self.baseline["main_sha256"]
        )

    def replace(self, old, new):
        if old == new or self.text.count(old) != 1:
            raise ValueError(f"Ambiguous or empty manuscript replacement: {old[:100]}")
        marked = (
            r"\edits{"
            + protect_deleted(old)
            + r"}"
            + (r"\edit{" + new + "}" if new else "")
        )
        self.text = self.text.replace(old, marked)
        self.changes.append(dict(old=old, new=marked))

    def add_before(self, anchor, text):
        if self.text.count(anchor) != 1:
            raise ValueError(f"Ambiguous addition anchor {anchor[:80]}")
        marked = r"\edit{" + text + "}" + "\n\n" + anchor
        self.text = self.text.replace(anchor, marked)
        self.changes.append(dict(old=anchor, new=marked))

    def save(self):
        self.path.write_text(self.text)
        self.baseline["code_revision_transformations"].extend(self.changes)
        self.ledger.write_text(json.dumps(self.baseline, indent=2) + "\n")
