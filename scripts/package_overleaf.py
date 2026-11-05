"""Package the built marked/clean paper and standalone response for Overleaf."""

from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
PAPER = MANUSCRIPT / "Revision"
RESPONSE = MANUSCRIPT / "Response_Letter"


def paper_files():
    """Use TeX's recorded dependencies rather than copying unrelated study files."""
    allowed = {
        ".tex",
        ".bib",
        ".cls",
        ".sty",
        ".bst",
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".eps",
        ".dat",
    }
    files = {}
    for document in ["Revision-1", "Revision-1-clean"]:
        recorder = PAPER / f"{document}.fls"
        if not recorder.is_file():
            raise FileNotFoundError("Run make manuscript before packaging.")
        for line in recorder.read_text().splitlines():
            if not line.startswith("INPUT "):
                continue
            path = (PAPER / line[6:].strip('"')).resolve()
            if (
                not path.is_file()
                or not path.is_relative_to(MANUSCRIPT)
                or path.suffix.lower() not in allowed
            ):
                continue
            relative = (
                path.relative_to(PAPER)
                if path.is_relative_to(PAPER)
                else path.relative_to(MANUSCRIPT)
            )
            if relative.name not in {"main.tex", "main.pdf"}:
                files[relative.as_posix()] = path
    for name in [
        "Revision-1.tex",
        "Revision-1-clean.tex",
        "bibliography.bib",
        "logo.eps",
        "logo-eps-converted-to.pdf",
    ]:
        path = PAPER / name
        if path.is_file():
            files[name] = path
    return files


def copy_project(files, destination):
    for name, source in files.items():
        target = destination / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".tex":
            text = (
                source.read_text()
                .replace("../revision-markup.tex", "revision-markup.tex")
                .replace("../figure-style.tex", "figure-style.tex")
                .replace("../revision-evidence/", "revision-evidence/")
            )
            target.write_text(text)
        else:
            target.write_bytes(source.read_bytes())


def archive(project, target):
    with ZipFile(target, "w", ZIP_DEFLATED) as output:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(project).as_posix())
    with ZipFile(target) as result:
        assert result.testzip() is None
        assert all(
            not name.startswith("/") and ".." not in Path(name).parts
            for name in result.namelist()
        )


def main():
    with tempfile.TemporaryDirectory(prefix="iivv-overleaf-") as work:
        paper = Path(work) / "paper"
        response = Path(work) / "response"
        paper.mkdir()
        response.mkdir()
        copy_project(paper_files(), paper)
        (paper / "README.md").write_text(
            "Upload this ZIP as a new Overleaf project. Select pdfLaTeX.\nSet Revision-1.tex as the main document for highlighted changes, or\nRevision-1-clean.tex for the clean paper. Both use the same source.\nThe package was tested locally with TeX Live 2025. Sophie placeholders\nremain open and require author confirmation before submission.\n"
        )
        archive(paper, ROOT / "overleaf.zip")
        files = {
            "resposeletter.tex": RESPONSE / "resposeletter.tex",
            "revision-markup.tex": RESPONSE / "revision-markup.tex",
            "bibliography.bib": RESPONSE / "bibliography.bib",
            "README.md": RESPONSE / "README.md",
            "paper-references/Revision-1.aux": PAPER / "Revision-1.aux",
        }
        source = (RESPONSE / "resposeletter.tex").read_text()
        for image in sorted((RESPONSE / "quality-evidence").glob("*.png")):
            name = image.relative_to(RESPONSE).as_posix()
            if "{" + name + "}" in source:
                files[name] = image
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run make manuscript first.")
        copy_project(files, response)
        path = response / "resposeletter.tex"
        text = (
            path.read_text()
            .replace("{../Paper/Revision-1}", "{Paper/Revision-1}")
            .replace("{../Paper/bibliography}", "{bibliography}")
        )
        path.write_text(text)
        archive(response, ROOT / "response-letter.zip")
    print("Created overleaf.zip and response-letter.zip.")


if __name__ == "__main__":
    main()
