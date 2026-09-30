"""Project-relative paths; entry points work independently of the current directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA = ROOT / 'input_data'
DUMPS = ROOT / 'dumps'
FIGURES = ROOT / 'figures'


def resolve_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value).expanduser() if value is not None else default
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
