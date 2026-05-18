"""Compatibility entry point for the podcast transcription CLI.

The implementation lives in ``src/podcast_transcribe``. Keeping this thin
wrapper preserves older commands such as ``python podcast_transcribe_host.py``
while allowing the project to use a conventional package layout.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from podcast_transcribe.cli import main


if __name__ == "__main__":
    main()
