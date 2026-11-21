"""Write the shared Umzi model container."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .._model import DIGEST_SIZE, HEADER_SIZE, MAGIC, MAX_SECTIONS, PREFIX, SECTION, VERSION


def write_package(path: Path, family: int, schema: int, sections: dict[int, bytes | Path]) -> None:
    if family < 1 or schema < 1 or not 0 < len(sections) <= MAX_SECTIONS or min(sections) < 1:
        raise ValueError("Invalid Umzi model family, schema, or section IDs")
    offset = HEADER_SIZE + len(sections) * SECTION.size
    table = []
    for kind, content in sorted(sections.items()):
        size = content.stat().st_size if isinstance(content, Path) else len(content)
        if not size:
            raise ValueError(f"Empty model section {kind}")
        table.append(SECTION.pack(kind, 0, offset, size))
        offset += size
    prefix = PREFIX.pack(MAGIC, VERSION, family, schema, len(sections), offset)
    digest = hashlib.sha256(prefix)
    with path.open("wb") as file:
        file.write(prefix)
        file.write(bytes(DIGEST_SIZE))
        for entry in table:
            file.write(entry)
            digest.update(entry)
        for _, content in sorted(sections.items()):
            if isinstance(content, Path):
                with content.open("rb") as source:
                    while chunk := source.read(1 << 20):
                        file.write(chunk)
                        digest.update(chunk)
            else:
                file.write(content)
                digest.update(content)
        if file.tell() != offset:
            raise ValueError("Model sections changed while exporting")
        file.seek(PREFIX.size)
        file.write(digest.digest())
