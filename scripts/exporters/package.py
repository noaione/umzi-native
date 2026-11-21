"""Read and write the shared Umzi model container."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

MAGIC = b"UMZINPKG"
VERSION = 1
GATERV3 = 1
PREFIX = struct.Struct("<8sIIIIQ")
SECTION = struct.Struct("<IIQQ")
HEADER_SIZE = PREFIX.size + 32


def write_package(path: Path, family: int, schema: int, sections: dict[int, bytes | Path]) -> None:
    if family < 1 or schema < 1 or not 0 < len(sections) <= 32 or min(sections) < 1:
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
        file.write(bytes(32))
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


def read_sections(file, size: int, family: int, schema: int) -> dict[int, tuple[int, int]]:
    prefix = file.read(PREFIX.size)
    magic, version, found_family, found_schema, count, full_size = PREFIX.unpack(prefix)
    if (magic, version, found_family, found_schema) != (MAGIC, VERSION, family, schema):
        raise ValueError("Unsupported Umzi model package")
    if not 0 < count <= 32 or full_size != size:
        raise ValueError("Invalid Umzi model package size")
    checksum = file.read(32)
    digest = hashlib.file_digest(file, lambda: hashlib.sha256(prefix)).digest()
    if checksum != digest:
        raise ValueError("Umzi model package checksum mismatch")
    file.seek(HEADER_SIZE)
    position = HEADER_SIZE + count * SECTION.size
    sections = {}
    previous = 0
    for _ in range(count):
        kind, reserved, offset, length = SECTION.unpack(file.read(SECTION.size))
        if kind <= previous or reserved or offset != position or not 0 < length <= size - position:
            raise ValueError("Invalid Umzi model package section")
        sections[kind] = (offset, length)
        position += length
        previous = kind
    if position != size:
        raise ValueError("Unexpected trailing Umzi model data")
    return sections
