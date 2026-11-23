"""Read Umzi model packages and report whether this build supports them."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

from ._loader import PathArg

__all__ = [
    "GFISRv2Model",
    "GateRV2Model",
    "GateRV3Model",
    "MoESRModel",
    "ModelInfo",
    "ModelSection",
    "SMoSRModel",
    "peek_model",
]

MAGIC = b"UMZINPKG"
VERSION = 1
MAX_SIZE = 1 << 30
MAX_WEIGHT_BYTES = 256 << 20
MAX_SECTIONS = 32
DIGEST_SIZE = 32
PREFIX = struct.Struct("<8sIIIIQ")
HEADER_SIZE = PREFIX.size + DIGEST_SIZE
SECTION = struct.Struct("<IIQQ")
ARCHITECTURE = struct.Struct("<8I")
TENSOR_SPEC = struct.Struct("<2i")
OPERATION = struct.Struct("<11i")

# Keep the tables below in step with native/runtime/registry.cpp and the
# family parsers under native/kernels/.
GATERV3_FAMILY = 1
GFISRV2_FAMILY = 2
GATERV2_FAMILY = 3
MOESR_FAMILY = 4
SMOSR_FAMILY = 5
FAMILIES = {
    GATERV3_FAMILY: "gaterv3",
    GFISRV2_FAMILY: "gfisrv2",
    GATERV2_FAMILY: "gaterv2",
    MOESR_FAMILY: "moesr",
    SMOSR_FAMILY: "smosr",
}
SCHEMAS = {"gaterv3": 1, "gfisrv2": 1, "gaterv2": 1, "moesr": 1, "smosr": 1}
SECTION_NAMES = {1: "architecture", 2: "schedule", 3: "weights", 4: "checkpoint"}
# GateRV2 schema 1 architecture: input, output, scale, alignment, base
# channels, levels, latent blocks, the attention epsilon as float32, four
# encoder depths and four decoder depths. Unused levels are zero.
GATERV2_ARCHITECTURE = struct.Struct("<7If8I")
# GateRV2 schema 1 parameter table entry: byte offset and float32 count.
GATERV2_PARAMETER = struct.Struct("<QQ")
GATERV2_MAX_LEVELS = 4
GATERV2_MAX_WEIGHT_BYTES = 512 << 20
MOESR_ARCHITECTURE = struct.Struct("<10I")
MOESR_CONFIG = (3, 3, 4, 2, 64, 9, 4, 3, 170, 96)
GFISRV2_ARCHITECTURE = struct.Struct("<7I")
GFISRV2_CONFIG = (3, 3, 4, 48, 24, 128, 30)
SMOSR_ARCHITECTURE = struct.Struct("<8I")
SMOSR_CONFIG = (3, 3, 2, 2, 48, 2, 3, 1)
# Input channels, output channels, kernel width after padding lanes for cuDNN.
SMOSR_FILTERS = (
    (8, 16, 1),
    (8, 48, 1),
    (8, 48, 3),
    (48, 48, 3),
    (48, 96, 3),
    *((48, 48, 3), (48, 48, 3), (48, 96, 3)) * 5,
    (48, 48, 3),
    (64, 16, 3),
)


class GateRV3Op(IntEnum):
    PACK = 0
    CONV = 1
    RMS = 2
    LOCAL = 3
    MEAN_PROJECT = 4
    LOCAL_SKIP = 5
    INCEPTION = 6
    GLOBAL_SKIP = 7
    SHUFFLE = 8
    CONCAT = 9
    SILU = 10
    SPAN = 11
    DEPTHWISE = 12
    ATTENTION = 13
    GATE = 14
    ADD = 15
    RGB = 16
    SHUFFLE_CONCAT = 17


OP_KINDS = {op.value: op.name.title().replace("_", "") for op in GateRV3Op}


@dataclass(frozen=True)
class ModelSection:
    """One entry of the package section table."""

    id: int
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class GateRV3Model:
    """Architecture declared by a GateRV3 payload (schema 1)."""

    input_channels: int
    output_channels: int
    base_channels: int
    scale: int
    alignment: int
    tensors: int
    operations: int
    weight_bytes: int
    checkpoint: str | None
    op_kinds: dict[str, int]


@dataclass(frozen=True)
class GateRV2Model:
    """Architecture declared by a GateRV2 payload (schema 1)."""

    input_channels: int
    output_channels: int
    base_channels: int
    scale: int
    alignment: int
    encoder_depths: tuple[int, ...]
    decoder_depths: tuple[int, ...]
    latent_blocks: int
    attention_eps: float
    parameters: int
    weight_bytes: int
    checkpoint: str | None


@dataclass(frozen=True)
class MoESRModel:
    """The supported direct-shuffle 4x MoESR topology (schema 1)."""

    input_channels: int
    output_channels: int
    scale: int
    alignment: int
    base_channels: int
    groups: int
    full_blocks: int
    reduced_blocks: int
    full_hidden: int
    reduced_hidden: int
    weight_bytes: int
    checkpoint: str


@dataclass(frozen=True)
class GFISRv2Model:
    """Direct-shuffle GFISRv2 payload (schema 1)."""

    input_channels: int
    output_channels: int
    scale: int
    base_channels: int
    blocks: int
    hidden_channels: int
    fourier_channels: int
    weight_bytes: int
    checkpoint: str


@dataclass(frozen=True)
class SMoSRModel:
    """Direct-shuffle 2x SMoSR payload (schema 1)."""

    input_channels: int
    output_channels: int
    scale: int
    reflection_padding: int
    base_channels: int
    first_blocks: int
    middle_blocks: int
    end_blocks: int
    weight_bytes: int
    checkpoint: str


def gfisrv2_parameter_sizes() -> list[int]:
    """FP32 tensor sizes in bytes in native execution order."""
    cin, cout, scale, dim, blocks, hidden, fourier = GFISRV2_CONFIG
    band = (dim - fourier) // 3
    packed = 2 * fourier

    def conv(inputs: int, outputs: int, taps: int = 9, groups: int = 1) -> list[int]:
        return [outputs * (inputs // groups) * taps * 4, outputs * 4]

    block = [
        dim * 4,
        dim * 4,
        *conv(dim, 2 * hidden),
        packed * 4,
        packed * 4,
        *conv(packed, packed, groups=packed),
        *conv(packed, packed, 1),
        fourier * 4,
        fourier * 4,
        *conv(band, band, groups=band),
        *conv(band, band, 11, band),
        *conv(band, band, 11, band),
        *conv(hidden, dim),
        dim * 4,
    ]
    return conv(cin, dim) + block * blocks + conv(dim, 2 * dim) + conv(2 * dim, dim) + conv(dim, cout * scale**2)


def moesr_parameter_sizes() -> list[int]:
    """Packed parameter sizes in bytes, in the order used by the native runner."""
    _, _, _, _, dim, groups, full, reduced, full_hidden, reduced_hidden = MOESR_CONFIG
    branch = dim // 8

    def convolution(cin: int, cout: int) -> list[int]:
        return [cin * cout * 9 * 2, cout * 4]

    def block(hidden: int) -> list[int]:
        depthwise = [size for taps in (9, 11, 11) for size in (branch * taps * 4, branch * 4)]
        return [dim * 4, dim * 4, *convolution(dim, 2 * hidden), *depthwise, *convolution(hidden, dim), dim * 4]

    sizes = convolution(3, dim)
    for _ in range(groups):
        sizes += block(full_hidden) * full
        sizes += convolution(dim, dim // 4)
        sizes += block(reduced_hidden) * reduced
        sizes += convolution(dim, 4 * dim)
    return sizes + convolution(dim, 48)


def _meta_counts(dim: int) -> list[int]:
    hidden, branch = 3 * dim // 2, dim // 8
    return [
        *(dim, dim, 2 * dim * dim, 2 * dim, 2 * dim * 2 * 9, 2 * dim, dim * dim, dim, dim),
        *(dim, dim, 2 * hidden * dim, 2 * hidden),
        *(branch * 9, branch, branch * 11, branch, branch * 11, branch),
        *(dim * hidden, dim, dim),
    ]


def gaterv2_parameter_counts(
    base: int, encoder: tuple[int, ...], decoder: tuple[int, ...], latent: int, channels: int = 3
) -> list[int]:
    """Return the float32 element count of each packed GateRV2 parameter in payload order.

    Keep the order in step with ``Network`` in native/kernels/gaterv2/runner.cpp.
    """
    counts = [base * channels * 9, base]
    for level, depth in enumerate(encoder):
        dim = base << level
        for _ in range(depth):
            counts += _meta_counts(dim)
        counts.append(dim // 2 * dim * 9)
    width = base << len(encoder)
    keys = width // 16
    for _ in range(latent):
        counts += [width, width, 3 * width * width, 3 * width, (2 * keys + width) * width, 2 * keys + width]
        counts += [width * 3 * width // 2, width]
    for index, depth in enumerate(decoder):
        dim = width >> index
        counts += [2 * dim * dim * 9, dim // 2 * dim, dim // 2]
        for _ in range(depth):
            counts += _meta_counts(dim // 2)
    counts += [channels * base * 9, channels]
    return counts


@dataclass(frozen=True)
class ModelInfo:
    """What a model package declares, and whether this runtime accepts it.

    ``payload`` holds the family-specific metadata. Packages with an unknown
    payload layout have ``payload = None``.
    """

    path: Path
    size: int
    version: int
    family: int
    family_name: str | None
    schema: int
    digest: str
    sections: tuple[ModelSection, ...]
    payload: GateRV3Model | GateRV2Model | MoESRModel | GFISRv2Model | SMoSRModel | None = None
    problems: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """Whether the bundled runtime can open this package."""
        return not self.problems


def _read_section(file: BinaryIO, offset: int, length: int) -> bytes:
    file.seek(offset)
    return file.read(length)


def peek_model(path: PathArg) -> ModelInfo:
    """Read a ``.ummodel`` package and report the model information it declares.

    The container, section table, checksum, kernel family, payload schema and
    declared architecture are checked against the rules of the bundled runtime,
    so ``ModelInfo.supported`` says whether a session can open the package.
    Shape-dependent operation validation still runs when a session opens it.

    A file that is not a well-formed package raises ``ValueError``. A package
    from an unknown family, with an unsupported payload schema, or with an
    unsupported architecture is returned with the reasons in ``problems``.
    """
    model = Path(path).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Umzi model package not found: {model}")
    with model.open("rb") as file:
        size = file.seek(0, 2)
        if not HEADER_SIZE <= size <= MAX_SIZE:
            raise ValueError("Invalid Umzi model package size")
        file.seek(0)
        prefix = file.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError("Invalid Umzi model package header")
        magic, version, family, schema, count, total = PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError("Invalid Umzi model package magic")
        if version != VERSION:
            raise ValueError("Unsupported Umzi model package version")
        if not family or not schema or not 0 < count <= MAX_SECTIONS or total != size:
            raise ValueError("Invalid Umzi model package header")
        checksum = file.read(DIGEST_SIZE)
        table = file.read(count * SECTION.size)
        if len(checksum) != DIGEST_SIZE or len(table) != count * SECTION.size:
            raise ValueError("Invalid Umzi model package header")
        hasher = hashlib.sha256(prefix + table)
        while chunk := file.read(1 << 20):
            hasher.update(chunk)
        if hasher.digest() != checksum:
            raise ValueError("Umzi model package checksum mismatch")

        entries = []
        position = HEADER_SIZE + count * SECTION.size
        previous = 0
        for kind, reserved, offset, length in SECTION.iter_unpack(table):
            if kind <= previous or reserved or offset != position or not 0 < length <= size - position:
                raise ValueError("Invalid Umzi model section table")
            entries.append((kind, offset, length))
            previous = kind
            position += length
        if position != size:
            raise ValueError("Unexpected trailing Umzi model data")
        # The weights can be hundreds of MiB, so only the small sections are read.
        contents = {kind: _read_section(file, offset, length) for kind, offset, length in entries if kind in (1, 2, 4)}
        weights = next((length for kind, _, length in entries if kind == 3), 0)

    sections = tuple(
        ModelSection(kind, SECTION_NAMES.get(kind, f"section {kind}"), offset, length)
        for kind, offset, length in entries
    )
    family_name = FAMILIES.get(family)
    problems: list[str] = []
    payload = None
    if family_name is None:
        problems.append(f"Kernel family {family} is not supported by this build")
    elif family_name == "gaterv3":
        payload = _read_gaterv3(schema, contents.get(1), contents.get(2), weights, contents.get(4), problems)
    elif family_name == "gaterv2":
        payload = _read_gaterv2(schema, contents.get(1), contents.get(2), weights, contents.get(4), problems)
    elif family_name == "moesr":
        payload = _read_moesr(schema, contents.get(1), contents.get(2), weights, contents.get(4), problems)
    elif family_name == "gfisrv2":
        payload = _read_gfisrv2(schema, contents.get(1), contents.get(2), weights, contents.get(4), problems)
    elif family_name == "smosr":
        payload = _read_smosr(schema, contents.get(1), contents.get(2), weights, contents.get(4), problems)
    return ModelInfo(
        path=model,
        size=size,
        version=version,
        family=family,
        family_name=family_name,
        schema=schema,
        digest=checksum.hex(),
        sections=sections,
        payload=payload,
        problems=tuple(problems),
    )


def _read_gaterv3(
    schema: int,
    architecture: bytes | None,
    schedule: bytes | None,
    weight_bytes: int,
    checkpoint: bytes | None,
    problems: list[str],
) -> GateRV3Model | None:
    """Mirror the checks in ``ModelFile::parse`` and the structural part of ``ModelFile::validate``."""
    if schema != SCHEMAS["gaterv3"]:
        problems.append("Unsupported GateRV3 model payload version")
        return None
    if (
        architecture is None
        or len(architecture) != ARCHITECTURE.size
        or (checkpoint is not None and len(checkpoint) != DIGEST_SIZE)
    ):
        problems.append("Invalid GateRV3 model metadata size")
        return None
    (
        input_channels,
        scale,
        height,
        width,
        base_channels,
        output_channels,
        tensors,
        operations,
    ) = ARCHITECTURE.unpack(architecture)
    if (
        (input_channels, output_channels, scale, base_channels) != (3, 3, 1, 32)
        or height not in (8, 16)
        or height != width
    ):
        problems.append("Unsupported GateRV3 model architecture or padding alignment")
    op_kinds: dict[str, int] = {}
    schedule_size = tensors * TENSOR_SPEC.size + operations * OPERATION.size
    if (
        not 2 <= operations <= 2000
        or tensors != operations + 1
        or schedule is None
        or len(schedule) != schedule_size
        or not 0 < weight_bytes <= MAX_WEIGHT_BYTES
    ):
        problems.append("Invalid GateRV3 execution data or weights")
    else:
        split = tensors * TENSOR_SPEC.size
        specs = list(TENSOR_SPEC.iter_unpack(schedule[:split]))
        ops = list(OPERATION.iter_unpack(schedule[split:]))
        deepest = 3 if height == 8 else 4
        if any(not -1 <= level <= deepest or not 1 <= channels <= 2560 for level, channels in specs):
            problems.append("Invalid GateRV3 tensor description")
        elif max(level for level, _ in specs) != deepest:
            problems.append("GateRV3 padding alignment does not match the schedule depth")
        if (specs[0], specs[-1]) != ((0, 3), (0, 3)):
            problems.append("Invalid GateRV3 RGB shape")
        for index, op in enumerate(ops):
            kind, out = op[0], op[1]
            inputs, weights = op[2:6], op[6:9]
            ended = False
            invalid = out != index + 1 or kind not in OP_KINDS or inputs[0] < 0
            invalid |= (kind == GateRV3Op.PACK) != (index == 0) or (kind == GateRV3Op.RGB) != (index == len(ops) - 1)
            for item in inputs:
                invalid |= item < -1 or item > index or (ended and item != -1)
                ended |= item < 0
            for weight in weights:
                invalid |= weight >= 0 and (weight % 256 or weight >= weight_bytes)
            if invalid:
                problems.append(f"Invalid native model operation {index}")
                break
        for op in ops:
            name = OP_KINDS.get(op[0], f"kind {op[0]}")
            op_kinds[name] = op_kinds.get(name, 0) + 1
    return GateRV3Model(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        scale=scale,
        alignment=height,
        tensors=tensors,
        operations=operations,
        weight_bytes=weight_bytes,
        checkpoint=checkpoint.hex() if checkpoint else None,
        op_kinds=op_kinds,
    )


def _read_gaterv2(
    schema: int,
    architecture: bytes | None,
    table: bytes | None,
    weight_bytes: int,
    checkpoint: bytes | None,
    problems: list[str],
) -> GateRV2Model | None:
    """Mirror the payload checks in ``GateRV2Session::open``."""
    if schema != SCHEMAS["gaterv2"]:
        problems.append("Unsupported GateRV2 model payload version")
        return None
    if (
        architecture is None
        or len(architecture) != GATERV2_ARCHITECTURE.size
        or (checkpoint is not None and len(checkpoint) != DIGEST_SIZE)
    ):
        problems.append("Invalid GateRV2 model metadata size")
        return None
    values = GATERV2_ARCHITECTURE.unpack(architecture)
    input_channels, output_channels, scale, alignment, base, levels, latent, eps = values[:8]
    depths = values[8:]
    encoder_all, decoder_all = depths[:GATERV2_MAX_LEVELS], depths[GATERV2_MAX_LEVELS:]
    if not (
        (input_channels, output_channels, scale) == (3, 3, 1)
        and 1 <= levels <= GATERV2_MAX_LEVELS
        and alignment == 1 << levels
        and 16 <= base <= 64
        and base % 16 == 0
        and 1 <= latent <= 64
        and 0 < eps < 1
        and all(depth <= 32 for depth in depths)
        and not any(encoder_all[levels:])
        and not any(decoder_all[levels:])
    ):
        problems.append("Unsupported GateRV2 model architecture")
        return None
    encoder, decoder = tuple(encoder_all[:levels]), tuple(decoder_all[:levels])
    counts = gaterv2_parameter_counts(base, encoder, decoder, latent)
    if (
        table is None
        or len(table) != len(counts) * GATERV2_PARAMETER.size
        or not 0 < weight_bytes <= GATERV2_MAX_WEIGHT_BYTES
    ):
        problems.append("Invalid GateRV2 parameter table or weights")
    else:
        end = 0
        entries = GATERV2_PARAMETER.iter_unpack(table)
        for index, ((offset, stored), count) in enumerate(zip(entries, counts, strict=True)):
            if (
                stored != count
                or offset % 256
                or offset < end
                or offset > weight_bytes
                or count > (weight_bytes - offset) // 4
            ):
                problems.append(f"Invalid GateRV2 parameter {index}")
                break
            end = offset + count * 4
        else:
            if end != weight_bytes:
                problems.append("Unexpected trailing GateRV2 weight data")
    return GateRV2Model(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base,
        scale=scale,
        alignment=alignment,
        encoder_depths=encoder,
        decoder_depths=decoder,
        latent_blocks=latent,
        attention_eps=eps,
        parameters=len(counts),
        weight_bytes=weight_bytes,
        checkpoint=checkpoint.hex() if checkpoint else None,
    )


def _read_moesr(
    schema: int,
    architecture: bytes | None,
    table: bytes | None,
    weight_bytes: int,
    checkpoint: bytes | None,
    problems: list[str],
) -> MoESRModel | None:
    if schema != SCHEMAS["moesr"]:
        problems.append("Unsupported MoESR model payload version")
    elif architecture != MOESR_ARCHITECTURE.pack(*MOESR_CONFIG) or checkpoint is None or len(checkpoint) != DIGEST_SIZE:
        problems.append("Unsupported MoESR model architecture or checkpoint metadata")
    else:
        expected = 0
        for size in moesr_parameter_sizes():
            expected = (expected + 255) // 256 * 256 + size
        if table is None and expected == weight_bytes:
            return MoESRModel(*MOESR_CONFIG, weight_bytes, checkpoint.hex())
        problems.append("Invalid MoESR weight layout")
    return None


def _read_smosr(
    schema: int,
    architecture: bytes | None,
    table: bytes | None,
    weight_bytes: int,
    checkpoint: bytes | None,
    problems: list[str],
) -> SMoSRModel | None:
    if schema != SCHEMAS["smosr"]:
        problems.append("Unsupported SMoSR model payload version")
    elif architecture != SMOSR_ARCHITECTURE.pack(*SMOSR_CONFIG) or checkpoint is None or len(checkpoint) != DIGEST_SIZE:
        problems.append("Unsupported SMoSR model architecture or checkpoint metadata")
    else:
        expected = 0
        for cin, cout, k in SMOSR_FILTERS:
            for size in (cin * cout * k * k * 2, cout * 4):
                expected = (expected + 255) // 256 * 256 + size
        if table is None and expected == weight_bytes:
            return SMoSRModel(*SMOSR_CONFIG, weight_bytes, checkpoint.hex())
        problems.append("Invalid SMoSR weight layout")
    return None


def _read_gfisrv2(
    schema: int,
    architecture: bytes | None,
    table: bytes | None,
    weight_bytes: int,
    checkpoint: bytes | None,
    problems: list[str],
) -> GFISRv2Model | None:
    if schema != SCHEMAS["gfisrv2"]:
        problems.append("Unsupported GFISRv2 model payload version")
    elif (
        architecture != GFISRV2_ARCHITECTURE.pack(*GFISRV2_CONFIG)
        or checkpoint is None
        or len(checkpoint) != DIGEST_SIZE
    ):
        problems.append("Unsupported GFISRv2 model architecture or checkpoint metadata")
    else:
        expected = 0
        for size in gfisrv2_parameter_sizes():
            expected = (expected + 255) // 256 * 256 + size
        if table is None and expected == weight_bytes:
            return GFISRv2Model(*GFISRV2_CONFIG, weight_bytes, checkpoint.hex())
        problems.append("Invalid GFISRv2 weight layout")
    return None
