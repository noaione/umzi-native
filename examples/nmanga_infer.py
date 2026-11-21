# A modified version of nmanga/private toolkit denoise inference, adjusted for umzi native

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

import umzi_native as umzi

if TYPE_CHECKING:
    from onnxruntime import InferenceSession as OrtSession

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/1x_umzi_digital_decompress_gaterv3_1.ummodel"
ONNX_MODEL = ROOT / "output/baseline/gaterv3-fp16-t128-b16.onnx"
IMAGE = ROOT / "samples/input/page-001.jpg"
BATCH_SIZE = 16
TILE_SIZE = 128


class _RtxSession:
    context = 0
    scale_factor = 1

    def __init__(self, session: OrtSession) -> None:
        self.session = session

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def run(self, names, feeds):
        return self.session.run(names, feeds)


type ModelSession = umzi.InferenceSession | _RtxSession


def _run_tta(infer: "Callable[[np.ndarray], np.ndarray]", batch: "np.ndarray") -> "np.ndarray":
    """Average `infer` over the batch's 8 flips/rotations, each output turned back first.

    Test-time augmentation, as waifu2x runs it: 8 runs of the whole batch.
    """
    output = np.zeros(1, np.float32)
    for k in range(8):
        x = np.rot90(batch, k % 4, axes=(2, 3))
        y = infer(np.ascontiguousarray(x[..., ::-1] if k >= 4 else x))
        y = y[..., ::-1] if k >= 4 else y
        output = output + np.rot90(y, -(k % 4), axes=(2, 3))
    return output / 8


def _denoise_image(
    input_image: Image.Image,
    model: ModelSession,
    *,
    batch_size: int,
    tile_size: int,
    tile_overlap: int,
    context: int | None,
    contrast_stretch: bool,
    background: Literal["white", "black"],
    use_fp32: bool,
    tta: bool,
    pad_final_batch: bool = False,
    inference_times: list[float] | None = None,
) -> Image.Image:
    if context is None:
        context = getattr(model, "context", 0) or 0
    if not isinstance(context, int):
        raise ValueError("Model context must be an integer.")
    scale_factor = getattr(model, "scale_factor", None)
    if not isinstance(scale_factor, int):
        raise ValueError("Model scale factor is not set. Ensure the model was prepared correctly.")
    if scale_factor < 1:
        if context:
            # Shape-based scale detection sees a valid-convolution x1 model as shrinking.
            scale_factor = 1
        else:
            raise ValueError("Model scale factor must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if tile_size < 1:
        raise ValueError("Tile size must be at least 1.")
    if context < 0 or context * 2 >= tile_size:
        raise ValueError("Context must be non-negative and smaller than half the tile size.")
    core_tile_size = tile_size - context * 2
    if tile_overlap < 0 or tile_overlap >= core_tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than the output tile size.")
    if context and scale_factor != 1:
        raise ValueError("Context trimming is only supported for x1 models.")

    input_spec = model.get_inputs()[0]
    input_channel_count = input_spec.shape[1]
    is_grayscale = input_channel_count == 1
    model_dtype = np.float16

    orig_img_mode = input_image.mode
    has_alpha = "A" in input_image.getbands() or (
        input_image.mode == "P" and input_image.info.get("transparency") is not None
    )
    alpha = None
    if has_alpha:
        if "A" in input_image.getbands():
            alpha = input_image.getchannel("A")
        else:
            with input_image.convert("RGBA") as rgba:
                alpha = rgba.getchannel("A")

    img = input_image.convert("L" if is_grayscale else "RGB")
    image_width, image_height = img.size

    overlapping = tile_overlap > 0 or context > 0
    step = core_tile_size - tile_overlap
    if overlapping:
        tile_count_height = max(1, -(-(image_height - tile_overlap) // step))
        tile_count_width = max(1, -(-(image_width - tile_overlap) // step))
        padded_height = (tile_count_height - 1) * step + core_tile_size
        padded_width = (tile_count_width - 1) * step + core_tile_size

        image_array = np.asarray(img)
        if is_grayscale:
            image_array = image_array[..., None]
        # Context is trimmed from each output, so pad by that much to keep the first and last
        # output pixels tied to the real edge.
        image_array = np.pad(
            image_array,
            (
                (context, padded_height - image_height + context),
                (context, padded_width - image_width + context),
                (0, 0),
            ),
            mode="edge",
        )
    else:
        tile_count_height = int(np.ceil(image_height / tile_size))
        tile_count_width = int(np.ceil(image_width / tile_size))
        padded_height = tile_count_height * tile_size
        padded_width = tile_count_width * tile_size
        # A white background helps with contrast stretching (can lead to reversing CMYK shift).
        pad_value = 255 if background == "white" else 0
        canvas = Image.new("L" if is_grayscale else "RGB", (padded_width, padded_height), pad_value)
        canvas.paste(img, (0, 0))
        image_array = np.asarray(canvas)
        if is_grayscale:
            image_array = image_array[..., None]

    # Pre-Processing
    # contrast stretching: scaling the image based on its own darkest and brightest pixels.
    # For example, a very dark photo (e.g., pixel values from 10 to 50) and a very bright photo (e.g., values
    # from 200 to 250) will both be stretched to the full [0.0, 1.0] range.
    # The model loses all information about the image's absolute brightness. Can be desirable with a white
    # background to reverse CMYK shift in rare (color) images.
    # Not desirable in general so the alternative path should be taken in the vast majority of cases.
    # See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)#Contrast_Stretching_for_Image_Enhancement
    if not contrast_stretch and image_array.dtype == np.uint8:
        # Convert tiles as they are batched so a large page does not need a full float copy.
        lookup = (np.arange(256) / 255.0).astype(model_dtype)
        if step < tile_size:
            # Shared pixels would otherwise be converted once for every tile that contains them.
            image_array = lookup[image_array]
            lookup = None
    else:
        lookup = None
        # A page of one colour has nothing to stretch, and would divide by zero.
        span = np.ptp(image_array) if contrast_stretch else 0
        image_array = (image_array - np.min(image_array)) / span if span else image_array / 255.0
        image_array = image_array.astype(model_dtype)
    # Shape the padded page as C x H x W, then take a view of each tile.
    image_array = image_array.transpose(2, 0, 1)

    # Cut into tiles
    windows = np.lib.stride_tricks.sliding_window_view(image_array, (tile_size, tile_size), axis=(1, 2))[
        :, ::step, ::step
    ].transpose(1, 2, 0, 3, 4)
    num_chunks = tile_count_height * tile_count_width

    def tiles_of(start: int, stop: int) -> "np.ndarray":
        ys, xs = np.divmod(np.arange(start, stop), tile_count_width)
        batch = windows[ys, xs]
        if lookup is not None:
            return lookup[batch]
        return np.ascontiguousarray(batch, dtype=model_dtype)

    input_name = input_spec.name
    # An int on a static input (e.g. a TensorRT-RTX engine): short batches get padded up to it below.
    input_shape = getattr(input_spec, "shape", None)
    fixed_batch = input_shape[0] if input_shape and isinstance(input_shape[0], int) else None
    if fixed_batch is None and pad_final_batch:
        fixed_batch = batch_size
    if fixed_batch is not None and batch_size > fixed_batch:
        raise ValueError(f"Batch size {batch_size} exceeds the model's fixed batch {fixed_batch}.")
    output_name = model.get_outputs()[0].name

    def infer(batch: "np.ndarray") -> "np.ndarray":
        return np.asarray(model.run([output_name], {input_name: batch})[0])

    if scale_factor > 1:
        image_height = image_height * scale_factor
        image_width = image_width * scale_factor

    output_tile_size = core_tile_size * scale_factor
    output_step = step * scale_factor
    output_overlap = tile_overlap * scale_factor
    ramp = np.minimum(
        1.0,
        np.minimum(np.arange(1, output_tile_size + 1), np.arange(output_tile_size, 0, -1)) / (output_overlap + 1),
    ).astype(np.float32)
    weight = np.outer(ramp, ramp)[..., None] if tile_overlap else None
    reconstructed_image: "np.ndarray | None" = None
    weight_sum = (
        np.zeros((padded_height * scale_factor, padded_width * scale_factor, 1), np.float32) if tile_overlap else None
    )

    def place(first_tile: int, outputs: "np.ndarray") -> None:
        nonlocal reconstructed_image
        if reconstructed_image is None:
            reconstructed_image = np.zeros(
                (padded_height * scale_factor, padded_width * scale_factor, outputs.shape[1]), np.float32
            )
        for n, output in enumerate(outputs, first_tile):
            y, x = (value * output_step for value in divmod(n, tile_count_width))
            region = (slice(y, y + output_tile_size), slice(x, x + output_tile_size))
            output_image = output.transpose(1, 2, 0)
            if weight_sum is not None and weight is not None:
                reconstructed_image[region] += output_image * weight
                weight_sum[region] += weight
            else:
                reconstructed_image[region] = output_image

    for start in range(0, num_chunks, batch_size):
        batch_of_tiles = tiles_of(start, min(start + batch_size, num_chunks))
        real_tiles = len(batch_of_tiles)
        if fixed_batch is not None and real_tiles < fixed_batch:
            # Repeat a real tile: tiles do not affect each other, and blank ones could give NaNs.
            batch_of_tiles = np.pad(
                batch_of_tiles,
                ((0, fixed_batch - real_tiles), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )

        started = time.perf_counter() if inference_times is not None else 0.0
        model_output = _run_tta(infer, batch_of_tiles) if tta else infer(batch_of_tiles)
        if inference_times is not None:
            inference_times.append(time.perf_counter() - started)
        place(start, model_output[:real_tiles])

    if reconstructed_image is None:
        raise RuntimeError("The image produced no tiles.")
    if weight_sum is not None:
        reconstructed_image /= np.maximum(weight_sum, np.finfo(np.float32).eps)

    # Post-processing
    # See an explanation for image-specific normalization in the pre-processing section.
    if contrast_stretch:
        postprocessed_array = reconstructed_image.astype(np.float32)
        span = np.ptp(postprocessed_array)
        postprocessed_array = (
            (postprocessed_array - np.min(postprocessed_array)) / span
            if span
            else np.clip(postprocessed_array, 0.0, 1.0)
        )
    else:
        postprocessed_array = np.clip(reconstructed_image, 0.0, 1.0, out=reconstructed_image)
    # Scaling back to [0,255] from [0.0, 1.0], then rounding for a clean cast to uint8.
    postprocessed_array *= 255.0
    np.round(postprocessed_array, out=postprocessed_array)
    postprocessed_array = postprocessed_array.astype(np.uint8)
    if is_grayscale:
        postprocessed_array = np.squeeze(postprocessed_array, axis=-1)

    output_image = Image.fromarray(postprocessed_array)
    # Cropping the image from the overall canvas.
    output_image = output_image.crop((0, 0, image_width, image_height))
    if alpha is not None:
        alpha_mode = "LA" if is_grayscale or orig_img_mode == "LA" else "RGBA"
        output_image = output_image.convert(mode="L" if alpha_mode == "LA" else "RGB")
        output_image = output_image.convert(mode=alpha_mode)
        resized_alpha = alpha.resize(output_image.size, Image.Resampling.BICUBIC)
        output_image.putalpha(resized_alpha)
        resized_alpha.close()
        alpha.close()
    elif not is_grayscale:
        if input_image.palette:
            palette_image = Image.new("P", (1, 1))
            palette_image.putpalette(input_image.palette)
            output_image = output_image.convert(mode="RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
        else:
            output_image = output_image.convert(mode=orig_img_mode)
    else:
        output_image = output_image.convert(mode="L")

    return output_image


def denoise_single_image(
    input_image: Image.Image,
    model: ModelSession,
    *,
    batch_size: int = 64,
    tile_size: int = 128,
    context: int | None = None,
    contrast_stretch: bool = False,
    background: Literal["white", "black"] = "black",
    use_fp32: bool = False,
    tta: bool = False,
) -> Image.Image:
    """Process an image tile by tile, without blending neighbouring tiles.

    A x1 model removes noise; anything bigger enlarges the page.

    :param input_image: The page to process
    :param model: The native or TensorRT-RTX session
    :param batch_size: How many tiles to run per inference call
    :param tile_size: The side of the square tiles the model is fed
    :param contrast_stretch: Normalise each page by its own darkest and brightest pixels
    :param background: The colour the last partial tile row/column is padded with
    :param use_fp32: Fall back to fp32 input when the model declares no precision
    :param tta: Average each batch over its 8 flips/rotations (test-time augmentation)
    :param context: Input pixels the model trims off each tile; defaults to the model's detected context
    :return: The processed page with alpha and palette preserved
    """

    return _denoise_image(
        input_image,
        model,
        batch_size=batch_size,
        tile_size=tile_size,
        tile_overlap=0,
        context=context,
        contrast_stretch=contrast_stretch,
        background=background,
        use_fp32=use_fp32,
        tta=tta,
    )


def load_session(model_path: Path) -> umzi.InferenceSession:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This example uses CUDA PyTorch to load the native runtime dependencies.")

    options = umzi.SessionOptions(
        precision="fp16",
        cache_dir=model_path.parent / "cached",
        enable_cuda_graph=True,
        kernel_options={
            "fuse_gate_pool": 1,
            "tune_projections": 1,
        },
    )

    return umzi.InferenceSession(model_path, options)


def _load_rtx_session() -> _RtxSession:
    from rtx_session import create_rtx_session

    if not ONNX_MODEL.is_file():
        raise FileNotFoundError(f"Missing fixed-shape FP16 ONNX model: {ONNX_MODEL}")
    session, _ = create_rtx_session(ONNX_MODEL, BATCH_SIZE, TILE_SIZE, MODEL.parent / "cached" / "rtx")
    input_spec, output_spec = session.get_inputs()[0], session.get_outputs()[0]
    if (
        output_spec.shape != input_spec.shape
        or input_spec.type != "tensor(float16)"
        or output_spec.type != "tensor(float16)"
    ):
        raise ValueError(f"Expected an FP16 1x GateRV3 ONNX model: {ONNX_MODEL}")
    return _RtxSession(session)


def _benchmark(label: str, model: ModelSession, image: Image.Image) -> float:
    print(f"Warming up {label}...", flush=True)
    shape = (BATCH_SIZE, 3, TILE_SIZE, TILE_SIZE)
    if isinstance(model, umzi.InferenceSession):
        model.prepare(shape)
    feeds = {model.get_inputs()[0].name: np.full(shape, 0.5, dtype=np.float16)}
    for _ in range(3):
        model.run(None, feeds)

    inference_runs = []
    page_runs = []
    for _ in range(3):
        inference_times: list[float] = []
        started = time.perf_counter()
        with _denoise_image(
            image,
            model,
            batch_size=BATCH_SIZE,
            tile_size=TILE_SIZE,
            tile_overlap=0,
            context=None,
            contrast_stretch=False,
            background="black",
            use_fp32=False,
            tta=False,
            pad_final_batch=True,
            inference_times=inference_times,
        ):
            page_runs.append(time.perf_counter() - started)
        inference_runs.append(sum(inference_times))

    inference = statistics.median(inference_runs)
    pipeline = statistics.median(page_runs)
    print(f"{label}: {inference * 1000:.2f} ms inference, {pipeline * 1000:.2f} ms full page", flush=True)
    return inference


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    print(f"{IMAGE.name}, FP16, {TILE_SIZE}x{TILE_SIZE} tiles, batch {BATCH_SIZE}")
    with Image.open(IMAGE) as image:
        image.load()
        with load_session(MODEL) as native:
            native_time = _benchmark("Umzi native", native, image)
        rtx_time = _benchmark("TensorRT-RTX", _load_rtx_session(), image)

    if native_time <= rtx_time:
        print(f"Umzi native speedup: {rtx_time / native_time:.2f}x")
    else:
        print(f"TensorRT-RTX speedup: {native_time / rtx_time:.2f}x")


if __name__ == "__main__":
    main()
