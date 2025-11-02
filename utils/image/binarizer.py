"""Deep-learning-based image binarization utility.

This module wraps the Eynollah binarization network exported to ONNX. The model is described at
https://dl.acm.org/doi/10.1145/3604951.3605513, and weights are fetched from the Hugging Face
Hub on first use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from churro.utils.log_utils import logger


DEFAULT_REPO_ID = "stanford-oval/eynollah_binarizer_onnx"
DEFAULT_ONNX_FILENAME = "eynollah_binarizer.onnx"
MODEL_HEIGHT = 224
MODEL_WIDTH = 448
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
GPU_PREFERRED_PROVIDERS: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
)


@dataclass(slots=True)
class _PredictBatchContext:
    """Book-keeping for trimming predictions back to original image dimensions."""

    original_shape: tuple[int, int]
    index_start_h: int
    index_start_w: int


@dataclass(slots=True)
class _PatchBatchContext:
    """Batch-level state for reconstructing tiled predictions."""

    image: np.ndarray
    prediction: np.ndarray
    nxf: int
    nyf: int


@dataclass(slots=True)
class _PatchMetadata:
    """Metadata describing a single tiled patch in a batched inference run."""

    image_index: int
    i: int
    j: int
    index_x_d: int
    index_x_u: int
    index_y_d: int
    index_y_u: int


class ImageBinarizer:
    """High-level wrapper around the ONNX-exported Eynollah binarizer model.

    The class encapsulates model discovery, execution provider selection, and helpers for
    tiling oversized images into overlapping patches so that the ONNX Runtime session can
    process arbitrary page sizes. Public helpers are provided for both numpy arrays and
    Pillow images to match the data formats used throughout the HistoryGenie pipelines.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_ONNX_FILENAME,
        providers: Sequence[str] | None = None,
        max_patch_batch_size: int | None = 8,
    ) -> None:
        """Instantiate the binarizer and prepare the ONNX Runtime session.

        Args:
            repo_id: Hugging Face Hub repository that stores the ONNX artifacts.
            filename: Model filename relative to the repository snapshot.
            providers: Optional explicit list of ONNX Runtime execution providers.
                When ``None`` (default) the runtime will prefer GPU providers if available.
            max_patch_batch_size: Optional upper bound on the number of tiled patches
                processed per inference call. ``None`` means unbounded, otherwise values
                must be positive.

        Raises:
            ValueError: If ``max_patch_batch_size`` is provided but evaluates to < 1.
            FileNotFoundError: If the requested ``filename`` cannot be located in the
                resolved repository snapshot.
        """
        self.model_height = MODEL_HEIGHT
        self.model_width = MODEL_WIDTH
        if max_patch_batch_size is not None and max_patch_batch_size < 1:
            raise ValueError("max_patch_batch_size must be >= 1 when provided")
        self._max_patch_batch_size = max_patch_batch_size

        artifacts_dir = self._ensure_artifacts_dir()
        onnx_path = self._resolve_model_path(artifacts_dir, repo_id, filename)

        session_providers = tuple(providers) if providers else self._select_execution_providers()
        self.session = self._create_session(onnx_path, session_providers)
        self.input_name = self.session.get_inputs()[0].name

    def _effective_patch_batch_size(self, requested: int) -> int:
        """Clamp a requested patch batch size to the configured safe interval.

        Args:
            requested: Raw patch batch size requested by the caller.

        Returns:
            int: Value guaranteed to be at least 1 and, if configured, not exceed
                ``max_patch_batch_size``.
        """
        safe = max(1, requested)
        if self._max_patch_batch_size is not None:
            safe = min(safe, self._max_patch_batch_size)
        return safe

    def _select_execution_providers(self) -> tuple[str, ...]:
        """Choose the best available ONNX Runtime execution providers.

        The method prefers GPU accelerators when present and logs the final selection.

        Returns:
            tuple[str, ...]: Providers passed to ``onnxruntime.InferenceSession``.
        """
        available = set(ort.get_available_providers())
        if not available:
            logger.warning("No ONNX Runtime execution providers detected; defaulting to CPU.")
            chosen = DEFAULT_PROVIDERS
        else:
            for provider in GPU_PREFERRED_PROVIDERS:
                if provider in available:
                    chosen = (provider, "CPUExecutionProvider")
                    break
            else:
                if DEFAULT_PROVIDERS[0] in available:
                    chosen = DEFAULT_PROVIDERS
                else:
                    logger.warning(
                        "Preferred ONNX Runtime providers unavailable; using first detected provider instead."
                    )
                    # Preserve deterministic order when falling back to arbitrary providers
                    chosen = tuple(sorted(available))

        logger.info(f"Configured ONNX Runtime providers: {chosen}")
        return chosen

    def _predict(self, img: np.ndarray, n_batch_inference: int) -> np.ndarray:
        """Predict a binarization mask for a single image.

        Args:
            img: HxWxC image array in RGB order with values in ``[0, 255]``.
            n_batch_inference: Target mini-batch size used for tiled inference.

        Returns:
            np.ndarray: Binary class prediction aligned with ``img``.
        """
        predictions = self._predict_batch((img,), n_batch_inference)
        return predictions[0]

    def _predict_batch(
        self, imgs: Sequence[np.ndarray], n_batch_inference: int
    ) -> list[np.ndarray]:
        """Predict binarization masks for a sequence of images.

        Args:
            imgs: Iterable of HxWxC RGB arrays.
            n_batch_inference: Target mini-batch size forwarded to tiled inference.

        Returns:
            list[np.ndarray]: Per-image binary masks with original shapes preserved.

        Raises:
            ValueError: If any image does not have three dimensions.
        """
        if not imgs:
            return []

        contexts: list[_PredictBatchContext] = []
        padded_images: list[np.ndarray] = []
        for img in imgs:
            if img.ndim != 3:
                raise ValueError("Images must be HxWxC arrays.")

            img_org_h, img_org_w = img.shape[:2]
            img_padded, index_start_h, index_start_w = self._pad_image(
                img, self.model_height, self.model_width
            )

            contexts.append(
                _PredictBatchContext(
                    original_shape=(img_org_h, img_org_w),
                    index_start_h=index_start_h,
                    index_start_w=index_start_w,
                )
            )
            padded_images.append(img_padded)

        effective_batch = self._effective_patch_batch_size(n_batch_inference)
        predictions = self._predict_with_patches_batch(
            padded_images, self.model_height, self.model_width, effective_batch
        )

        trimmed: list[np.ndarray] = []
        for prediction, ctx in zip(predictions, contexts, strict=False):
            index_start_h = ctx.index_start_h
            index_start_w = ctx.index_start_w
            orig_h, orig_w = ctx.original_shape
            cropped = prediction[
                index_start_h : index_start_h + orig_h,
                index_start_w : index_start_w + orig_w,
            ]
            trimmed.append(cropped.astype(np.uint8))

        return trimmed

    @staticmethod
    def _ensure_artifacts_dir() -> Path:
        """Return the cache directory that stores downloaded ONNX artifacts.

        Returns:
            Path: Fully resolved path to the ``artifacts`` folder located next to this
                module. The directory is created if it does not yet exist.
        """
        artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    @staticmethod
    def _create_session(model_path: Path, providers: Sequence[str]) -> ort.InferenceSession:
        """Configure ONNX Runtime and create an inference session.

        Args:
            model_path: Location of the ONNX file on disk.
            providers: Ordered execution providers to supply to ONNX Runtime.

        Returns:
            onnxruntime.InferenceSession: Ready-to-run inference session.
        """
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, os.cpu_count() or 1)
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=list(providers),
        )

    def _resolve_model_path(self, artifacts_dir: Path, repo_id: str, filename: str) -> Path:
        """Return the path to the ONNX model, downloading it if absent.

        Args:
            artifacts_dir: Directory used for storing fetched artifacts.
            repo_id: Hugging Face Hub repository to search.
            filename: Expected ONNX file name within ``repo_id``.

        Returns:
            Path: Resolved path to the local ONNX model.

        Raises:
            FileNotFoundError: If ``filename`` is missing from the downloaded snapshot.
        """
        local_onnx_path = artifacts_dir / filename
        if local_onnx_path.exists():
            return local_onnx_path

        snapshot_path = self._download_snapshot(repo_id)
        candidate_path = snapshot_path / filename
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"File {filename} not found in snapshot for repo {repo_id}. "
                "Ensure the ONNX artifact has been uploaded."
            )
        return candidate_path

    @staticmethod
    def _download_snapshot(repo_id: str) -> Path:
        """Download a Hugging Face Hub snapshot with local cache fallback.

        Args:
            repo_id: Hugging Face repository identifier.

        Returns:
            Path: Local cache directory containing the repository snapshot.
        """
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        try:
            return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        except LocalEntryNotFoundError:
            snapshot_path = Path(snapshot_download(repo_id=repo_id))
            logger.info(
                f"Downloaded ONNX model from Hugging Face Hub repo {repo_id} to cache {snapshot_path}"
            )
            return snapshot_path

    @staticmethod
    def _pad_image(
        img: np.ndarray, model_height: int, model_width: int
    ) -> tuple[np.ndarray, int, int]:
        """Pad an image symmetrically until it matches the model receptive field.

        Args:
            img: HxWxC image to be padded.
            model_height: Minimum height expected by the ONNX model.
            model_width: Minimum width expected by the ONNX model.

        Returns:
            tuple[np.ndarray, int, int]: Padded image plus the top-left indices at which
                the original pixels begin inside the padded array.
        """
        img_h, img_w = img.shape[:2]
        channels = img.shape[2]

        if img_h < model_height and img_w >= model_width:
            img_padded = np.zeros((model_height, img_w, channels), dtype=img.dtype)
            index_start_h = int(abs(img_h - model_height) / 2.0)
            index_start_w = 0
            img_padded[index_start_h : index_start_h + img_h, :, :] = img

        elif img_h >= model_height and img_w < model_width:
            img_padded = np.zeros((img_h, model_width, channels), dtype=img.dtype)
            index_start_h = 0
            index_start_w = int(abs(img_w - model_width) / 2.0)
            img_padded[:, index_start_w : index_start_w + img_w, :] = img

        elif img_h < model_height and img_w < model_width:
            img_padded = np.zeros((model_height, model_width, channels), dtype=img.dtype)
            index_start_h = int(abs(img_h - model_height) / 2.0)
            index_start_w = int(abs(img_w - model_width) / 2.0)
            img_padded[
                index_start_h : index_start_h + img_h,
                index_start_w : index_start_w + img_w,
                :,
            ] = img
        else:
            index_start_h = 0
            index_start_w = 0
            img_padded = np.copy(img)

        return img_padded, index_start_h, index_start_w

    @staticmethod
    def _resize_numpy_image(
        image: np.ndarray,
        *,
        scale: float | None = None,
        size: tuple[int, int] | None = None,
        resample: int = Image.Resampling.LANCZOS,
    ) -> np.ndarray:
        """Resize a numpy-backed image using Pillow as the backend.

        Args:
            image: Input array with shape HxWxC or HxW.
            scale: Optional multiplicative factor applied to width and height.
            size: Optional explicit output size as (width, height). When provided it
                overrides ``scale``.
            resample: Pillow resampling filter to use.

        Returns:
            np.ndarray: Resized array cast back to the original dtype when possible.

        Raises:
            ValueError: If neither ``scale`` nor ``size`` are provided.
            TypeError: If ``image`` is not a NumPy array or cannot be coerced to one.
        """
        if scale is None and size is None:
            raise ValueError("Either scale or size must be provided for resizing.")

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if size is None:
            height, width = image.shape[:2]
            target_width = max(1, int(round(width * scale)))
            target_height = max(1, int(round(height * scale)))
            size = (target_width, target_height)

        array = image
        original_dtype = image.dtype
        unit_range_float = False
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                scaled = array
                if scaled.size:
                    max_value = float(np.max(scaled))
                else:  # pragma: no cover - empty guard
                    max_value = 0.0
                if max_value <= 1.0:
                    scaled = scaled * 255.0
                    unit_range_float = True
                array = np.clip(scaled, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        try:
            pil_image = Image.fromarray(array)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise TypeError("Unsupported image format for resizing.") from exc

        resized = pil_image.resize(size, resample=resample)
        resized_array = np.asarray(resized)

        if original_dtype == np.uint8:
            return resized_array

        if np.issubdtype(original_dtype, np.floating):
            if unit_range_float:
                resized_array = resized_array.astype(np.float32) / 255.0
            else:
                resized_array = resized_array.astype(np.float32)
            return resized_array.astype(original_dtype, copy=False)

        return resized_array.astype(original_dtype, copy=False)

    def _predict_with_patches(
        self,
        img: np.ndarray,
        model_height: int,
        model_width: int,
        n_batch_inference: int,
    ) -> np.ndarray:
        """Predict a mask by tiling a single image into overlapping patches.

        Args:
            img: HxWxC RGB array to process.
            model_height: Patch height consumed by the ONNX model.
            model_width: Patch width consumed by the ONNX model.
            n_batch_inference: Preferred batch size for forwarding patches.

        Returns:
            np.ndarray: Binary mask with the same height and width as ``img``.

        Raises:
            ValueError: If derived patch dimensions would be non-positive.
        """
        margin = int(0.1 * model_width)
        width_mid = model_width - 2 * margin
        height_mid = model_height - 2 * margin
        if width_mid <= 0 or height_mid <= 0:
            raise ValueError(
                "Computed patch dimensions must be positive; check model size and margin."
            )

        img_h, img_w = img.shape[:2]

        prediction_true = np.zeros((img_h, img_w), dtype=np.uint8)

        nxf = int(np.ceil(img_w / width_mid))
        nyf = int(np.ceil(img_h / height_mid))

        n_batch_inference = self._effective_patch_batch_size(n_batch_inference)
        channels = img.shape[2]
        batch_buffer = np.empty(
            (n_batch_inference, model_height, model_width, channels), dtype=np.float32
        )
        batch_metadata: list[tuple[int, int, int, int, int, int]] = []
        current_batch_size = 0

        def flush_batch() -> None:
            nonlocal current_batch_size
            if current_batch_size == 0:
                return

            batch = batch_buffer[:current_batch_size]
            outputs = self.session.run(None, {self.input_name: batch})
            batch_seg = np.argmax(np.asarray(outputs[0], dtype=np.float32), axis=-1).astype(
                np.uint8
            )

            h = self.model_height
            w = self.model_width

            for seg_patch, meta in zip(batch_seg, batch_metadata, strict=False):
                i, j, index_x_d, _index_x_u, index_y_d, _index_y_u = meta

                if nyf == 1:
                    crop_y_start = 0
                    crop_y_end = h
                else:
                    crop_y_start = 0 if j == 0 else margin
                    crop_y_end = h if j == nyf - 1 else h - margin

                if nxf == 1:
                    crop_x_start = 0
                    crop_x_end = w
                else:
                    crop_x_start = 0 if i == 0 else margin
                    crop_x_end = w if i == nxf - 1 else w - margin

                y_start = index_y_d + crop_y_start
                y_end = index_y_d + crop_y_end
                x_start = index_x_d + crop_x_start
                x_end = index_x_d + crop_x_end

                prediction_true[y_start:y_end, x_start:x_end] = seg_patch[
                    crop_y_start:crop_y_end, crop_x_start:crop_x_end
                ]

            batch_metadata.clear()
            current_batch_size = 0

        num_patches = 0
        for i in range(nxf):
            for j in range(nyf):
                index_x_d = i * width_mid
                index_x_u = min(index_x_d + model_width, img_w)
                if index_x_u == img_w:
                    index_x_d = img_w - model_width

                index_y_d = j * height_mid
                index_y_u = min(index_y_d + model_height, img_h)
                if index_y_u == img_h:
                    index_y_d = img_h - model_height

                np.multiply(
                    img[index_y_d:index_y_u, index_x_d:index_x_u, :],
                    1.0 / 255.0,
                    out=batch_buffer[current_batch_size],
                    casting="unsafe",
                )
                batch_metadata.append((i, j, index_x_d, index_x_u, index_y_d, index_y_u))
                num_patches += 1
                current_batch_size += 1

                if current_batch_size == n_batch_inference:
                    flush_batch()

        flush_batch()
        logger.debug(f"Split image into {num_patches} patches for binarization.")

        return prediction_true

    def _predict_with_patches_batch(
        self,
        imgs: Sequence[np.ndarray],
        model_height: int,
        model_width: int,
        n_batch_inference: int,
    ) -> list[np.ndarray]:
        """Predict masks for many images by tiling them into overlapping patches.

        Args:
            imgs: Iterable of HxWxC RGB arrays.
            model_height: Patch height consumed by the ONNX model.
            model_width: Patch width consumed by the ONNX model.
            n_batch_inference: Preferred patch batch size for inference calls.

        Returns:
            list[np.ndarray]: Binary masks aligned with each input image.

        Raises:
            ValueError: If derived patch dimensions would be non-positive or images have
                differing channel counts.
        """
        if not imgs:
            return []

        margin = int(0.1 * model_width)
        width_mid = model_width - 2 * margin
        height_mid = model_height - 2 * margin
        if width_mid <= 0 or height_mid <= 0:
            raise ValueError(
                "Computed patch dimensions must be positive; check model size and margin."
            )

        channels = imgs[0].shape[2]
        for img in imgs:
            if img.shape[2] != channels:
                raise ValueError("All images must have the same channel count.")

        contexts: list[_PatchBatchContext] = []
        for img in imgs:
            img_h, img_w = img.shape[:2]
            contexts.append(
                _PatchBatchContext(
                    image=img,
                    prediction=np.zeros((img_h, img_w), dtype=np.uint8),
                    nxf=int(np.ceil(img_w / width_mid)),
                    nyf=int(np.ceil(img_h / height_mid)),
                )
            )

        n_batch_inference = self._effective_patch_batch_size(n_batch_inference)
        batch_buffer = np.empty(
            (n_batch_inference, model_height, model_width, channels), dtype=np.float32
        )
        batch_metadata: list[_PatchMetadata] = []
        current_batch_size = 0

        def flush_batch() -> None:
            nonlocal current_batch_size
            if current_batch_size == 0:
                return

            batch = batch_buffer[:current_batch_size]
            outputs = self.session.run(None, {self.input_name: batch})
            batch_seg = np.argmax(np.asarray(outputs[0], dtype=np.float32), axis=-1).astype(
                np.uint8
            )

            for seg_patch, meta in zip(batch_seg, batch_metadata, strict=False):
                context = contexts[meta.image_index]
                prediction = context.prediction
                nxf = context.nxf
                nyf = context.nyf

                if nyf == 1:
                    crop_y_start = 0
                    crop_y_end = model_height
                else:
                    crop_y_start = 0 if meta.j == 0 else margin
                    crop_y_end = model_height if meta.j == nyf - 1 else model_height - margin

                if nxf == 1:
                    crop_x_start = 0
                    crop_x_end = model_width
                else:
                    crop_x_start = 0 if meta.i == 0 else margin
                    crop_x_end = model_width if meta.i == nxf - 1 else model_width - margin

                y_start = meta.index_y_d + crop_y_start
                y_end = meta.index_y_d + crop_y_end
                x_start = meta.index_x_d + crop_x_start
                x_end = meta.index_x_d + crop_x_end

                prediction[y_start:y_end, x_start:x_end] = seg_patch[
                    crop_y_start:crop_y_end, crop_x_start:crop_x_end
                ]

            batch_metadata.clear()
            current_batch_size = 0

        total_patches = 0
        for img_idx, context in enumerate(contexts):
            img = context.image
            img_h, img_w = img.shape[:2]
            nxf = context.nxf
            nyf = context.nyf

            for i in range(nxf):
                for j in range(nyf):
                    index_x_d = i * width_mid
                    index_x_u = min(index_x_d + model_width, img_w)
                    if index_x_u == img_w:
                        index_x_d = img_w - model_width

                    index_y_d = j * height_mid
                    index_y_u = min(index_y_d + model_height, img_h)
                    if index_y_u == img_h:
                        index_y_d = img_h - model_height

                    np.multiply(
                        img[index_y_d:index_y_u, index_x_d:index_x_u, :],
                        1.0 / 255.0,
                        out=batch_buffer[current_batch_size],
                        casting="unsafe",
                    )
                    batch_metadata.append(
                        _PatchMetadata(
                            image_index=img_idx,
                            i=i,
                            j=j,
                            index_x_d=index_x_d,
                            index_x_u=index_x_u,
                            index_y_d=index_y_d,
                            index_y_u=index_y_u,
                        )
                    )
                    current_batch_size += 1
                    total_patches += 1

                    if current_batch_size == n_batch_inference:
                        flush_batch()

        flush_batch()
        logger.debug(
            f"Split {len(imgs)} images into {total_patches} patches for batched binarization."
        )
        return [context.prediction for context in contexts]

    def _binarize_numpy_batch(
        self, images: Sequence[np.ndarray], scale: float = 1.0, n_batch_inference: int = 16
    ) -> list[np.ndarray]:
        """Binarize many numpy-backed images, batching patches across the group.

        Args:
            images: Sequence of HxWxC RGB arrays to binarize.
            scale: Optional scaling factor applied prior to inference for all images.
            n_batch_inference: Preferred patch batch size for tiled inference.

        Returns:
            list[np.ndarray]: Binary masks with uint8 dtype where foreground is 0 and
                background is 255.
        """
        if not images:
            return []

        working_images: list[np.ndarray] = []
        original_shapes: list[tuple[int, int]] = []
        scale_factors: list[bool] = []
        for image in images:
            original_shapes.append(image.shape[:2])
            if np.isclose(scale, 1.0):
                working_images.append(image)
                scale_factors.append(False)
            else:
                resized = self._resize_numpy_image(image, scale=scale)
                working_images.append(resized)
                scale_factors.append(True)

        predictions = self._predict_batch(working_images, n_batch_inference=n_batch_inference)

        binaries: list[np.ndarray] = []
        for prediction, original_shape, needs_rescale in zip(
            predictions, original_shapes, scale_factors, strict=False
        ):
            binary_image = np.where(prediction == 0, 255, 0).astype(np.uint8)
            if needs_rescale:
                binary_image = self._resize_numpy_image(
                    binary_image,
                    size=(original_shape[1], original_shape[0]),
                )
            binaries.append(binary_image)

        return binaries

    def binarize_pil_batch(
        self, images: Sequence[Image.Image], scale: float = 1.0, n_batch_inference: int = 16
    ) -> list[Image.Image]:
        """Binarize many Pillow images and return outputs in mode ``L``.

        Args:
            images: Sequence of ``PIL.Image.Image`` instances to process.
            scale: Optional scaling factor applied prior to inference.
            n_batch_inference: Preferred patch batch size for tiled inference.

        Returns:
            list[Image.Image]: Binarized images in single-channel (``L``) mode.

        Raises:
            TypeError: If any element of ``images`` is not a Pillow image.
        """
        if not images:
            return []

        normalized: list[Image.Image] = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("images must be instances of PIL.Image.Image")

            try:
                image = ImageOps.exif_transpose(image)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug(f"Failed to normalize EXIF orientation: {exc}")

            if image.mode != "RGB":
                image = image.convert("RGB")

            normalized.append(image)

        np_images = [np.asarray(img) for img in normalized]
        binary_arrays = self._binarize_numpy_batch(
            np_images, scale=scale, n_batch_inference=n_batch_inference
        )
        return [Image.fromarray(arr, mode="L") for arr in binary_arrays]

    def binarize_pil(
        self, image: Image.Image, scale: float = 1.0, n_batch_inference: int = 16
    ) -> Image.Image:
        """Binarize a Pillow image and return the result in mode ``L``.

        Args:
            image: Pillow image to process.
            scale: Optional scaling factor applied prior to inference.
            n_batch_inference: Preferred patch batch size for tiled inference.

        Returns:
            PIL.Image.Image: Binarized single-channel output.
        """
        results = self.binarize_pil_batch(
            (image,), scale=scale, n_batch_inference=n_batch_inference
        )
        return results[0]
