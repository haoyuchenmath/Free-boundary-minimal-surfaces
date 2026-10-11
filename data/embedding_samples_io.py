"""Save evaluated embedding samples in one reusable pickle file.

The file contains sample arrays and compact reconstruction parameters. It does
not pickle the solved SQP object and never creates a companion ``__sqp.pkl``.
Loading rebuilds the solved state, or reuses ``sqp_res=...`` when supplied.
Version-1 bundles referencing an existing state file remain readable.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib import import_module
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile
from typing import Any


PathLike = str | Path
_FORMAT_NAME = "embedding_samples_bundle"
_FORMAT_VERSION = 2


def _data_io():
    """Support installation in ``data/`` or next to a standalone data_io.py."""
    if __package__:
        return import_module(".data_io", __package__)
    return import_module("data_io")


def _sample_path(file_name: PathLike) -> Path:
    path = Path(file_name).expanduser()
    if not path.name:
        raise ValueError("file_name must be nonempty.")
    if not path.suffix:
        path = path.with_suffix(".pkl")
    elif path.suffix.lower() != ".pkl":
        raise ValueError("file_name must have extension '.pkl'.")
    return _data_io()._resolve_data_path(path).resolve()


def _existing_state_path(file_name: PathLike, sample_path: Path) -> Path:
    reference = Path(file_name).expanduser()
    if not reference.suffix:
        reference = reference.with_suffix(".pkl")
    if reference.is_absolute():
        candidate = reference
    else:
        candidate = sample_path.parent / reference
        if not candidate.is_file():
            candidate = _data_io()._resolve_data_path(reference)
    if not candidate.is_file():
        raise FileNotFoundError(f"Saved Steklov parameter file not found: {candidate}")
    return candidate.resolve()


def save_embedding_samples(
    samples,
    file_name: PathLike,
    *,
    sqp_state_file: PathLike | None = None,
) -> Path:
    """Save arrays and compact state parameters together, writing only one file.

    ``sqp_state_file`` may name an already saved parameter file; its parameters
    are copied into the sample file instead of extracted from ``samples.sqp_res``.
    No state file is created, overwritten, or required on subsequent loads.
    Bare filenames are saved beside data_io.py; explicit paths are also accepted.
    """
    if not is_dataclass(samples) or isinstance(samples, type):
        raise TypeError("samples must be an embedding-sample dataclass instance.")
    if "sqp_res" not in {item.name for item in fields(samples)}:
        raise TypeError("samples must expose a dataclass field named 'sqp_res'.")

    sample_path = _sample_path(file_name)
    io = _data_io()
    if sqp_state_file is None:
        parameters = io._parameters_from_result(samples.sqp_res)
    else:
        state_path = _existing_state_path(sqp_state_file, sample_path)
        _, saved = io._read_parameter_file(state_path)
        parameters = {key: saved[key] for key in io._REQUIRED_KEYS}
        for key in ("returned_iteration", "best_score", "returned_from_best",
                    "density_transform", "softplus_clip"):
            if key in saved:
                parameters[key] = saved[key]

    payload = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "sample_class_module": type(samples).__module__,
        "sample_class_name": type(samples).__name__,
        "state_parameters": parameters,
        "sample_fields": {
            item.name: getattr(samples, item.name)
            for item in fields(samples)
            if item.name != "sqp_res"
        },
    }

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=sample_path.parent,
            prefix=sample_path.name + ".", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(sample_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return sample_path


def load_embedding_samples(
    file_name: PathLike,
    *,
    sqp_state_file: PathLike | None = None,
    sqp_res=None,
    sample_class: type | None = None,
    load_options: dict[str, Any] | None = None,
):
    """Load samples without repeating sampling or writing any state file.

    By default, rebuild the solved state from parameters inside a version-2
    file. Pass ``sqp_res=existing_result`` to skip that rebuild. An explicit
    ``sqp_state_file`` can override the embedded parameters or repair a legacy
    version-1 reference. Supply ``sample_class`` if the original sample class
    has moved to another module. ``load_options`` contains options for load_data.
    """
    if sqp_res is not None and sqp_state_file is not None:
        raise ValueError("Supply either sqp_res or sqp_state_file, not both.")

    sample_path = _sample_path(file_name)
    if not sample_path.is_file():
        raise FileNotFoundError(f"Embedding-sample file not found: {sample_path}")
    with sample_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict) or payload.get("format") != _FORMAT_NAME:
        raise ValueError(f"{sample_path} is not an embedding-sample bundle.")
    version = payload.get("version")
    if version not in (1, _FORMAT_VERSION):
        raise ValueError(f"Unsupported embedding-sample bundle version: {version!r}.")

    cached = payload.get("sample_fields")
    if not isinstance(cached, dict):
        raise ValueError("The embedding-sample file does not contain sample_fields.")

    embedded_state = payload.get("state_parameters") or {}
    saved_transform = embedded_state.get("density_transform")
    if sqp_res is not None and saved_transform is not None:
        provided_state = getattr(sqp_res, "ss", None)
        provided_transform = getattr(provided_state, "density_transform", None)
        if provided_transform != saved_transform:
            raise ValueError(
                f"Sample bundle uses density_transform={saved_transform!r}, "
                f"but supplied sqp_res uses {provided_transform!r}."
            )

    if sample_class is None:
        module_name = str(payload["sample_class_module"])
        class_name = str(payload["sample_class_name"])
        try:
            sample_class = getattr(import_module(module_name), class_name)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"Cannot restore {module_name}.{class_name}; "
                "pass sample_class=YourCurrentSampleClass if that class has moved."
            ) from error
    if not isinstance(sample_class, type) or not is_dataclass(sample_class):
        raise TypeError("sample_class must be an embedding-sample dataclass type.")

    expected = {item.name for item in fields(sample_class) if item.name != "sqp_res"}
    missing = expected - cached.keys()
    unexpected = cached.keys() - expected
    if missing or unexpected:
        raise ValueError(
            "Saved samples and the current sample class do not agree. "
            f"Missing fields: {sorted(missing)}; unexpected fields: {sorted(unexpected)}."
        )

    if sqp_res is None:
        io = _data_io()
        options = dict(load_options or {})
        modes = cached.get("modes", ())
        options.setdefault("n_keep", max(10, max(modes, default=0) + 1))
        if sqp_state_file is not None:
            state_path = _existing_state_path(sqp_state_file, sample_path)
            sqp_res = io.load_data(state_path, **options)
        elif version == _FORMAT_VERSION:
            sqp_res = io._rebuild_from_parameters(
                payload.get("state_parameters"), source=sample_path, **options,
            )
        else:
            reference = payload.get("sqp_state_file")
            if not reference:
                raise ValueError("Version-1 bundle has no SQP-state reference.")
            try:
                state_path = _existing_state_path(reference, sample_path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Legacy samples require the recorded state file {reference!r}. "
                    "Pass sqp_res=existing_result or sqp_state_file='existing.pkl'."
                ) from error
            sqp_res = io.load_data(state_path, **options)

    return sample_class(sqp_res=sqp_res, **cached)


__all__ = ["load_embedding_samples", "save_embedding_samples"]
