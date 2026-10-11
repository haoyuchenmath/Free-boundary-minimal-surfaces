"""Save parameters and rebuild weighted-Steklov states without optimizer imports.

Install this module in the project-level ``data`` package. Bare filenames are
resolved beside this file, independently of the notebook working directory.
Saved files contain plain parameter dictionaries, never optimizer instances.
Numerical solver modules are imported only when rebuilding a solved state.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np



DATA_DIR = Path(__file__).resolve().parent

_REQUIRED_KEYS = frozenset(
    {
        "cs",
        "rs",
        "cos_coeffs",
        "sin_coeffs",
        "tau",
        "Ns",
    }
)


@dataclass
class SteklovDataHistory:
    """Iteration fields used by downstream code; independent of any optimizer."""

    iterations: list[int] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    sigmas_nontrivial: list[np.ndarray] = field(default_factory=list)
    sigmas_tilde_nontrivial: list[np.ndarray] = field(default_factory=list)
    tau: list[complex] = field(default_factory=list)
    cs: list[np.ndarray] = field(default_factory=list)
    rs: list[np.ndarray] = field(default_factory=list)
    accepted: list[bool] = field(default_factory=list)
    step_scale: list[float] = field(default_factory=list)
    gap_after_sigma1: list[float] = field(default_factory=list)
    message: list[str] = field(default_factory=list)


@dataclass
class SteklovDataResult:
    """Rebuilt state exposing the result fields used by evaluation and plotting."""

    tau: complex
    ss: Any
    lp: Any
    pb: Any
    gr: Any
    cs: np.ndarray
    rs: np.ndarray
    cos_coeffs: list[np.ndarray]
    sin_coeffs: list[np.ndarray]
    history: SteklovDataHistory
    returned_iteration: int = 0
    best_score: float = np.nan
    returned_from_best: bool = False


class _ParameterUnpickler(pickle.Unpickler):
    """Read optional legacy history without importing its former optimizer."""

    def find_class(self, module: str, name: str):
        if module.rsplit(".", 1)[-1] in {
            "optimizer", "steklov_optimizer_midline_refinement"
        }:
            if name == "SteklovOptimizationHistory":
                return SteklovDataHistory
            if name == "SteklovOptimizationResult":
                return SteklovDataResult
        return super().find_class(module, name)


def _resolve_data_path(load_path: str | Path) -> Path:
    """Resolve a bare filename relative to the directory containing this file."""
    path = Path(load_path).expanduser()

    if path.is_absolute():
        return path

    # The main interface is load_data("filename.pkl"): data and loader live
    # together, so this does not depend on the notebook's working directory.
    if path.parent == Path("."):
        return DATA_DIR / path.name

    return path.resolve()


def _read_parameter_file(load_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _resolve_data_path(load_path)

    if not path.is_file():
        raise FileNotFoundError(f"Saved Steklov parameter file not found: {path}")

    with path.open("rb") as file:
        params = _ParameterUnpickler(file).load()

    if not isinstance(params, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, found {type(params).__name__}."
        )

    missing = _REQUIRED_KEYS.difference(params)
    if missing:
        raise KeyError(
            "Saved parameter file is missing required keys: "
            + ", ".join(sorted(missing))
        )

    return path, params


def _loaded_snapshot_history(
    pb: Any,
    *,
    tau: complex,
    cs: np.ndarray,
    rs: np.ndarray,
    objective_mode: int,
    n_record_nontrivial: int,
) -> SteklovDataHistory:
    """Create a one-state history when the original history was not saved."""
    history = SteklovDataHistory()

    n_record = int(n_record_nontrivial)
    sigmas = np.full(n_record, np.nan, dtype=float)
    sigmas_tilde = np.full(n_record, np.nan, dtype=float)
    n_available = max(0, min(n_record, len(pb.sigmas) - 1))

    if n_available:
        sigmas[:n_available] = np.real(pb.sigmas[1 : 1 + n_available])
        sigmas_tilde[:n_available] = np.real(
            pb.sigmas_tilde[1 : 1 + n_available]
        )

    gap = np.nan
    if len(pb.sigmas_tilde) > 2:
        gap = float(np.real(pb.sigmas_tilde[2] - pb.sigmas_tilde[1]))

    history.iterations.append(0)
    history.objective.append(float(np.real(pb.sigmas_tilde[objective_mode])))
    history.sigmas_nontrivial.append(sigmas)
    history.sigmas_tilde_nontrivial.append(sigmas_tilde)
    history.tau.append(complex(tau))
    history.cs.append(cs.copy())
    history.rs.append(rs.copy())
    history.accepted.append(True)
    history.step_scale.append(0.0)
    history.gap_after_sigma1.append(gap)
    history.message.append(
        "loaded state; original optimization history was not saved"
    )

    return history


def _rebuild_from_parameters(
    params: dict[str, Any],
    *,
    source: str | Path = "embedded sample parameters",
    n_keep: int = 10,
    objective_mode: int = 1,
    n_record_nontrivial: int = 5,
    positive_rho: bool = True,
    softplus_clip: float | None = None,
    density_transform: str | None = None,
    mean_weight: str = "ds",
    check_disjoint: bool = True,
    margin: float = 1e-5,
    search_radius: int = 3,
    wrap_centers: bool = False,
    zero_tol: float = 1e-10,
    imag_tol: float = 1e-8,
    verbose: bool = True,
) -> SteklovDataResult:
    """Rebuild a state from a plain parameter dictionary, without optimizer code."""
    from src.layer_potentials import MyLayerPot
    from src.partial_derivatives import SteklovGradients
    from src.problems_solver import MyProblems
    from src.segment_shapes import circles

    if not isinstance(params, dict):
        raise TypeError("Saved parameters must be a dictionary.")
    missing = _REQUIRED_KEYS.difference(params)
    if missing:
        raise KeyError("Saved parameter file is missing required keys: "
                       + ", ".join(sorted(missing)))
    path = source

    saved_transform = params.get("density_transform")
    if saved_transform is None and density_transform is None:
        raise ValueError(
            "This unversioned Steklov file predates the density-transform marker. "
            "For pre-submission files call load_data(..., density_transform='softplus'); "
            "if the file truly stores log-density coefficients, use 'exp'. "
            "Never interpret old Fourier coefficients as log(rho) automatically."
        )
    if (saved_transform is not None and density_transform is not None
            and saved_transform != density_transform):
        raise ValueError(
            f"Saved density_transform={saved_transform!r}, requested {density_transform!r}. "
            "Convert the coefficients explicitly before changing parametrization."
        )
    density_transform = density_transform or saved_transform
    if density_transform not in ("exp", "softplus", "identity"):
        raise ValueError("Invalid saved density_transform.")
    if density_transform == "identity":
        positive_rho = False
    if softplus_clip is None:
        softplus_clip = params.get("softplus_clip")

    cs = np.asarray(params["cs"], dtype=complex).reshape(-1)
    rs = np.asarray(params["rs"], dtype=float).reshape(-1)
    cos_coeffs = [
        np.asarray(coeff, dtype=float).reshape(-1).copy()
        for coeff in params["cos_coeffs"]
    ]
    sin_coeffs = [
        np.asarray(coeff, dtype=float).reshape(-1).copy()
        for coeff in params["sin_coeffs"]
    ]
    tau = complex(params["tau"])

    Ns_array = np.asarray(params["Ns"], dtype=int)
    Ns: int | np.ndarray
    if Ns_array.ndim == 0:
        Ns = int(Ns_array.item())
    else:
        Ns = Ns_array.reshape(-1)

    M = cs.size
    if M == 0:
        raise ValueError("Loaded cs must contain at least one center.")
    if rs.size != M:
        raise ValueError(f"Loaded len(rs)={rs.size}, but len(cs)={M}.")
    if len(cos_coeffs) != M or len(sin_coeffs) != M:
        raise ValueError(
            "Loaded cosine and sine coefficient lists must both have "
            f"length M={M}."
        )
    if not np.isscalar(Ns) and np.asarray(Ns).size != M:
        raise ValueError(
            f"Loaded Ns has length {np.asarray(Ns).size}, but M={M}."
        )

    objective_mode = int(objective_mode)
    n_record_nontrivial = int(n_record_nontrivial)
    if objective_mode < 0:
        raise ValueError("objective_mode must be nonnegative.")
    if n_record_nontrivial < 0:
        raise ValueError("n_record_nontrivial must be nonnegative.")

    # Keep the requested modes and one extra mode for the first spectral gap.
    n_keep = max(
        int(n_keep),
        objective_mode + 2,
        n_record_nontrivial + 1,
    )

    ss = circles(
        cs=cs,
        rs=rs,
        Ns=Ns,
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        tau=tau,
        check_disjoint=check_disjoint,
        margin=margin,
        search_radius=search_radius,
        wrap_centers=wrap_centers,
        positive_rho=positive_rho,
        softplus_clip=softplus_clip,
        density_transform=density_transform,
    )
    lp = MyLayerPot(ss, tau=tau)
    pb = MyProblems(
        lp,
        n_keep=n_keep,
        mean_weight=mean_weight,
        normalize=True,
        zero_tol=zero_tol,
        imag_tol=imag_tol,
        solve_now=True,
    )
    gr = SteklovGradients(
        pb,
        mean_weight=mean_weight,
        rho_transform="auto",
    )

    if objective_mode >= len(pb.sigmas_tilde):
        raise IndexError(
            f"objective_mode={objective_mode} is unavailable; only "
            f"{len(pb.sigmas_tilde)} eigenvalues were retained."
        )

    current_score = float(np.real(pb.sigmas_tilde[objective_mode]))

    saved_history = params.get("history")
    if isinstance(saved_history, SteklovDataHistory):
        history = saved_history
    else:
        history = _loaded_snapshot_history(
            pb,
            tau=tau,
            cs=cs,
            rs=rs,
            objective_mode=objective_mode,
            n_record_nontrivial=n_record_nontrivial,
        )

    returned_iteration = int(
        params.get(
            "returned_iteration",
            history.iterations[-1] if history.iterations else 0,
        )
    )
    best_score = float(params.get("best_score", current_score))
    returned_from_best = bool(params.get("returned_from_best", False))

    result = SteklovDataResult(
        tau=tau,
        ss=ss,
        lp=lp,
        pb=pb,
        gr=gr,
        cs=cs.copy(),
        rs=rs.copy(),
        cos_coeffs=[coeff.copy() for coeff in cos_coeffs],
        sin_coeffs=[coeff.copy() for coeff in sin_coeffs],
        history=history,
        returned_iteration=returned_iteration,
        best_score=best_score,
        returned_from_best=returned_from_best,
    )

    if verbose:
        orders = [
            (cosine.size - 1, sine.size)
            for cosine, sine in zip(cos_coeffs, sin_coeffs)
        ]
        print(f"Loaded and rebuilt Steklov state from {path}")
        print(f"M = {M}")
        print(f"Fourier orders (cos, sin) = {orders}")
        print(f"Ns = {result.ss.Ns}")
        print(f"tau = {result.tau}")
        print(f"density transform = {density_transform}")
        print(result.pb.summary(n=min(n_keep, 10)))

    return result


def load_data(
    load_path: str | Path,
    *,
    n_keep: int = 10,
    objective_mode: int = 1,
    n_record_nontrivial: int = 5,
    positive_rho: bool = True,
    softplus_clip: float | None = None,
    density_transform: str | None = None,
    mean_weight: str = "ds",
    check_disjoint: bool = True,
    margin: float = 1e-5,
    search_radius: int = 3,
    wrap_centers: bool = False,
    zero_tol: float = 1e-10,
    imag_tol: float = 1e-8,
    verbose: bool = True,
) -> SteklovDataResult:
    """Load a parameter file and rebuild ``ss``, ``lp``, ``pb``, and ``gr``.

    Accepts existing files from save_data. Unmarked legacy files require an
    explicit density_transform='softplus' (or a justified alternative). The result preserves the downstream
    attribute interface, but is a SteklovDataResult rather than an optimizer
    class. Parameter-only files produce a single loaded-state history entry.
    """
    path, params = _read_parameter_file(load_path)
    return _rebuild_from_parameters(
        params, source=path, n_keep=n_keep, objective_mode=objective_mode,
        n_record_nontrivial=n_record_nontrivial, positive_rho=positive_rho,
        softplus_clip=softplus_clip, density_transform=density_transform,
        mean_weight=mean_weight,
        check_disjoint=check_disjoint, margin=margin, search_radius=search_radius,
        wrap_centers=wrap_centers, zero_tol=zero_tol, imag_tol=imag_tol,
        verbose=verbose,
    )


def _resolve_save_path(file_name: str | Path) -> Path:
    """Resolve a parameter filename beside this module."""
    name = Path(file_name)
    if name.is_absolute() or name.parent != Path("."):
        raise ValueError(
            "file_name must be a filename, not a path; "
            "the file is saved automatically in the data directory."
        )
    if not name.name:
        raise ValueError("file_name must be nonempty.")
    if not name.suffix:
        name = name.with_suffix(".pkl")
    elif name.suffix.lower() != ".pkl":
        raise ValueError("file_name must have extension '.pkl'.")
    return DATA_DIR / name


def _result_value(res, name: str, default=None):
    if isinstance(res, Mapping):
        return res.get(name, default)
    return getattr(res, name, default)


def _parameters_from_result(res) -> dict[str, Any]:
    """Extract compact parameters by attributes, without importing result classes."""
    ss = _result_value(res, "ss")
    if ss is None:
        ss = _result_value(_result_value(res, "pb"), "ss")
    density_transform = _result_value(ss, "density_transform")
    if density_transform is None:
        raise TypeError(
            "Cannot save a state without ss.density_transform: refusing to "
            "write Fourier coefficients with an ambiguous density convention."
        )

    def required(name):
        value = _result_value(res, name)
        if value is None and ss is not None:
            value = _result_value(ss, name)
        if value is None:
            raise TypeError(f"res must expose {name!r} directly or through ss/pb.ss.")
        return value

    return {
        "density_transform": density_transform,
        "softplus_clip": _result_value(ss, "softplus_clip"),
        "cs": list(np.asarray(required("cs"), dtype=complex).reshape(-1)),
        "rs": np.asarray(required("rs"), dtype=float).reshape(-1).copy(),
        "cos_coeffs": [np.asarray(c, dtype=float).reshape(-1).copy()
                       for c in required("cos_coeffs")],
        "sin_coeffs": [np.asarray(c, dtype=float).reshape(-1).copy()
                       for c in required("sin_coeffs")],
        "tau": complex(required("tau")),
        "Ns": list(np.asarray(required("Ns"), dtype=int).reshape(-1)),
        "returned_iteration": int(_result_value(res, "returned_iteration", 0)),
        "best_score": float(_result_value(res, "best_score", np.nan)),
        "returned_from_best": bool(_result_value(res, "returned_from_best", False)),
    }


def save_data(res, file_name: str | Path) -> Path:
    """Save compact parameters from an SQP, optimization, or rebuilt result.

    Only its parameter attributes are needed; no optimizer or sampling module
    is imported. Bare filenames are saved in this module's directory. The
    .pkl suffix is added when omitted. Returns the absolute saved file path.
    """
    save_path = _resolve_save_path(file_name)
    params = _parameters_from_result(res)
    with save_path.open("wb") as file:
        pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
    resolved_path = save_path.resolve()
    print(f"Saved Steklov parameters to {resolved_path}")
    return resolved_path


__all__ = ["DATA_DIR", "load_data", "save_data"]
