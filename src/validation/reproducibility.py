"""Reproducibility framework for experiment tracking and validation."""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ReproducibilityError(Exception):
    """Raised when reproducibility requirements are violated."""

    pass


def ensure_reproducibility(
    seed: int, libraries: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Ensure reproducible computation across relevant libraries.

    Parameters
    ----------
    seed : int
        Master random seed
    libraries : list or None
        Libraries to seed (default: ["numpy", "random"])

    Returns
    -------
    seed_report : dict
        Seeding status and environment information

    Raises
    ------
    ReproducibilityError
        If seeding fails for critical libraries
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")

    if libraries is None:
        libraries = ["numpy", "random"]

    seed_status = {}
    failed_libraries = []

    # Set numpy seed
    if "numpy" in libraries:
        try:
            np.random.seed(seed)
            # Also set the new Generator-based random state
            np.random.default_rng(seed)
            seed_status["numpy"] = "success"
        except Exception as e:
            seed_status["numpy"] = f"failed: {e}"
            failed_libraries.append("numpy")

    # Set Python's random seed
    if "random" in libraries:
        try:
            import random

            random.seed(seed)
            seed_status["random"] = "success"
        except Exception as e:
            seed_status["random"] = f"failed: {e}"
            failed_libraries.append("random")

    # Optional: PyTorch (if available)
    if "torch" in libraries:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            seed_status["torch"] = "success"
        except ImportError:
            seed_status["torch"] = "not_available"
        except Exception as e:
            seed_status["torch"] = f"failed: {e}"
            failed_libraries.append("torch")

    # Environment information
    environment = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "timestamp": time.time(),
        "hostname": platform.node(),
    }

    report = {
        "seed": seed,
        "seed_status": seed_status,
        "failed_libraries": failed_libraries,
        "environment": environment,
        "reproducible": len(failed_libraries) == 0,
    }

    if failed_libraries:
        raise ReproducibilityError(
            f"Failed to seed critical libraries: {failed_libraries}. "
            f"Reproducibility cannot be guaranteed."
        )

    return report


def compute_data_hash(
    data: Union[np.ndarray, Dict[str, np.ndarray]], algorithm: str = "sha256"
) -> str:
    """Compute cryptographic hash of data for integrity checking.

    Parameters
    ----------
    data : ndarray or dict of ndarrays
        Data to hash
    algorithm : str
        Hash algorithm ("sha256", "md5", "sha1")

    Returns
    -------
    data_hash : str
        Hexadecimal hash string

    Notes
    -----
    Hashes are sensitive to data type, shape, and byte order.
    Use for detecting data corruption or unintended modifications.
    """
    hasher = hashlib.new(algorithm)

    if isinstance(data, dict):
        # Hash dictionary of arrays
        for key in sorted(data.keys()):  # Ensure deterministic order
            hasher.update(key.encode("utf-8"))
            arr = np.asarray(data[key])
            hasher.update(arr.tobytes())
    else:
        # Hash single array
        arr = np.asarray(data)
        hasher.update(arr.tobytes())

    return hasher.hexdigest()


def log_experiment_config(
    config: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    include_environment: bool = True,
) -> str:
    """Log experiment configuration for reproducibility.

    Parameters
    ----------
    config : dict
        Experiment configuration parameters
    output_path : str, Path, or None
        Where to save config (if None, return JSON string only)
    include_environment : bool
        Include environment information

    Returns
    -------
    config_json : str
        JSON representation of configuration
    """
    # Create comprehensive config
    full_config = config.copy()

    if include_environment:
        env_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "timestamp": time.time(),
            "hostname": platform.node(),
            "working_directory": str(Path.cwd()),
            "command_line_args": sys.argv,
        }

        # Add library versions
        try:
            import scipy

            env_info["scipy_version"] = scipy.__version__
        except ImportError:
            pass

        full_config["environment"] = env_info

    # Convert config to JSON (handle numpy types)
    config_json = json.dumps(full_config, indent=2, default=_json_numpy_serializer)

    # Save to file if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(config_json)

    return config_json


def validate_experiment_reproducibility(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
    ignore_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate that two experiment configurations are equivalent for reproducibility.

    Parameters
    ----------
    config1, config2 : dict
        Experiment configurations to compare
    ignore_keys : list or None
        Keys to ignore in comparison (e.g., "timestamp", "hostname")

    Returns
    -------
    validation_report : dict
        Reproducibility validation results
    """
    if ignore_keys is None:
        ignore_keys = [
            "timestamp",
            "hostname",
            "working_directory",
            "command_line_args",
        ]

    # Deep copy to avoid modifying originals
    c1 = json.loads(json.dumps(config1, default=_json_numpy_serializer))
    c2 = json.loads(json.dumps(config2, default=_json_numpy_serializer))

    # Remove ignored keys recursively
    _remove_keys_recursive(c1, ignore_keys)
    _remove_keys_recursive(c2, ignore_keys)

    # Compare configurations
    differences = _find_config_differences(c1, c2)
    reproducible = len(differences) == 0

    report = {
        "reproducible": reproducible,
        "differences": differences,
        "ignored_keys": ignore_keys,
        "config1_keys": list(c1.keys()),
        "config2_keys": list(c2.keys()),
    }

    return report


def create_experiment_fingerprint(
    data_hash: str, config_hash: str, code_version: Optional[str] = None
) -> str:
    """Create unique fingerprint for experiment reproducibility.

    Parameters
    ----------
    data_hash : str
        Hash of input data
    config_hash : str
        Hash of configuration
    code_version : str or None
        Code version identifier (git commit, version tag, etc.)

    Returns
    -------
    fingerprint : str
        Unique experiment fingerprint
    """
    hasher = hashlib.sha256()

    hasher.update(data_hash.encode("utf-8"))
    hasher.update(config_hash.encode("utf-8"))

    if code_version is not None:
        hasher.update(code_version.encode("utf-8"))

    # Add timestamp truncated to hour for partial time-independence
    hour_timestamp = int(time.time() // 3600) * 3600
    hasher.update(str(hour_timestamp).encode("utf-8"))

    return hasher.hexdigest()[:16]  # Short fingerprint


def verify_data_integrity(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    expected_hash: str,
    algorithm: str = "sha256",
) -> Dict[str, Any]:
    """Verify data integrity against expected hash.

    Parameters
    ----------
    data : ndarray or dict of ndarrays
        Data to verify
    expected_hash : str
        Expected hash value
    algorithm : str
        Hash algorithm used

    Returns
    -------
    verification_report : dict
        Data integrity verification results

    Raises
    ------
    ReproducibilityError
        If data integrity check fails
    """
    computed_hash = compute_data_hash(data, algorithm)
    integrity_valid = computed_hash == expected_hash

    report = {
        "integrity_valid": integrity_valid,
        "expected_hash": expected_hash,
        "computed_hash": computed_hash,
        "algorithm": algorithm,
    }

    if not integrity_valid:
        raise ReproducibilityError(
            f"Data integrity check failed. "
            f"Expected: {expected_hash}, Computed: {computed_hash}. "
            f"Data may have been corrupted or modified."
        )

    return report


def seed_all_rngs(seed: int) -> Dict[str, Any]:
    """Comprehensive seeding of all relevant random number generators.

    Parameters
    ----------
    seed : int
        Master seed value

    Returns
    -------
    seeding_report : dict
        Complete seeding status report
    """
    # System-level reproducibility
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Core library seeding
    libraries = ["numpy", "random"]

    # Optional libraries
    try:
        import torch

        libraries.append("torch")
    except ImportError:
        pass

    return ensure_reproducibility(seed, libraries)


# Helper functions


def _json_numpy_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _remove_keys_recursive(d: Dict[str, Any], keys_to_remove: List[str]) -> None:
    """Recursively remove keys from nested dictionary."""
    for key in keys_to_remove:
        d.pop(key, None)

    for value in d.values():
        if isinstance(value, dict):
            _remove_keys_recursive(value, keys_to_remove)


def _find_config_differences(
    config1: Dict[str, Any], config2: Dict[str, Any], path: str = ""
) -> List[str]:
    """Find differences between two configuration dictionaries."""
    differences = []

    # Check keys in config1
    for key in config1:
        current_path = f"{path}.{key}" if path else key

        if key not in config2:
            differences.append(f"Key missing in config2: {current_path}")
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recursively check nested dictionaries
            differences.extend(
                _find_config_differences(config1[key], config2[key], current_path)
            )
        elif config1[key] != config2[key]:
            differences.append(
                f"Value mismatch at {current_path}: "
                f"{config1[key]} != {config2[key]}"
            )

    # Check for keys only in config2
    for key in config2:
        current_path = f"{path}.{key}" if path else key
        if key not in config1:
            differences.append(f"Extra key in config2: {current_path}")

    return differences
