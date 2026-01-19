"""Performance monitoring and scaling cliff detection."""

import time
import warnings
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import numpy as np
import psutil


class PerformanceError(Exception):
    """Raised when performance limits are exceeded."""

    pass


def check_memory_limits(
    matrix_size: Tuple[int, int],
    dtype: np.dtype = np.float64,
    max_memory_gb: float = 32.0,
    safety_factor: float = 0.8,
) -> Dict[str, Any]:
    """Check if matrix operations will exceed memory limits.

    Parameters
    ----------
    matrix_size : tuple
        (rows, cols) of matrix
    dtype : numpy dtype
        Data type of matrix elements
    max_memory_gb : float
        Maximum allowed memory usage in GB
    safety_factor : float
        Safety factor to account for temporary allocations

    Returns
    -------
    memory_report : dict
        Memory usage analysis and recommendations

    Raises
    ------
    PerformanceError
        If operation would exceed memory limits
    """
    rows, cols = matrix_size
    bytes_per_element = np.dtype(dtype).itemsize

    # Estimate memory requirements
    matrix_memory_bytes = rows * cols * bytes_per_element
    matrix_memory_gb = matrix_memory_bytes / (1024**3)

    # Account for temporary allocations (SVD, matrix products, etc.)
    estimated_peak_gb = matrix_memory_gb / safety_factor

    # Get current system memory
    system_memory = psutil.virtual_memory()
    available_gb = system_memory.available / (1024**3)
    total_gb = system_memory.total / (1024**3)

    # Check limits
    exceeds_system = estimated_peak_gb > available_gb
    exceeds_limit = estimated_peak_gb > max_memory_gb

    report = {
        "within_limits": not (exceeds_system or exceeds_limit),
        "matrix_size": matrix_size,
        "dtype": str(dtype),
        "matrix_memory_gb": float(matrix_memory_gb),
        "estimated_peak_gb": float(estimated_peak_gb),
        "available_memory_gb": float(available_gb),
        "total_memory_gb": float(total_gb),
        "max_allowed_gb": max_memory_gb,
        "safety_factor": safety_factor,
        "recommendations": [],
    }

    if exceeds_limit or exceeds_system:
        recommendations = []

        if matrix_memory_gb > 1.0:  # Only for large matrices
            recommendations.extend(
                [
                    "Use sparse matrix representations if possible",
                    "Consider iterative methods instead of direct factorization",
                    "Process data in chunks/batches",
                    "Use lower precision (float32) if acceptable",
                ]
            )

        if exceeds_system:
            recommendations.append(
                f"Insufficient system memory: need {estimated_peak_gb:.1f}GB, "
                f"available {available_gb:.1f}GB"
            )

        report["recommendations"] = recommendations

        raise PerformanceError(
            f"Memory limit exceeded: operation requires {estimated_peak_gb:.1f}GB "
            f"(limit: {max_memory_gb:.1f}GB, available: {available_gb:.1f}GB). "
            f"Matrix size: {rows}×{cols}, dtype: {dtype}"
        )

    return report


def monitor_runtime_complexity(
    func: Callable,
    input_sizes: list,
    max_time_seconds: float = 300.0,
    expected_complexity: str = "O(n²)",
) -> Dict[str, Any]:
    """Monitor runtime complexity and detect scaling issues.

    Parameters
    ----------
    func : callable
        Function to benchmark (should accept size parameter)
    input_sizes : list
        List of input sizes to test
    max_time_seconds : float
        Maximum allowed time for any single test
    expected_complexity : str
        Expected complexity class for validation

    Returns
    -------
    complexity_report : dict
        Runtime complexity analysis

    Raises
    ------
    PerformanceError
        If runtime exceeds limits or shows concerning scaling
    """
    if len(input_sizes) < 3:
        raise ValueError("Need at least 3 input sizes for complexity analysis")

    input_sizes = sorted(input_sizes)
    runtimes = []
    memory_usage = []

    for size in input_sizes:
        # Monitor memory before
        mem_before = psutil.Process().memory_info().rss / (1024**2)  # MB

        start_time = time.time()
        try:
            func(size)
            end_time = time.time()
        except Exception as e:
            raise PerformanceError(f"Function failed at size {size}: {e}")

        runtime = end_time - start_time
        runtimes.append(runtime)

        # Monitor memory after
        mem_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        memory_usage.append(mem_after - mem_before)

        # Check individual runtime limit
        if runtime > max_time_seconds:
            raise PerformanceError(
                f"Runtime limit exceeded at size {size}: {runtime:.1f}s > {max_time_seconds:.1f}s"
            )

    # Analyze scaling behavior
    sizes_array = np.array(input_sizes, dtype=float)
    times_array = np.array(runtimes)

    # Fit power law: t = a * n^b
    log_sizes = np.log(sizes_array)
    log_times = np.log(times_array)

    # Linear regression in log space
    coeffs = np.polyfit(log_sizes, log_times, 1)
    exponent = coeffs[0]

    # Extrapolate to larger sizes
    next_size = input_sizes[-1] * 2
    predicted_time = runtimes[-1] * (2**exponent)

    # Classify complexity
    if exponent < 1.2:
        complexity_class = "O(n) or better"
    elif exponent < 1.8:
        complexity_class = "O(n log n) to O(n^1.5)"
    elif exponent < 2.5:
        complexity_class = "O(n²)"
    elif exponent < 3.5:
        complexity_class = "O(n³)"
    else:
        complexity_class = "O(n⁴) or worse"

    report = {
        "input_sizes": input_sizes,
        "runtimes": runtimes,
        "memory_usage": memory_usage,
        "scaling_exponent": float(exponent),
        "complexity_class": complexity_class,
        "expected_complexity": expected_complexity,
        "predicted_next_runtime": float(predicted_time),
        "next_size": next_size,
        "concerning_scaling": exponent > 3.0 or predicted_time > max_time_seconds,
    }

    # Warnings for concerning behavior
    if exponent > 3.0:
        warnings.warn(
            f"Concerning scaling detected: exponent {exponent:.2f} suggests {complexity_class}. "
            f"Algorithm may not scale to large inputs.",
            stacklevel=2,
        )

    if predicted_time > max_time_seconds:
        warnings.warn(
            f"Predicted runtime at size {next_size} is {predicted_time:.1f}s, "
            f"exceeding {max_time_seconds:.1f}s limit.",
            stacklevel=2,
        )

    return report


def detect_scaling_cliffs(
    n_values: list, complexity_test_func: Callable, cliff_threshold: float = 10.0
) -> Dict[str, Any]:
    """Detect sudden scaling degradation (cliffs) in algorithm performance.

    Parameters
    ----------
    n_values : list
        Input sizes to test
    complexity_test_func : callable
        Function that returns runtime for given input size
    cliff_threshold : float
        Factor increase that indicates a scaling cliff

    Returns
    -------
    cliff_report : dict
        Scaling cliff detection analysis

    Raises
    ------
    PerformanceError
        If scaling cliffs are detected
    """
    if len(n_values) < 3:
        raise ValueError("Need at least 3 sizes for cliff detection")

    n_values = sorted(n_values)
    runtimes = []

    for n in n_values:
        try:
            runtime = complexity_test_func(n)
            runtimes.append(runtime)
        except Exception as e:
            raise PerformanceError(f"Test failed at size {n}: {e}")

    # Detect cliffs (sudden jumps in runtime)
    cliffs_detected = []
    runtime_ratios = []

    for i in range(1, len(runtimes)):
        if runtimes[i - 1] > 0:
            ratio = runtimes[i] / runtimes[i - 1]
            runtime_ratios.append(ratio)

            # Size ratio for normalization
            size_ratio = n_values[i] / n_values[i - 1]
            normalized_ratio = ratio / (size_ratio**2)  # Normalize by O(n²) expectation

            if normalized_ratio > cliff_threshold:
                cliffs_detected.append(
                    {
                        "size_from": n_values[i - 1],
                        "size_to": n_values[i],
                        "runtime_ratio": float(ratio),
                        "normalized_ratio": float(normalized_ratio),
                        "likely_cause": _diagnose_cliff_cause(
                            n_values[i], normalized_ratio
                        ),
                    }
                )

    report = {
        "cliffs_detected": cliffs_detected,
        "n_values": n_values,
        "runtimes": runtimes,
        "runtime_ratios": runtime_ratios,
        "cliff_count": len(cliffs_detected),
        "cliff_threshold": cliff_threshold,
    }

    if cliffs_detected:
        cliff_summary = "; ".join(
            [
                f"size {c['size_to']}: {c['runtime_ratio']:.1f}x slower"
                for c in cliffs_detected
            ]
        )

        raise PerformanceError(
            f"Scaling cliffs detected: {cliff_summary}. "
            f"Algorithm may hit performance bottlenecks at these sizes."
        )

    return report


def monitor_operation_performance(
    operation_name: str, max_time: float = 60.0, max_memory_mb: float = 1024.0
):
    """Decorator to monitor performance of critical operations.

    Parameters
    ----------
    operation_name : str
        Name of operation for logging
    max_time : float
        Maximum allowed time in seconds
    max_memory_mb : float
        Maximum allowed memory increase in MB

    Returns
    -------
    decorator : callable
        Performance monitoring decorator
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**2)

            # Time the operation
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
            except Exception as e:
                runtime = time.time() - start_time
                raise PerformanceError(
                    f"Operation '{operation_name}' failed after {runtime:.2f}s: {e}"
                )

            runtime = end_time - start_time

            # Memory after
            mem_after = process.memory_info().rss / (1024**2)
            mem_increase = mem_after - mem_before

            # Check limits
            if runtime > max_time:
                raise PerformanceError(
                    f"Operation '{operation_name}' exceeded time limit: "
                    f"{runtime:.2f}s > {max_time:.2f}s"
                )

            if mem_increase > max_memory_mb:
                warnings.warn(
                    f"Operation '{operation_name}' used {mem_increase:.1f}MB memory "
                    f"(limit: {max_memory_mb:.1f}MB)",
                    stacklevel=2,
                )

            return result

        return wrapper

    return decorator


def estimate_graph_memory_usage(
    n_nodes: int, k_neighbors: int, dtype: np.dtype = np.float64
) -> Dict[str, Any]:
    """Estimate memory usage for k-NN graph construction.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    k_neighbors : int
        Number of neighbors per node
    dtype : numpy dtype
        Data type for edge weights

    Returns
    -------
    memory_estimate : dict
        Detailed memory usage estimates
    """
    bytes_per_element = np.dtype(dtype).itemsize

    # Sparse adjacency matrix (CSR format)
    n_edges = n_nodes * k_neighbors  # Approximate (may have duplicates)

    # CSR format: data, indices, indptr
    data_bytes = n_edges * bytes_per_element
    indices_bytes = n_edges * 4  # int32 indices
    indptr_bytes = (n_nodes + 1) * 4  # int32 indptr

    sparse_total_bytes = data_bytes + indices_bytes + indptr_bytes
    sparse_total_mb = sparse_total_bytes / (1024**2)

    # Dense matrix comparison
    dense_bytes = n_nodes * n_nodes * bytes_per_element
    dense_mb = dense_bytes / (1024**2)

    # Temporary arrays during construction
    temp_distances_bytes = n_nodes * n_nodes * bytes_per_element  # Distance matrix
    temp_mb = temp_distances_bytes / (1024**2)

    return {
        "n_nodes": n_nodes,
        "k_neighbors": k_neighbors,
        "sparse_memory_mb": float(sparse_total_mb),
        "dense_memory_mb": float(dense_mb),
        "temp_memory_mb": float(temp_mb),
        "memory_savings_ratio": (
            float(dense_mb / sparse_total_mb) if sparse_total_mb > 0 else float("inf")
        ),
        "recommended_sparse": sparse_total_mb
        < dense_mb / 10,  # Use sparse if 10x smaller
    }


def _diagnose_cliff_cause(size: int, normalized_ratio: float) -> str:
    """Diagnose likely cause of performance cliff."""
    if size > 10000 and normalized_ratio > 50:
        return "Likely cache miss or memory bandwidth limit"
    elif size > 1000 and normalized_ratio > 20:
        return "Possible O(n²) → O(n³) complexity change"
    elif normalized_ratio > 100:
        return "Severe algorithmic degradation"
    else:
        return "Moderate performance degradation"
