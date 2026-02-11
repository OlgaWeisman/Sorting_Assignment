"""
Python Project 1 – Comparing Sorting Algorithms (No GitHub)

Folder structure:
Sorting_Assignment/
  main.py
  README.md

Rules:
- Do NOT use sorted() or list.sort() anywhere for sorting the test arrays.
- You MAY use standard libs for timing/statistics/plotting.
"""

from __future__ import annotations

import random
import time
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

# Plotting is allowed (not a sorting library)
import matplotlib.pyplot as plt


# -----------------------------
# Sorting Algorithms (No built-ins)
# -----------------------------

def bubble_sort(arr: List[int]) -> List[int]:
    a = arr[:]  # work on a copy
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def selection_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a


def insertion_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    if len(a) <= 1:
        return a

    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])

    # merge
    i = j = 0
    out: List[int] = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def quick_sort(arr: List[int]) -> List[int]:
    """
    Quicksort implemented without using Python's sort.
    Uses median-of-three pivot and recursion.
    """
    a = arr[:]
    if len(a) <= 1:
        return a

    # median-of-three pivot selection
    first = a[0]
    mid = a[len(a) // 2]
    last = a[-1]
    pivot = sorted_three(first, mid, last)[1]  # middle value without using sorted() on list

    less: List[int] = []
    equal: List[int] = []
    greater: List[int] = []
    for x in a:
        if x < pivot:
            less.append(x)
        elif x > pivot:
            greater.append(x)
        else:
            equal.append(x)

    return quick_sort(less) + equal + quick_sort(greater)


def sorted_three(x: int, y: int, z: int) -> Tuple[int, int, int]:
    """
    Return (min, mid, max) of three values WITHOUT using sorted().
    """
    # Simple comparisons
    if x <= y:
        if y <= z:
            return x, y, z
        # y > z
        if x <= z:
            return x, z, y
        return z, x, y
    # x > y
    if x <= z:
        return y, x, z
    # x > z
    if y <= z:
        return y, z, x
    return z, y, x


# -----------------------------
# Utilities
# -----------------------------

def is_sorted(a: List[int]) -> bool:
    for i in range(1, len(a)):
        if a[i - 1] > a[i]:
            return False
    return True


def make_random_array(n: int, max_value: int = 1_000_000) -> List[int]:
    return [random.randint(0, max_value) for _ in range(n)]


def make_nearly_sorted_array(n: int, max_value: int = 1_000_000, noise_percent: int = 5) -> List[int]:
    """
    Create a sorted array, then add noise by swapping some elements.
    noise_percent is 5 or 20 (or any 0..100).
    """
    a = make_random_array(n, max_value=max_value)
    # We must sort it, but we are NOT allowed to use built-in sort.
    # We'll use merge_sort (our own).
    a = merge_sort(a)

    swaps = int((noise_percent / 100.0) * n)
    # Each swap picks two random indices and swaps them
    for _ in range(swaps):
        i = random.randrange(n)
        j = random.randrange(n)
        a[i], a[j] = a[j], a[i]
    return a


def time_one_run(sort_fn: Callable[[List[int]], List[int]], arr: List[int]) -> float:
    """
    Returns runtime in seconds for a single run.
    """
    start = time.perf_counter()
    out = sort_fn(arr)
    end = time.perf_counter()

    # safety check (optional but useful)
    if not is_sorted(out):
        raise ValueError(f"Sorting failed: {sort_fn.__name__} did not return a sorted array.")
    return end - start


@dataclass
class RunResult:
    n: int
    mean_sec: float
    std_sec: float
    repeats: int
    skipped: bool
    reason: Optional[str] = None


def run_experiment(
    algorithms: Dict[str, Callable[[List[int]], List[int]]],
    sizes: List[int],
    repeats: int,
    experiment_type: str,
    noise_percent: int,
    max_value: int,
    skip_thresholds: Dict[str, int],
) -> Dict[str, List[RunResult]]:
    """
    experiment_type: "random" or "nearly_sorted"
    """
    results: Dict[str, List[RunResult]] = {name: [] for name in algorithms.keys()}

    for n in sizes:
        # generate ONE base array per size, then copy it per algorithm per repeat
        # (fair comparison: everyone sees same data distribution)
        if experiment_type == "random":
            base = make_random_array(n, max_value=max_value)
        else:
            base = make_nearly_sorted_array(n, max_value=max_value, noise_percent=noise_percent)

        for name, fn in algorithms.items():
            # skip extremely slow algorithms for large n
            threshold = skip_thresholds.get(name, 10**18)
            if n > threshold:
                results[name].append(
                    RunResult(n=n, mean_sec=float("nan"), std_sec=float("nan"),
                              repeats=0, skipped=True,
                              reason=f"Skipped because n={n} is above threshold {threshold} for {name}.")
                )
                continue

            times: List[float] = []
            for _ in range(repeats):
                # copy so the algorithm doesn't receive an already-sorted output from previous run
                arr = base[:]
                t = time_one_run(fn, arr)
                times.append(t)

            mean_t = statistics.mean(times)
            std_t = statistics.pstdev(times) if len(times) > 1 else 0.0

            results[name].append(
                RunResult(n=n, mean_sec=mean_t, std_sec=std_t, repeats=repeats, skipped=False)
            )

    return results


def plot_results(results: Dict[str, List[RunResult]], title: str) -> None:
    """
    Plots mean runtime with error bars (std).
    Skipped entries are ignored.
    """
    plt.figure(figsize=(10, 6))

    for alg_name, runs in results.items():
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []

        for r in runs:
            if r.skipped:
                continue
            xs.append(r.n)
            ys.append(r.mean_sec)
            es.append(r.std_sec)

        if xs:
            plt.errorbar(xs, ys, yerr=es, marker='o', capsize=4, label=alg_name)

    plt.xlabel("Array size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_table(results: Dict[str, List[RunResult]]) -> None:
    """
    Prints results in a readable text format.
    """
    print("\nResults (mean ± std, seconds):")
    for alg_name, runs in results.items():
        print(f"\n--- {alg_name} ---")
        for r in runs:
            if r.skipped:
                print(f"n={r.n:<8}  SKIPPED  ({r.reason})")
            else:
                print(f"n={r.n:<8}  {r.mean_sec:.6f} ± {r.std_sec:.6f}   (repeats={r.repeats})")


# -----------------------------
# CLI (Text Interface)
# -----------------------------

ALL_ALGOS: Dict[str, Callable[[List[int]], List[int]]] = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
}

DEFAULT_SKIP_THRESHOLDS: Dict[str, int] = {
    # These are practical safety limits (can be changed).
    # O(n^2) sorts are often too slow above ~20k–50k in Python.
    "Bubble Sort": 20_000,
    "Selection Sort": 20_000,
    "Insertion Sort": 50_000,
    # Faster ones can go large (depending on machine)
    "Merge Sort": 1_000_000,
    "Quick Sort": 1_000_000,
}


def choose_algorithms() -> Dict[str, Callable[[List[int]], List[int]]]:
    print("\nChoose 3 algorithms to compare (enter numbers separated by spaces):")
    names = list(ALL_ALGOS.keys())
    for i, name in enumerate(names, start=1):
        print(f"{i}. {name}")

    while True:
        raw = input("Your choice (e.g., 2 4 5): ").strip()
        parts = raw.split()
        if len(parts) != 3:
            print("Please choose exactly 3 algorithms.")
            continue
        try:
            idxs = [int(p) for p in parts]
        except ValueError:
            print("Please enter numbers only.")
            continue
        if any(i < 1 or i > len(names) for i in idxs):
            print("Invalid choice. Try again.")
            continue
        if len(set(idxs)) != 3:
            print("Please choose 3 DIFFERENT algorithms.")
            continue

        chosen = {names[i - 1]: ALL_ALGOS[names[i - 1]] for i in idxs}
        return chosen


def choose_experiment_type() -> Tuple[str, int]:
    print("\nChoose experiment type:")
    print("1. Random array")
    print("2. Nearly sorted array (add noise)")
    while True:
        raw = input("Your choice (1/2): ").strip()
        if raw == "1":
            return "random", 0
        if raw == "2":
            while True:
                noise_raw = input("Noise level percent (e.g., 5 or 20): ").strip()
                try:
                    noise = int(noise_raw)
                except ValueError:
                    print("Enter an integer (e.g., 5 or 20).")
                    continue
                if not (0 <= noise <= 100):
                    print("Noise must be between 0 and 100.")
                    continue
                return "nearly_sorted", noise
        print("Invalid choice. Try again.")


def choose_sizes() -> List[int]:
    print("\nEnter array sizes separated by spaces (example: 100 500 1000 5000).")
    print("Tip: include a large value up to 1000000 for the fast algorithms.")
    while True:
        raw = input("Sizes: ").strip()
        parts = raw.split()
        try:
            sizes = [int(p) for p in parts]
        except ValueError:
            print("Please enter integers only.")
            continue
        if any(n <= 0 for n in sizes):
            print("All sizes must be positive.")
            continue
        # remove duplicates and keep order
        seen = set()
        unique_sizes = []
        for n in sizes:
            if n not in seen:
                unique_sizes.append(n)
                seen.add(n)
        return unique_sizes


def choose_repeats() -> int:
    while True:
        raw = input("\nHow many repetitions per (algorithm, size)? (e.g., 5): ").strip()
        try:
            r = int(raw)
        except ValueError:
            print("Enter an integer.")
            continue
        if r <= 0:
            print("Repeats must be >= 1.")
            continue
        return r


def main() -> None:
    print("==============================================")
    print("Python Project 1 – Comparing Sorting Algorithms")
    print("==============================================")

    chosen_algos = choose_algorithms()
    exp_type, noise = choose_experiment_type()
    sizes = choose_sizes()
    repeats = choose_repeats()

    max_value = 1_000_000  # values up to one million (requirement)

    print("\nRunning experiment...")
    res = run_experiment(
        algorithms=chosen_algos,
        sizes=sizes,
        repeats=repeats,
        experiment_type=exp_type,
        noise_percent=noise,
        max_value=max_value,
        skip_thresholds=DEFAULT_SKIP_THRESHOLDS,
    )

    print_table(res)

    if exp_type == "random":
        title = "Sorting Runtime Comparison (Random Arrays)"
    else:
        title = f"Sorting Runtime Comparison (Nearly Sorted Arrays, noise={noise}%)"

    plot_results(res, title=title)

    print("\nDone.")


if __name__ == "__main__":
    main()
