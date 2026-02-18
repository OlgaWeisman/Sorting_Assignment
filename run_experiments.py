from __future__ import annotations

import argparse
import random
import time
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


# -----------------------------
# Sorting Algorithms (NO built-ins)
# -----------------------------

def bubble_sort(arr: List[int]) -> List[int]:
    a = arr[:]
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
    i = j = 0
    out: List[int] = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def sorted_three(x: int, y: int, z: int) -> Tuple[int, int, int]:
    if x <= y:
        if y <= z:
            return x, y, z
        if x <= z:
            return x, z, y
        return z, x, y
    if x <= z:
        return y, x, z
    if y <= z:
        return y, z, x
    return z, y, x


def quick_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    if len(a) <= 1:
        return a
    first = a[0]
    mid = a[len(a) // 2]
    last = a[-1]
    pivot = sorted_three(first, mid, last)[1]

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


# -----------------------------
# Helpers
# -----------------------------

def is_sorted(a: List[int]) -> bool:
    return all(a[i - 1] <= a[i] for i in range(1, len(a)))


def make_random_array(n: int, max_value: int = 1_000_000) -> List[int]:
    return [random.randint(0, max_value) for _ in range(n)]


def make_nearly_sorted_array(n: int, max_value: int = 1_000_000, noise_percent: int = 5) -> List[int]:
    a = make_random_array(n, max_value=max_value)
    # Sort using our own algorithm (allowed)
    a = merge_sort(a)

    swaps = int((noise_percent / 100.0) * n)
    for _ in range(swaps):
        i = random.randrange(n)
        j = random.randrange(n)
        a[i], a[j] = a[j], a[i]
    return a


def time_one_run(sort_fn: Callable[[List[int]], List[int]], arr: List[int]) -> float:
    start = time.perf_counter()
    out = sort_fn(arr)
    end = time.perf_counter()
    if not is_sorted(out):
        raise ValueError(f"{sort_fn.__name__} failed to sort correctly.")
    return end - start


@dataclass
class RunResult:
    n: int
    mean_sec: float
    std_sec: float
    repeats: int
    skipped: bool
    reason: Optional[str] = None


# Practical skip thresholds for slow O(n^2) sorts
DEFAULT_SKIP_THRESHOLDS: Dict[str, int] = {
    "Bubble Sort": 20_000,
    "Selection Sort": 20_000,
    "Insertion Sort": 50_000,
    "Merge Sort": 1_000_000,
    "Quick Sort": 1_000_000,
}


def run_experiment(
    algorithms: Dict[str, Callable[[List[int]], List[int]]],
    sizes: List[int],
    repeats: int,
    experiment_type: str,   # "random" or "nearly_sorted"
    noise_percent: int,
    max_value: int,
) -> Dict[str, List[RunResult]]:
    results: Dict[str, List[RunResult]] = {name: [] for name in algorithms.keys()}

    for n in sizes:
        if experiment_type == "random":
            base = make_random_array(n, max_value=max_value)
        else:
            base = make_nearly_sorted_array(n, max_value=max_value, noise_percent=noise_percent)

        for name, fn in algorithms.items():
            threshold = DEFAULT_SKIP_THRESHOLDS.get(name, 10**18)
            if n > threshold:
                results[name].append(
                    RunResult(
                        n=n, mean_sec=float("nan"), std_sec=float("nan"),
                        repeats=0, skipped=True,
                        reason=f"Skipped: n={n} > threshold={threshold}"
                    )
                )
                continue

            times: List[float] = []
            for _ in range(repeats):
                arr = base[:]  # fairness: same base distribution, fresh copy
                times.append(time_one_run(fn, arr))

            mean_t = statistics.mean(times)
            std_t = statistics.pstdev(times) if len(times) > 1 else 0.0

            results[name].append(RunResult(n=n, mean_sec=mean_t, std_sec=std_t, repeats=repeats, skipped=False))

    return results


def plot_results_shaded_std(
    results: Dict[str, List[RunResult]],
    title: str,
    out_file: str
) -> None:
    """
    Plot mean runtime curve with a shaded band of ±1 std (like the attached example).
    """
    plt.figure(figsize=(8, 5))

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

        if not xs:
            continue

        line, = plt.plot(xs, ys, linewidth=2, label=alg_name)
        # shaded std band
        lower = [y - e for y, e in zip(ys, es)]
        upper = [y + e for y, e in zip(ys, es)]
        plt.fill_between(xs, lower, upper, alpha=0.20)

    plt.xlabel("Array size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def print_summary_table(results: Dict[str, List[RunResult]]) -> None:
    print("\nResults (mean ± std seconds):")
    for alg, runs in results.items():
        print(f"\n--- {alg} ---")
        for r in runs:
            if r.skipped:
                print(f"n={r.n:<8}  SKIPPED  ({r.reason})")
            else:
                print(f"n={r.n:<8}  {r.mean_sec:.6f} ± {r.std_sec:.6f}  (repeats={r.repeats})")


# -----------------------------
# CLI
# -----------------------------

ALGO_BY_ID: Dict[int, Tuple[str, Callable[[List[int]], List[int]]]] = {
    1: ("Bubble Sort", bubble_sort),
    2: ("Selection Sort", selection_sort),
    3: ("Insertion Sort", insertion_sort),
    4: ("Merge Sort", merge_sort),
    5: ("Quick Sort", quick_sort),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run sorting runtime experiments and save plots (result1.png, result2.png)."
    )
    p.add_argument("-a", "--algorithms", nargs="+", type=int, required=True,
                   help="Algorithm IDs (choose 3). 1=Bubble, 2=Selection, 3=Insertion, 4=Merge, 5=Quick")
    p.add_argument("-s", "--sizes", nargs="+", type=int, required=True,
                   help="Array sizes, e.g. 100 500 3000")
    p.add_argument("-r", "--repeats", type=int, default=5,
                   help="Repetitions per (algorithm, size), e.g. 20")
    p.add_argument("-e", "--experiment", type=int, choices=[1, 2], default=1,
                   help="Controls noise for result2: 1 -> 5%% noise, 2 -> 20%% noise")
    p.add_argument("--max_value", type=int, default=1_000_000,
                   help="Max integer value in arrays (default: 1,000,000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # validate algorithms
    alg_ids = args.algorithms
    if len(alg_ids) != 3 or len(set(alg_ids)) != 3:
        raise SystemExit("Error: please choose EXACTLY 3 DIFFERENT algorithms using -a (e.g. -a 1 2 5).")
    for i in alg_ids:
        if i not in ALGO_BY_ID:
            raise SystemExit(f"Error: invalid algorithm id {i}. Valid: 1..5")

    algorithms: Dict[str, Callable[[List[int]], List[int]]] = {
        ALGO_BY_ID[i][0]: ALGO_BY_ID[i][1] for i in alg_ids
    }

    sizes = args.sizes
    repeats = args.repeats
    if repeats <= 0:
        raise SystemExit("Error: repeats must be >= 1")

    noise_percent = 5 if args.experiment == 1 else 20

    # -------- Experiment 1: random arrays -> result1.png
    res1 = run_experiment(
        algorithms=algorithms,
        sizes=sizes,
        repeats=repeats,
        experiment_type="random",
        noise_percent=0,
        max_value=args.max_value,
    )
    print_summary_table(res1)
    plot_results_shaded_std(
        res1,
        title="Runtime Comparison (Random Arrays)",
        out_file="result1.png"
    )
    print("\nSaved plot: result1.png")

    # -------- Experiment 2: nearly sorted arrays -> result2.png
    res2 = run_experiment(
        algorithms=algorithms,
        sizes=sizes,
        repeats=repeats,
        experiment_type="nearly_sorted",
        noise_percent=noise_percent,
        max_value=args.max_value,
    )
    print_summary_table(res2)
    plot_results_shaded_std(
        res2,
        title=f"Runtime Comparison (Nearly Sorted, noise={noise_percent}%)",
        out_file="result2.png"
    )
    print("\nSaved plot: result2.png")


if __name__ == "__main__":
    main()