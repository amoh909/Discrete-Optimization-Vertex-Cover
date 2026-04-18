import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

from src.graph_generation import (
    generate_random_graph,
    generate_grid_graph,
    generate_bipartite_graph,
    generate_cycle_graph,
    generate_complete_graph,
    generate_near_clique,
)
from src.lp_solver import solve_vertex_cover_lp
from src.ilp_solver import solve_vertex_cover_ilp
from src.rounding import threshold_rounding
from src.greedy import greedy_vertex_cover


def is_vertex_cover(G: nx.Graph, cover) -> bool:
    cover_set = set(cover)
    return all(u in cover_set or v in cover_set for u, v in G.edges())


def cover_size(cover) -> int:
    return len(set(cover))


# Thin timing wrappers
def _timed_lp(G: nx.Graph):
    t0 = time.perf_counter()
    result = solve_vertex_cover_lp(G)
    return result, time.perf_counter() - t0


def _timed_ilp(G: nx.Graph):
    t0 = time.perf_counter()
    result = solve_vertex_cover_ilp(G)
    return result, time.perf_counter() - t0


def _timed_greedy(G: nx.Graph):
    t0 = time.perf_counter()
    cover = greedy_vertex_cover(G)
    return cover, time.perf_counter() - t0

# Graph-density helper
def graph_density(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    return 2.0 * G.number_of_edges() / (n * (n - 1))


_FRAC_EPS = 1e-6  


def lp_num_fractional(x_values: dict) -> int:
    """Count vertices with strictly fractional LP value (not 0 or 1)."""
    return sum(
        1 for v in x_values.values()
        if _FRAC_EPS < v < 1.0 - _FRAC_EPS
    )


def lp_num_at_half(x_values: dict) -> int:
    """Count vertices whose LP value is exactly 1/2 (within tolerance)."""
    return sum(1 for v in x_values.values() if abs(v - 0.5) < _FRAC_EPS)


def lp_is_integral(x_values: dict) -> bool:
    """True iff the LP solution is already 0/1 integral."""
    return lp_num_fractional(x_values) == 0



# Experiment suite definition
def build_experiment_suite() -> List[Dict]:
    """
    Returns a list of graph specs.  Each spec carries:
        family      : str
        name        : str   - unique instance identifier
        parity      : str | None  - "odd"/"even" for cycles
        generator   : callable
        kwargs      : dict  - passed to the generator
        gen_params  : dict  - explicit reproducibility metadata [FIX #7]
    """
    specs: List[Dict] = []


    # 1. Random graphs G(n, p)  -  sparse, medium, dense
    for n in [10, 16, 24, 32]:
        for p in [0.15, 0.30, 0.50, 0.80, 0.90]:
            for trial in range(3):
                seed = 1000 + 100 * n + 10 * trial + int(100 * p)
                specs.append({
                    "family":     "random",
                    "name":       f"random_n{n}_p{p}_t{trial}",
                    "parity":     None,
                    "generator":  generate_random_graph,
                    "kwargs":     {"n": n, "p": p, "seed": seed},
                    "gen_params": {"n": n, "p": p, "seed": seed, "trial": trial},
                })


    # 2. Grid graphs
    for rows, cols in [(3, 3), (4, 4), (5, 5), (4, 6)]:
        specs.append({
            "family":     "grid",
            "name":       f"grid_{rows}x{cols}",
            "parity":     None,
            "generator":  generate_grid_graph,
            "kwargs":     {"rows": rows, "cols": cols},
            "gen_params": {"rows": rows, "cols": cols},
        })

    # 3. Bipartite graphs
    for n_left, n_right, p in [(6, 6, 0.3), (8, 8, 0.4), (10, 12, 0.3), (12, 12, 0.5)]:
        for trial in range(2):
            seed = 2000 + 50 * trial + n_left + n_right
            specs.append({
                "family":     "bipartite",
                "name":       f"bipartite_{n_left}x{n_right}_p{p}_t{trial}",
                "parity":     None,
                "generator":  generate_bipartite_graph,
                "kwargs":     {"n": n_left, "m": n_right, "p": p, "seed": seed},
                "gen_params": {"n_left": n_left, "n_right": n_right,
                               "p": p, "seed": seed, "trial": trial},
            })


    #Cycle graphs - odd and even
    for n in [5, 6, 7, 8, 9, 10, 11, 15]:
        parity = "odd" if n % 2 == 1 else "even"
        specs.append({
            "family":     "cycle",
            "name":       f"cycle_n{n}_{parity}",
            "parity":     parity,
            "generator":  generate_cycle_graph,
            "kwargs":     {"n": n},
            "gen_params": {"n": n, "parity": parity},
        })


    #Complete graphs K_n
    for n in [4, 6, 8, 10, 12, 16, 20, 24]:
        specs.append({
            "family":     "complete",
            "name":       f"complete_n{n}",
            "parity":     None,
            "generator":  generate_complete_graph,
            "kwargs":     {"n": n},
            "gen_params": {"n": n},
        })


    #Near-cliques
    for n, missing_frac, trial in [
        (12, 0.10, 0), (16, 0.15, 0), (20, 0.20, 0), (20, 0.10, 1)
    ]:
        seed = 3000 + trial + n
        specs.append({
            "family":     "near_clique",
            "name":       f"near_clique_n{n}_miss{missing_frac}_t{trial}",
            "parity":     None,
            "generator":  generate_near_clique,
            "kwargs":     {"n": n, "remove_fraction": missing_frac, "seed": seed},
            "gen_params": {"n": n, "remove_fraction": missing_frac,
                           "seed": seed, "trial": trial},
        })

    return specs


# Result container
@dataclass
class ExperimentRow:
    #identity 
    family:                str
    instance_name:         str
    parity:                Optional[str]

    # Stored flat; fields absent for a given family are None.
    param_n:               Optional[int]
    param_p:               Optional[float]
    param_seed:            Optional[int]
    param_trial:           Optional[int]
    param_rows:            Optional[int]
    param_cols:            Optional[int]
    param_n_left:          Optional[int]
    param_n_right:         Optional[int]
    param_remove_fraction: Optional[float]

    # graph structure
    n:                     int
    m:                     int
    density:               float

    #LP relaxation 
    lp_value:              float
    lp_runtime_sec:        float

    #LP solution structure 
    lp_num_fractional:     int   
    lp_num_at_half:        int   
    lp_is_integral:        bool  

    #threshold rounding 
    rounded_size:          int
    rounded_feasible:      bool
    rounded_vs_lp_ratio:   float  
    rounding_runtime_sec:  float

    # greedy 
    greedy_size:           int
    greedy_feasible:       bool   
    greedy_runtime_sec:    float
    greedy_vs_lp_ratio:    float

    # exact ILP (small graphs only) 
    ilp_value:             Optional[float]
    ilp_runtime_sec:       Optional[float]
    rounded_vs_ilp_ratio:  Optional[float]
    greedy_vs_ilp_ratio:   Optional[float]
    integrality_gap:       Optional[float] 


def _gp(gen_params: Dict, key: str):
    """Safely extract a key from gen_params, returning None if absent."""
    return gen_params.get(key, None)



# Single-instance experiment
ILP_MAX_N = 28


def run_single_experiment(
    G:             nx.Graph,
    family:        str,
    instance_name: str,
    parity:        Optional[str],
    gen_params:    Dict,
    ilp_max_n:     int = ILP_MAX_N,
) -> ExperimentRow:
    n = G.number_of_nodes()

    #LP relaxation 
    lp_result, lp_runtime = _timed_lp(G)
    if lp_result["status"] != "Optimal":
        raise RuntimeError(
            f"LP failed on {instance_name}: status={lp_result['status']}"
        )

    lp_value = float(lp_result["objective"])
    x_values = lp_result["x_values"]

    #LP solution structure 
    num_frac    = lp_num_fractional(x_values)
    num_at_half = lp_num_at_half(x_values)
    is_intgl    = lp_is_integral(x_values)

    #Threshold rounding 
    t0 = time.perf_counter()
    rounded_cover = threshold_rounding(x_values)
    rounding_runtime = time.perf_counter() - t0
    rounded_feasible = is_vertex_cover(G, rounded_cover)
    rounded_size     = cover_size(rounded_cover)
    rounded_vs_lp    = (rounded_size / lp_value) if lp_value > 0 else float("nan")

    #Greedy heuristic
    greedy_cover, greedy_runtime = _timed_greedy(G)
    greedy_feasible = is_vertex_cover(G, greedy_cover)  
    greedy_size     = cover_size(greedy_cover)
    greedy_vs_lp    = (greedy_size / lp_value) if lp_value > 0 else float("nan")

    #Exact ILP (small graphs only)
    ilp_value            = None
    ilp_runtime          = None
    rounded_vs_ilp_ratio = None
    greedy_vs_ilp_ratio  = None
    integrality_gap      = None

    if n <= ilp_max_n:
        ilp_result, ilp_t = _timed_ilp(G)
        if ilp_result["status"] != "Optimal":
            raise RuntimeError(
                f"ILP failed on {instance_name}: status={ilp_result['status']}"
            )

        ilp_value = float(ilp_result["objective"])
        ilp_runtime = float(ilp_t)

        if ilp_value > 0:
            rounded_vs_ilp_ratio = rounded_size / ilp_value
            greedy_vs_ilp_ratio = greedy_size / ilp_value
            integrality_gap = ilp_value / lp_value if lp_value > 0 else float("nan")
            
    return ExperimentRow(
        family                = family,
        instance_name         = instance_name,
        parity                = parity,
        # reproducibility 
        param_n               = _gp(gen_params, "n"),
        param_p               = _gp(gen_params, "p"),
        param_seed            = _gp(gen_params, "seed"),
        param_trial           = _gp(gen_params, "trial"),
        param_rows            = _gp(gen_params, "rows"),
        param_cols            = _gp(gen_params, "cols"),
        param_n_left          = _gp(gen_params, "n_left"),
        param_n_right         = _gp(gen_params, "n_right"),
        param_remove_fraction = _gp(gen_params, "remove_fraction"),
        # graph structure
        n                     = n,
        m                     = G.number_of_edges(),
        density               = graph_density(G),
        # LP
        lp_value              = lp_value,
        lp_runtime_sec        = lp_runtime,
        # LP structure
        lp_num_fractional     = num_frac,
        lp_num_at_half        = num_at_half,
        lp_is_integral        = bool(is_intgl),
        # rounding
        rounded_size          = rounded_size,
        rounded_feasible      = bool(rounded_feasible),
        rounding_runtime_sec = rounding_runtime,
        rounded_vs_lp_ratio   = rounded_vs_lp,
        # greedy
        greedy_size           = greedy_size,
        greedy_feasible       = bool(greedy_feasible),
        greedy_runtime_sec    = greedy_runtime,
        greedy_vs_lp_ratio    = greedy_vs_lp,
        # ILP
        ilp_value             = ilp_value,
        ilp_runtime_sec       = ilp_runtime,
        rounded_vs_ilp_ratio  = rounded_vs_ilp_ratio,
        greedy_vs_ilp_ratio   = greedy_vs_ilp_ratio,
        integrality_gap       = integrality_gap,
    )



# Batch runner
def run_all_experiments(
    output_dir: str = "results",
    ilp_max_n:  int = ILP_MAX_N,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for spec in build_experiment_suite():
        G   = spec["generator"](**spec["kwargs"])
        row = run_single_experiment(
            G             = G,
            family        = spec["family"],
            instance_name = spec["name"],
            parity        = spec["parity"],
            gen_params    = spec["gen_params"],
            ilp_max_n     = ilp_max_n,
        )
        rows.append(asdict(row))

        r_tag = "✓" if row.rounded_feasible else "✗ RND-INFEASIBLE"
        g_tag = "" if row.greedy_feasible   else "  ✗ GRD-INFEASIBLE"
        print(
            f"[{r_tag}{g_tag}] {spec['name']:55s}"
            f"  LP={row.lp_value:.2f}"
            f"  rnd={row.rounded_size}"
            f"  grd={row.greedy_size}"
            + (f"  ILP={row.ilp_value:.1f}  gap={row.integrality_gap:.4f}"
               if row.ilp_value is not None else "")
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["family", "n", "m", "instance_name"]).reset_index(drop=True)
    df.to_csv(output_path / "experiment_results.csv", index=False)


    agg_cols = {
        "instances":                ("instance_name",        "count"),
        "avg_n":                    ("n",                    "mean"),
        "avg_m":                    ("m",                    "mean"),
        "avg_lp_value":             ("lp_value",             "mean"),
        "avg_rounded_size":         ("rounded_size",         "mean"),
        "avg_greedy_size":          ("greedy_size",          "mean"),
        "avg_ilp_value":            ("ilp_value",            "mean"),
        "avg_integrality_gap":      ("integrality_gap",      "mean"),
        "max_integrality_gap":      ("integrality_gap",      "max"),
        "avg_rounded_vs_ilp_ratio": ("rounded_vs_ilp_ratio", "mean"),
        "max_rounded_vs_ilp_ratio": ("rounded_vs_ilp_ratio", "max"),
        "avg_greedy_vs_ilp_ratio":  ("greedy_vs_ilp_ratio",  "mean"),
        "avg_rounded_vs_lp_ratio":  ("rounded_vs_lp_ratio",  "mean"),
        "max_rounded_vs_lp_ratio":  ("rounded_vs_lp_ratio",  "max"),
        "avg_lp_runtime_sec":       ("lp_runtime_sec",       "mean"),
        "avg_greedy_runtime_sec":   ("greedy_runtime_sec",   "mean"),
        "avg_ilp_runtime_sec":      ("ilp_runtime_sec",      "mean"),
        "pct_lp_integral":          ("lp_is_integral",       "mean"),  
    }
    summary_fam = (
        df.groupby("family", dropna=False)
          .agg(**agg_cols)
          .reset_index()
    )
    summary_fam.to_csv(output_path / "summary_by_family.csv", index=False)


    summary_size = (
        df.groupby(["family", "n"], dropna=False)
          .agg(
              instances           = ("instance_name",        "count"),
              avg_m               = ("m",                    "mean"),
              avg_lp_value        = ("lp_value",             "mean"),
              avg_rounded_size    = ("rounded_size",         "mean"),
              avg_greedy_size     = ("greedy_size",          "mean"),
              avg_ilp_value       = ("ilp_value",            "mean"),
              avg_integrality_gap = ("integrality_gap",      "mean"),
              max_integrality_gap = ("integrality_gap",      "max"),
              avg_rounded_vs_ilp  = ("rounded_vs_ilp_ratio", "mean"),
              avg_rounded_vs_lp   = ("rounded_vs_lp_ratio",  "mean"),
              max_rounded_vs_lp   = ("rounded_vs_lp_ratio",  "max"),
              pct_lp_integral     = ("lp_is_integral",       "mean"),  
          )
          .reset_index()
    )
    summary_size.to_csv(output_path / "summary_by_family_and_size.csv", index=False)

    return df



# Theory validation
def validate_theory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows that violate any of the theoretical guarantees.

    Checks:
      1. rounded_feasible == True      (rounding always gives a valid cover)
      2. greedy_feasible  == True      (greedy  always gives a valid cover) [FIX #1]
      3. rounded_vs_lp_ratio  <= 2     (formal 2-approximation proof)
      4. rounded_vs_ilp_ratio <= 2     (empirical confirmation against ILP)

    Any row returned here signals a bug that needs investigation.
    """
    c = df.copy()
    c["rnd_feasible_ok"] = c["rounded_feasible"]
    c["grd_feasible_ok"] = c["greedy_feasible"]                           
    c["lp_ratio_ok"]     = (c["rounded_vs_lp_ratio"].isna()  |
                            (c["rounded_vs_lp_ratio"]  <= 2.0 + 1e-9))
    c["ilp_ratio_ok"]    = (c["rounded_vs_ilp_ratio"].isna() |
                            (c["rounded_vs_ilp_ratio"] <= 2.0 + 1e-9))
    all_ok = (c["rnd_feasible_ok"] & c["grd_feasible_ok"] &
              c["lp_ratio_ok"]     & c["ilp_ratio_ok"])
    return c.loc[~all_ok].copy()


# Worst-case extractor 
def worst_cases(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Returns the top-k worst instances across three axes:
      - highest integrality gap
      - highest rounded / ILP ratio
      - highest greedy  / ILP ratio

    These give you concrete named instances to cite in the report
    instead of just averages.
    """
    with_ilp = df[df["ilp_value"].notna()].copy()
    if with_ilp.empty:
        return pd.DataFrame()

    cols = ["family", "instance_name", "n", "m", "density",
            "lp_value", "ilp_value", "integrality_gap",
            "rounded_size", "rounded_vs_ilp_ratio",
            "greedy_size",  "greedy_vs_ilp_ratio"]

    frames = []
    for criterion in ["integrality_gap", "rounded_vs_ilp_ratio", "greedy_vs_ilp_ratio"]:
        subset = (
            with_ilp.nlargest(top_k, criterion)[cols]
                    .assign(worst_by=criterion)
        )
        frames.append(subset)

    return pd.concat(frames, ignore_index=True)



# Report printer

def print_report_notes(df: pd.DataFrame, output_dir: str = "results") -> None:
    with_ilp = df[df["ilp_value"].notna()].copy()
    sep = "=" * 65
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


    # Theory validation
    print(f"\n{sep}")
    print("THEORY VALIDATION")
    print(sep)

    rnd_rate = df["rounded_feasible"].mean()
    grd_rate = df["greedy_feasible"].mean()
    print(f"  Threshold rounding feasibility rate : {rnd_rate:.3f}  (must be 1.000)")
    print(f"  Greedy feasibility rate             : {grd_rate:.3f}  (must be 1.000)")

    max_lp_ratio = df["rounded_vs_lp_ratio"].max()
    print(f"  Max rounded / LP_value  ratio       : {max_lp_ratio:.4f}  (must be <= 2.000)")

    if not with_ilp.empty:
        max_ilp_ratio = with_ilp["rounded_vs_ilp_ratio"].max()
        print(f"  Max rounded / ILP_value ratio       : {max_ilp_ratio:.4f}  (must be <= 2.000)")

    violations = validate_theory(df)
    if violations.empty:
        print("\n  No violations found across all instances.")
    else:
        print(f"\n  {len(violations)} VIOLATION(S) FOUND:")
        print(violations[[
            "family", "instance_name",
            "rounded_feasible", "greedy_feasible",
            "rounded_vs_lp_ratio", "rounded_vs_ilp_ratio",
        ]].to_string(index=False))

  
    #Overall averages
    print(f"\n{sep}")
    print("OVERALL AVERAGES  (instances with exact ILP available)")
    print(sep)
    if not with_ilp.empty:
        print(f"  Avg integrality gap  (ILP/LP) : {with_ilp['integrality_gap'].mean():.4f}")
        print(f"  Max integrality gap  (ILP/LP) : {with_ilp['integrality_gap'].max():.4f}")
        print(f"  Avg rounded / ILP             : {with_ilp['rounded_vs_ilp_ratio'].mean():.4f}")
        print(f"  Avg greedy  / ILP             : {with_ilp['greedy_vs_ilp_ratio'].mean():.4f}")


    # Complete graphs - gap approaching 2
    print(f"\n{sep}")
    print("COMPLETE GRAPHS  -  integrality gap approaches 2 as n -> inf")
    print(sep)
    complete = with_ilp[with_ilp["family"] == "complete"].sort_values("n")
    if complete.empty:
        print("  (no complete graph ILP results within ilp_max_n)")
    else:
        print(f"  {'n':>4}  {'ILP':>6}  {'LP':>6}  {'gap (ILP/LP)':>14}  {'theory 2-2/n':>14}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*14}")
        for _, row in complete.iterrows():
            ni  = int(row["n"])
            gap = row["integrality_gap"]
            th  = 2 - 2 / ni
            print(f"  {ni:>4}  {row['ilp_value']:>6.1f}  {row['lp_value']:>6.1f}"
                  f"  {gap:>14.4f}  {th:>14.4f}")


    # Bipartite - LP tight (gap = 1)
    print(f"\n{sep}")
    print("BIPARTITE GRAPHS  -  LP is tight (integrality gap should = 1)")
    print(sep)
    bip = with_ilp[with_ilp["family"] == "bipartite"].sort_values("n")
    if bip.empty:
        print("  (no bipartite ILP results)")
    else:
        all_tight = (bip["integrality_gap"] - 1.0).abs().max() < 1e-6
        print(f"  All gaps = 1?  {'Yes' if all_tight else 'No - check instances below'}")
        print(f"\n  {'instance':45s}  {'ILP':>6}  {'LP':>6}  {'gap':>6}")
        print(f"  {'-'*45}  {'-'*6}  {'-'*6}  {'-'*6}")
        for _, row in bip.iterrows():
            print(f"  {row['instance_name']:45s}  {row['ilp_value']:>6.1f}"
                  f"  {row['lp_value']:>6.1f}  {row['integrality_gap']:>6.4f}")


    # Cycles - LP variable structure
    print(f"\n{sep}")
    print("CYCLES  -  LP solution structure (odd vs even)")
    print(sep)
    cycles = df[df["family"] == "cycle"].sort_values("n")
    if cycles.empty:
        print("  (no cycle results)")
    else:
        print(f"  {'instance':28s}  {'par':4}  {'LP':>5}  "
              f"{'#frac':>5}  {'#=0.5':>5}  {'intgl?':>6}  "
              f"{'rnd':>4}  {'ILP':>5}  {'gap':>6}")
        print(f"  {'-'*28}  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*5}"
              f"  {'-'*6}  {'-'*4}  {'-'*5}  {'-'*6}")
        for _, row in cycles.iterrows():
            gap_s = f"{row['integrality_gap']:.4f}" if pd.notna(row["integrality_gap"]) else "   N/A"
            ilp_s = f"{row['ilp_value']:.1f}"       if pd.notna(row["ilp_value"])       else "  N/A"
            intgl = "yes" if row["lp_is_integral"] else "NO"
            print(f"  {row['instance_name']:28s}  {str(row['parity']):4}  "
                  f"{row['lp_value']:>5.1f}  "
                  f"{row['lp_num_fractional']:>5}  {row['lp_num_at_half']:>5}  "
                  f"{intgl:>6}  "
                  f"{row['rounded_size']:>4}  {ilp_s:>5}  {gap_s:>6}")

        print("\n  Expected: odd cycles usually give half-integral LP optima")
        print("            even cycles often have an integral LP optimum")


    # Random graphs by p
    print(f"\n{sep}")
    print("RANDOM GRAPHS  -  approximation ratio by edge probability p")
    print(sep)
    rnd = with_ilp[with_ilp["family"] == "random"].copy()
    if rnd.empty:
        print("  (no random graph ILP results)")
    else:
        summary = (
            rnd.groupby("param_p")
               .agg(
                   avg_gap            = ("integrality_gap",      "mean"),
                   avg_rounded_vs_ilp = ("rounded_vs_ilp_ratio", "mean"),
                   avg_greedy_vs_ilp  = ("greedy_vs_ilp_ratio",  "mean"),
               )
               .reset_index()
               .sort_values("param_p")
        )
        print(f"  {'p':>6}  {'avg gap':>10}  {'avg rnd/ILP':>12}  {'avg grd/ILP':>12}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}")
        for _, row in summary.iterrows():
            print(f"  {row['param_p']:>6.2f}  {row['avg_gap']:>10.4f}"
                  f"  {row['avg_rounded_vs_ilp']:>12.4f}"
                  f"  {row['avg_greedy_vs_ilp']:>12.4f}")

    #Greedy vs rounding comparison
    print(f"\n{sep}")
    print("GREEDY VS ROUNDING  -  which performs better?")
    print(sep)
    cmp_df = with_ilp.copy()
    if cmp_df.empty:
        print("  (no ILP results available for comparison)")
    else:
        cmp_df["winner"] = np.where(
            cmp_df["greedy_vs_ilp_ratio"] < cmp_df["rounded_vs_ilp_ratio"], "greedy",
            np.where(
                cmp_df["greedy_vs_ilp_ratio"] > cmp_df["rounded_vs_ilp_ratio"], "rounding",
                "tie"
            )
        )
        counts = cmp_df["winner"].value_counts()
        print(f"  Greedy better   : {int(counts.get('greedy', 0))}")
        print(f"  Rounding better : {int(counts.get('rounding', 0))}")
        print(f"  Ties            : {int(counts.get('tie', 0))}")

        by_family = (
            cmp_df.groupby(["family", "winner"])
                  .size()
                  .unstack(fill_value=0)
                  .reset_index()
        )
        print("\n  By family:")
        print(by_family.to_string(index=False))


    #Worst-case instances
    print(f"\n{sep}")
    print("WORST-CASE INSTANCES  [top 5 per criterion]")
    print(sep)

    wc = worst_cases(df, top_k=5)
    if wc.empty:
        print("  (no ILP results available for worst-case analysis)")
        return

    wc.to_csv(output_path / "worst_cases.csv", index=False)

    for criterion in ["integrality_gap", "rounded_vs_ilp_ratio", "greedy_vs_ilp_ratio"]:
        subset = wc[wc["worst_by"] == criterion].head(5)
        label  = criterion.replace("_", " ")
        print(f"\n  Worst by {label}:")
        print(f"    {'instance':45s}  {'gap':>7}  {'rnd/ILP':>8}  {'grd/ILP':>8}")
        print(f"    {'-'*45}  {'-'*7}  {'-'*8}  {'-'*8}")
        for _, row in subset.iterrows():
            gap_s = f"{row['integrality_gap']:.4f}"      if pd.notna(row["integrality_gap"])      else "    N/A"
            rnd_s = f"{row['rounded_vs_ilp_ratio']:.4f}" if pd.notna(row["rounded_vs_ilp_ratio"]) else "    N/A"
            grd_s = f"{row['greedy_vs_ilp_ratio']:.4f}"  if pd.notna(row["greedy_vs_ilp_ratio"])  else "    N/A"
            print(f"    {row['instance_name']:45s}  {gap_s:>7}  {rnd_s:>8}  {grd_s:>8}")


def save_representative_lp_samples(df: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Save a small table of representative LP-structure examples for the report.
    This avoids storing every x_value vector, but preserves key evidence.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidates = []

    # 1 smallest odd cycle with ILP
    odd_cycle = df[(df["family"] == "cycle") & (df["parity"] == "odd")].sort_values("n")
    if not odd_cycle.empty:
        candidates.append(odd_cycle.iloc[0])

    # 1 smallest even cycle with ILP
    even_cycle = df[(df["family"] == "cycle") & (df["parity"] == "even")].sort_values("n")
    if not even_cycle.empty:
        candidates.append(even_cycle.iloc[0])

    # largest complete graph with ILP
    complete = df[(df["family"] == "complete") & (df["ilp_value"].notna())].sort_values("n")
    if not complete.empty:
        candidates.append(complete.iloc[-1])

    # 1 bipartite example
    bip = df[(df["family"] == "bipartite") & (df["ilp_value"].notna())].sort_values("n")
    if not bip.empty:
        candidates.append(bip.iloc[0])

    if not candidates:
        return

    rep = pd.DataFrame(candidates)[[
        "family", "instance_name", "n", "m",
        "lp_value", "ilp_value", "integrality_gap",
        "lp_num_fractional", "lp_num_at_half", "lp_is_integral",
        "rounded_size", "rounded_vs_ilp_ratio",
        "greedy_size", "greedy_vs_ilp_ratio"
    ]]
    rep.to_csv(output_path / "representative_lp_samples.csv", index=False)




# Main
if __name__ == "__main__":
    output_dir = "results"
    print("Running CMPS 351 Project 3 experiments ...\n")
    df = run_all_experiments(output_dir=output_dir, ilp_max_n=ILP_MAX_N)
    print_report_notes(df, output_dir=output_dir)
    save_representative_lp_samples(df, output_dir=output_dir)
    print("\nSaved to:")
    print(f"  {output_dir}/experiment_results.csv")
    print(f"  {output_dir}/summary_by_family.csv")
    print(f"  {output_dir}/summary_by_family_and_size.csv")
    print(f"  {output_dir}/worst_cases.csv")
    print(f"  {output_dir}/representative_lp_samples.csv")
