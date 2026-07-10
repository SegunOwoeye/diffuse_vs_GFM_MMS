"""Update the Report 2 run tracker from result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


TRACKER = Path("docs/report2_run_tracker.md")
RESULT_BASE = Path("results/quantitative")


@dataclass(frozen=True)
class RunRow:
    section: str
    run: str
    selector: str
    root: str
    evidence: str
    notes: str
    default_status: str = "Pending"
    default_plots: str = "Pending"


CORE_ROWS = [
    RunRow("Core Report Runs", "Exact Fedkiw references", "tools/generate_fedkiw_exact_references.py", "data/exact/fedkiw", "generated exact CSVs", "Required before FedkiwD2 accuracy tables"),
    RunRow("Core Report Runs", "Toro test 5 1D Euler check", "--case toro_1d,test5", "results/quantitative/report_selected_toro_test5", "error, convergence, and PNG outputs", "Underlying 1D accuracy and convergence"),
    RunRow("Core Report Runs", "2D explosion Euler check", "--case explosion2d", "results/quantitative/report_selected_explosion_2d", "PNG outputs and runtime files", "Underlying 2D behaviour"),
    RunRow("Core Report Runs", "3D explosion Euler check", "--case explosion3d", "results/quantitative/report_selected_explosion_3d", "PNG outputs and runtime files", "3D extension evidence"),
    RunRow("Core Report Runs", "FedkiwD2 1D SIM versus DIM", "--case fedkiw_1d,test5", "results/quantitative/report_selected_fedkiw_d2_1d", "error, convergence, conservation, interface, performance, and PNG outputs", "Main 1D multimaterial comparison"),
    RunRow("Core Report Runs", "2D helium shock-bubble", "--case bubble", "results/quantitative/report_selected_helium_bubble_2d", "conservation, bubble feature, interface, performance, and PNG outputs", "Main 2D gas-gas comparison"),
    RunRow("Core Report Runs", "3D helium shock-bubble", "--case bubble3d", "results/quantitative/report_selected_helium_bubble_3d", "summary, runtime, and PNG outputs", "Main 3D gas-gas extension"),
    RunRow("Core Report Runs", "2D Gorsse TC9 water-air bubble", "--case gorsse_tc9", "results/quantitative/report_selected_gorsse_tc9_water_air_2d", "conservation, interface, performance, and Schlieren PNG outputs", "Main water-air and multiple-EOS case"),
    RunRow("Core Report Runs", "He 2023 three-material 1D convergence", "--case he2023_three_material_1d", "results/quantitative/report_selected_he2023_three_material_1d", "self-reference convergence, conservation, interface, performance, and PNG outputs", "Initial K-material 1D validation against N=2000 reference"),
    RunRow("Core Report Runs", "He 2023 three-material triple-point 2D", "--case he2023_triple_point", "results/quantitative/report_selected_he2023_three_material_triple_point_2d", "density, pressure, material-map, Schlieren, conservation, and performance outputs", "Main three-material DIM versus rGFM comparison at paper resolution 1400x600"),
    RunRow("Core Report Runs", "3D Gorsse TC9 water-air bubble", "--case gorsse_tc9_3d", "results/quantitative/report_selected_gorsse_tc9_water_air_3d", "summary, runtime, and PNG outputs", "3D water-air extension with zero z velocity"),
]

SENSITIVITY_ROWS = [
    RunRow("Sensitivity Runs", "rGFM reinitialisation sensitivity", "--sensitivity sim_reinit", "results/quantitative/report_selected_sim_reinit_sensitivity", "sensitivity summary and PNG panel", "Best visual evidence for rGFM level-set maintenance"),
    RunRow("Sensitivity Runs", "DIM interface-thickness sensitivity", "--sensitivity dim_epsilon", "results/quantitative/report_selected_dim_alpha_sensitivity", "sensitivity summary and PNG panel", "Current tan-alpha smoothing proxy"),
]

PERFORMANCE_ROWS = [
    RunRow("Performance Runs", "OpenMP shock-bubble scaling", "--scaling openmp_threads", "results/quantitative/report_selected_openmp_scaling", "performance summary and PNG output", "Main speedup evidence"),
    RunRow("Performance Runs", "Additional scaling for long one-core cases", "To be selected after first runtime pass", "", "runtime files and performance summaries", "Add only for cases with one-core runtime above about 60 seconds", "Optional", "Optional"),
]

OPTIONAL_ROWS = [
    RunRow("Optional Or Future Runs", "Elastoplastic or solid-fluid extension caution", "Excluded from selected suite", "docs/elastoplastic_scope_caution.md", "scope note only", "Exploratory architecture evidence; not part of the main DIM versus rGFM comparison", "Optional", "Optional"),
    RunRow("Optional Or Future Runs", "Four or more interacting fluids demonstration", "Not selected yet", "", "case-specific summary and plot", "Three-material case is selected; four-fluid case still needs a separate diagnostic", "Optional", "Optional"),
]


def root_path(row: RunRow) -> Path | None:
    if not row.root:
        return None
    return Path(row.root)


def run_status(row: RunRow) -> tuple[str, str]:
    if row.default_status in {"Optional", "Blocked"}:
        return row.default_status, ""

    if row.run == "Exact Fedkiw references":
        root = Path(row.root)
        count = len(list(root.glob("*exact*.csv"))) if root.exists() else 0
        if count > 0:
            return "Complete", completed_date(root)
        return "Pending", ""

    root = root_path(row)
    if root is None:
        return row.default_status, ""
    if (root / "manifest.json").exists() and (root / "summaries" / "summary.csv").exists():
        return "Complete", completed_date(root)
    return row.default_status, ""


def plot_status(row: RunRow) -> str:
    if row.default_plots in {"Optional", "Blocked"}:
        return row.default_plots
    root = root_path(row)
    if root is None or row.run == "Exact Fedkiw references":
        return "N/A"
    local_pngs = list((root / "figures" / "report_png").glob("*.png"))
    global_pngs = list((RESULT_BASE / "report_selected_figures").glob("*.png"))
    key = key_for_run(row)
    matching_global = [path for path in global_pngs if key in path.name]
    if local_pngs or matching_global:
        return "Complete"
    return row.default_plots


def completed_date(root: Path) -> str:
    candidates = [root / "manifest.json", root / "summaries" / "summary.csv"]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        existing = list(root.glob("*.csv"))
    if not existing:
        return ""
    latest = max(path.stat().st_mtime for path in existing)
    return date.fromtimestamp(latest).isoformat()


def key_for_run(row: RunRow) -> str:
    return (
        row.run.lower()
        .replace("1d", "1d")
        .replace("2d", "2d")
        .replace("3d", "3d")
        .replace(" ", "_")
        .replace("-", "_")
    )


def render_table(rows: list[RunRow]) -> list[str]:
    lines = [
        "| Status | Plots | Run | Command selector | Result root | Completed | Evidence to check | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        status, completed = run_status(row)
        plots = plot_status(row)
        lines.append(
            f"| {status} | {plots} | {row.run} | `{row.selector}` | `{row.root}` | {completed} | {row.evidence} | {row.notes} |"
        )
    return lines


def render() -> str:
    return "\n".join([
        "# Report 2 Run Tracker",
        "",
        "This file is generated by:",
        "",
        "```bash",
        "scripts/update_report2_run_tracker.py",
        "```",
        "",
        "Status values: `Pending`, `Running`, `Complete`, `Blocked`, and `Optional`.",
        "",
        "# [0] Threading",
        "",
        "The one-stop runner currently defaults to six OpenMP threads:",
        "",
        "```bash",
        "scripts/run_report2_selected_suite.sh",
        "```",
        "",
        "On the university machine, increase the thread count with:",
        "",
        "```bash",
        "scripts/run_report2_selected_suite.sh --omp-threads 32",
        "```",
        "",
        "For a faster partial run:",
        "",
        "```bash",
        "scripts/run_report2_selected_suite.sh --skip-3d --skip-scaling",
        "```",
        "",
        "For the current university-machine catch-up run, use:",
        "",
        "```bash",
        "scripts/run_report2_university_suite.sh --omp-threads 32",
        "```",
        "",
        "For your personal default, use the wrapper that skips scaling unless you ask for it:",
        "",
        "```bash",
        "scripts/run_report2_personal_suite.sh",
        "```",
        "",
        "For a CSC machine at 32 cores, use:",
        "",
        "```bash",
        "scripts/run_report2_csc.sh --omp-threads 32 --account-name oo338",
        "```",
        "",
        "Only include the optional 3D Gorsse TC9 extension if time is available:",
        "",
        "```bash",
        "scripts/run_report2_university_suite.sh --omp-threads 32 --include-gorsse-3d",
        "```",
        "",
        "# [1] Core Report Runs",
        "",
        *render_table(CORE_ROWS),
        "",
        "# [2] Sensitivity Runs",
        "",
        *render_table(SENSITIVITY_ROWS),
        "",
        "# [3] Performance Runs",
        "",
        *render_table(PERFORMANCE_ROWS),
        "",
        "# [4] Optional Or Future Runs",
        "",
        *render_table(OPTIONAL_ROWS),
        "",
        "# [5] Completion Checklist",
        "",
        "Mark a run `Complete` only after these files exist:",
        "",
        "```text",
        "results/quantitative/<run_name>/manifest.json",
        "results/quantitative/<run_name>/summaries/summary.csv",
        "```",
        "",
        "For report figures, also check:",
        "",
        "```text",
        "results/quantitative/<run_name>/figures/report_png/*.png",
        "results/quantitative/report_selected_figures/*.png",
        "```",
        "",
    ]) + "\n"


def main() -> None:
    TRACKER.write_text(render(), encoding="utf-8")
    print(f"Updated {TRACKER}")


if __name__ == "__main__":
    main()
