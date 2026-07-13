# Graphing Module Layout

The production report plotting entry point is:

```text
src/graphing/plot_report2_selected_suite.py
```

It is called by:

```text
scripts/plot_report2_selected_suite.sh
```

Use that path for Report 2 figures. It handles the selected quantitative suite, keeps output names consistent, mirrors figures into the shared report figure directory and updates the run tracker through the wrapper script.

## Shared Helpers

Keep reusable plotting logic in shared modules:

- `multid_io.py`
- `multid_render.py`
- `multid_schlieren.py`
- `multid_slices.py`
- `plot_style.py`
- `error_metrics.py`
- `exact_reference.py`
- `fedkiw_common.py`

## Legacy Or Exploratory Plotters

Several older files remain for archived validation workflows, especially solid and elastoplastic checks. Do not add new report cases by creating another top-level paper-specific plotter. Add report cases to `plot_report2_selected_suite.py` and reuse shared helpers instead.

The report-facing graphing policy is documented in:

```text
docs/graphing_consolidation.md
```
