# Solid Modelling Refactor Report

## Active Structure

- `src/app/solid_main.cpp` is now a small entry point for active Barton elastoplastic configs.
- `src/solid/elastoplastic/barton.hpp` is the public include.
- Active Barton code is merged by solver responsibility:
  - `src/solid/elastoplastic/barton/state.hpp`
  - `src/solid/elastoplastic/barton/eos.hpp`
  - `src/solid/elastoplastic/barton/flux.hpp`
  - `src/solid/elastoplastic/barton/plasticity.hpp`
  - `src/solid/elastoplastic/barton/advance.hpp`
  - `src/solid/elastoplastic/barton/config.hpp`
  - `src/solid/elastoplastic/barton/initial_conditions.hpp`
  - `src/solid/elastoplastic/barton/output.hpp`
  - `src/solid/elastoplastic/barton/driver.hpp`
- Shared text/config parsing helpers live in:
  - `src/solid/common/config_text.hpp`
- Shared finite-volume infrastructure now lives in:
  - `src/core/fv/limiters.hpp`
  - `src/core/fv/directional_update.hpp`
- The older `src/fv/...` headers remain as compatibility wrappers for existing SIM/DIM code.

## Removed Bloat

- Deleted unused legacy solid code:
  - `src/solid/legacy/`
- Deleted unused legacy solid configs:
  - `configs/solid/legacy/`
- Deleted unused legacy validation helpers:
  - `tools/validation/solid/legacy/`
  - `tests/validation/5_Elastoplastic_Tests/EP_legacy_validation.sh`

## Active Validation Layout

- `configs/solid/Barton_1D_validation/test1.txt`
- `configs/solid/Barton_1D_validation/test2.txt`
- `configs/solid/Barton_2D_validation/test1.txt`
- `configs/solid/Barton_2D_validation/test1_debug.txt`
- `configs/solid/Barton_3D_validation/test1.txt`
- `configs/solid/Barton_3D_validation/test1_debug.txt`

## Notes

- Validation configs may still describe the physical benchmark, material, and paper reference.
- Solver files should not be named after a paper section, material, or one specific test case.
- The tensor material constants are read from the config `material = ...` line; copper is a validation material, not a hard-coded solver identity.
- Barton tensor configs now support `dimension = 2` and `dimension = 3`; the 3D Section 6.3 case writes Cartesian scatter data and a one-dimensional spherical reference.
- Barton tensor configs can also export passive Sulsky-style material-point tracers for ParaView using `material_points = true`.
- Euler, DIM, and Barton now share the same canonical limiter implementation through `src/core/fv/limiters.hpp`.
