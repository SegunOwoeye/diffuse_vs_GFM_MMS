# Diffuse Interface and rGFM Solvers for Multimaterial Compressible Flow

Author: Olusegun Owoeye

This repository contains the research code used to compare sharp and diffuse-interface treatments for compressible multimaterial flow. It includes a single-material Euler solver, a real Ghost Fluid Method (rGFM) sharp-interface method (SIM) solver, and a diffuse-interface method (DIM) based on the Allaire five-equation model.

The standard way to use it is through the scripts in `tests/validation/`.

---

## Contents

```text
configs/      Input files for Toro, Fedkiw-style, DIM, rGFM, and bubble cases
data/         Generated CSV files, plots, reference data, and analysis outputs
src/          C++ solver code and Python post-processing scripts
tests/        Bash scripts for compiling, running, and plotting validation cases
```

Main source folders:

```text
src/app/       Program entry points and solver drivers
src/euler/     Shared Euler state, flux, thermodynamics, EoS, and Euler Riemann/reconstruction code
src/sim/       Sharp-interface method (SIM): GFM/rGFM, level sets, and sharp-interface advancement
src/dim/       Diffuse-interface Allaire-model code
src/fv/        Shared finite-volume utilities used by more than one method
src/io/        Config parsing and CSV output
src/setup/     Initial conditions
src/graphing/  Plotting and error analysis
```

---

## System Requirements

Recommended platform: Linux or WSL.

Required software:

- `g++` with C++17 and OpenMP support
- Bash
- Python 3.10 or newer
- Python packages: `numpy`, `pandas`, `matplotlib`
- Optional: `openpyxl`, if working with Excel validation data

The development environment used for the thesis runs was:

```text
g++ 13.3.0
Python 3.12.3
numpy 2.4.3
pandas 3.0.1
matplotlib 3.10.8
```

Create the Python environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib openpyxl
```

All commands below assume they are run from the repository root.

---

## Building

The project does not use CMake. The validation scripts compile the required executables directly with `g++`.

Single-material solver:

```bash
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/sm_main.cpp -o sm_main_1d
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/sm_main.cpp -o sm_main_2d
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=3 src/app/sm_main.cpp -o sm_main_3d
```

Multimaterial solver, used for rGFM and DIM:

```bash
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=1 src/app/multimaterial_main.cpp -o mm_main_1d
g++ -std=c++17 -O2 -fopenmp -I. -DAPP_DIM=2 src/app/multimaterial_main.cpp -o mm_main_2d
```

The spatial dimension is selected at compile time using `APP_DIM`.

---

## Quick Start

Run the single-material validation suite:

```bash
chmod +x tests/validation/validation_Tests.sh
./tests/validation/validation_Tests.sh --method sm
```

Run both multimaterial methods in 1D:

```bash
./tests/validation/validation_Tests.sh --method both --dims 1
```

Run the 2D multimaterial validation suite, including the 45-degree oblique Fedkiw reductions:

```bash
./tests/validation/validation_Tests.sh --method both --dims 2
```

Run the 2D helium shock-bubble case:

```bash
CORES=6 ./tests/validation/validation_Tests.sh --case bubble --method both
```

Large 2D and 3D cases generate large CSV files and can take a long time. For a fresh installation, start with the 1D validation cases.

---

## Running Individual Cases

Each executable takes one configuration file:

```bash
./sm_main_1d configs/toro/test1.txt
./sm_main_2d configs/toro/explosion1.txt
./sm_main_3d configs/toro/explosion2.txt

./mm_main_1d configs/GFM/MM_1D_validation/test1.txt
./mm_main_1d configs/DIM/MM_1D_validation/test1.txt
./mm_main_2d configs/GFM/MM_2D_validation/test6.txt
./mm_main_2d configs/DIM/MM_2D_validation/test6.txt
```

OpenMP settings can be set in the usual way:

```bash
OMP_NUM_THREADS=6 OMP_SCHEDULE=dynamic ./mm_main_2d configs/GFM/MM_2D_validation/test6.txt
```

---

## Validation Scripts

Single-material Euler:

```bash
./tests/validation/1_SM_Euler_Tests/SM_simulation.sh
./tests/validation/1_SM_Euler_Tests/SM_graphing.sh
```

rGFM:

```bash
./tests/validation/2_MM_GFM_Tests/MM_GFM_simulation.sh --dims 1
./tests/validation/2_MM_GFM_Tests/MM_GFM_graphing.sh --dims 1
```

DIM:

```bash
./tests/validation/3_MM_DIM_Tests/MM_DIM_simulation.sh --dims 1
./tests/validation/3_MM_DIM_Tests/MM_DIM_graphing.sh --dims 1
```

rGFM/DIM comparison plots:

```bash
./tests/validation/3_MM_DIM_Tests/gfm_dim_comparison.sh
```

Bubble collapse:

```bash
./tests/bubble_collapse_tests.sh gfm
./tests/bubble_collapse_tests.sh dim
./tests/bubble_collapse_tests.sh both
```

Useful driver options:

```text
--method sm|gfm|dim|both|all
--dims 1|2|1,2|all
--case all|bubble
--no-sim
--no-plot
--archive
--clean
--conservation
--conservation-interval N
```

`--clean` deletes generated outputs in `data/csv/` and `data/plots/` while preserving other `data/` assets such as references and validation images.

---

## Implemented Methods

### Single-Material Euler Solver

- Finite-volume discretisation of the compressible Euler equations.
- MUSCL-Hancock reconstruction.
- HLLC numerical fluxes.
- 1D Toro shock tubes with exact/reference output.
- 2D cylindrical and 3D spherical explosion tests.

### rGFM Sharp-Interface Solver

- Level-set representation of the material interface.
- Material-specific equations of state.
- Real/ghost state construction near the interface.
- 1D, 2D grid-aligned, and 45-degree oblique Fedkiw-style validation cases.
- 2D shock interaction with a helium bubble.

### Diffuse-Interface Solver

- Allaire five-equation model for two-material flow.
- Partial densities, mixture momentum, total energy, and volume fraction.
- HLLC update for the conservative mixture subsystem.
- Separate volume-fraction update using the Riemann face velocity.
- Volume-fraction bounding and positivity recovery in mixed cells.
- 1D, 2D grid-aligned, and 45-degree oblique validation cases matching the rGFM workflow.

### Post-Processing

- 1D profile plots.
- L1 error summaries.
- 2D centreline and transverse slices.
- 45-degree oblique normal-coordinate validation slices.
- 2D field maps and 3D wireframe surface plots.
- Schlieren-style density-gradient plots for bubble collapse.

---

## Configuration Files

Configuration files are plain text files with `key = value` entries. Comments begin with `#`.

Common keys:

```text
dimension          Must match APP_DIM used at compile time
domain_min         Lower physical coordinates
domain_max         Upper physical coordinates
N                  Grid resolution; multiple N lines run multiple resolutions
tfinal             Final simulation time
output_times       Optional intermediate output times
cfl                CFL number
interface_method   SM, GFM, or DIM
time_update        split or unsplit; defaults to split
use_level_set      Required for GFM, disabled for DIM
bc_lo, bc_hi       Boundary conditions
material           Material id, EOS type, and EOS parameters
output_prefix      Output file prefix
output_dir         Output directory
```

Supported initial-condition modes:

```text
regions
explosion
double_explosion
shock_bubble
```

Supported boundary conditions:

```text
transmissive
zero_gradient
outflow
reflective
reflection
```

Example 1D single-material configuration:

```text
dimension = 1
domain_min = [0.0]
domain_max = [1.0]
N = [400]
tfinal = 0.25
cfl = 0.3

interface_method = SM
use_level_set = false

material = 0, ideal_gas, gamma=1.4

region = [0.0], [0.5], rho=1.0, vel=[0.0], p=1.0, material=0
region = [0.5], [1.0], rho=0.125, vel=[0.0], p=0.1, material=0

exact_riemann = true
output_prefix = toro1
output_dir = data/csv/toro
```

For rGFM, set `interface_method = GFM` and `use_level_set = true`. For DIM, set
`interface_method = DIM`, `use_level_set = false`, and specify
`interface_thickness`.

---

## Output and Plotting

CSV files are written under the configured `output_dir`, for example:

```text
data/csv/toro/toro1/
data/csv/gfm/MM_1D_validation/gfm_FedkiwA/
data/csv/dim/MM_2D_validation/dim_helium_bubble_2d/
```

Typical CSV columns include:

```text
x0, x1, x2       Cell-centre coordinates
rho              Density or mixture density
p                Pressure
e                Specific internal energy
u0, u1, u2       Velocity components
mat              Material id for sharp-interface outputs
phi...           Level-set fields for rGFM outputs
alpha...         Volume fractions for DIM outputs
mass...          Partial masses for DIM outputs
```

Common plotting commands:

```bash
python src/graphing/plot_1d.py toro/toro1
python src/graphing/compute_l1.py toro/toro1

python src/graphing/plot_multid.py toro/explosion1
python src/graphing/plot_multid.py toro/explosion2

python src/graphing/plot_multid.py --schlieren gfm/MM_2D_validation/gfm_helium_bubble_2d
python src/graphing/plot_multid.py --schlieren dim/MM_2D_validation/dim_helium_bubble_2d
```

---

## Conservation Diagnostics

The validation driver can enable conservation logs:

```bash
./tests/validation/validation_Tests.sh --method both --dims 1 --conservation
```

The output interval can be changed:

```bash
./tests/validation/validation_Tests.sh \
    --method both \
    --dims 1 \
    --conservation \
    --conservation-interval 10
```

The rGFM update uses material-specific real and ghost states near the interface, so it is not expected to be exactly conservative across material boundaries in the same sense as the conservative bulk finite-volume update. Conservation diagnostics are therefore useful for comparing rGFM and DIM behaviour.

---

## Validation Cases

| Location | Purpose |
| --- | --- |
| `configs/toro/test1.txt` to `test5.txt` | 1D Toro shock tubes |
| `configs/toro/explosion1.txt` | 2D cylindrical explosion |
| `configs/toro/explosion2.txt` | 3D spherical explosion |
| `configs/GFM/MM_1D_validation/` | 1D rGFM Fedkiw-style cases |
| `configs/DIM/MM_1D_validation/` | 1D DIM versions of the same cases |
| `configs/GFM/MM_2D_validation/` | 2D rGFM planar and shock-bubble cases |
| `configs/DIM/MM_2D_validation/` | 2D DIM planar and shock-bubble cases |

The current 2D planar Fedkiw-style tests are grid-aligned reductions of the 1D
cases. A useful future extension would initialise the planar discontinuity
obliquely to the Cartesian grid, for example with
`x*cos(theta) + y*sin(theta) = s0`, to test grid-orientation sensitivity.

---

## Known Limitations

- Cartesian structured grids only.
- Directionally split multidimensional update.
- No general-purpose mesh generation or adaptive mesh refinement.
- rGFM is not strictly conservative across the material interface.
- DIM uses practical volume-fraction and positivity repairs in mixed cells.
- 3D validation currently focuses on the single-material spherical explosion.
- Build process is script-based rather than package-based.

---

## References

Core BibTeX entries:

```bibtex
@book{toro2009,
  author    = {Toro, Eleuterio F.},
  title     = {Riemann Solvers and Numerical Methods for Fluid Dynamics:
               A Practical Introduction},
  edition   = {3},
  publisher = {Springer},
  address   = {Berlin, Heidelberg},
  year      = {2009},
  doi       = {10.1007/b79761}
}

@article{FEDKIW1999,
  author  = {Fedkiw, Ronald P. and Aslam, Tariq and Merriman, Barry and
             Osher, Stanley},
  title   = {A Non-oscillatory Eulerian Approach to Interfaces in
             Multimaterial Flows (the Ghost Fluid Method)},
  journal = {Journal of Computational Physics},
  volume  = {152},
  number  = {2},
  pages   = {457--492},
  year    = {1999},
  doi     = {10.1006/jcph.1999.6236}
}

@article{wang2006,
  author  = {Wang, C. W. and Liu, T. G. and Khoo, B. C.},
  title   = {A Real Ghost Fluid Method for the Simulation of Multimedium
             Compressible Flow},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {28},
  number  = {1},
  pages   = {278--302},
  year    = {2006},
  doi     = {10.1137/030601363}
}

@article{Allaire2002,
  author  = {Allaire, Gregoire and Clerc, Sebastien and Kokh, Samuel},
  title   = {A Five-Equation Model for the Simulation of Interfaces between
             Compressible Fluids},
  journal = {Journal of Computational Physics},
  volume  = {181},
  number  = {2},
  pages   = {577--616},
  year    = {2002},
  doi     = {10.1006/jcph.2002.7143}
}

@article{MURRONE2005,
  author  = {Murrone, Angelo and Guillard, Herve},
  title   = {A Five Equation Reduced Model for Compressible Two Phase Flow
             Problems},
  journal = {Journal of Computational Physics},
  volume  = {202},
  number  = {2},
  pages   = {664--698},
  year    = {2005},
  doi     = {10.1016/j.jcp.2004.07.019}
}

@article{SaurelAbgrall1999,
  author  = {Saurel, Richard and Abgrall, Remi},
  title   = {A Simple Method for Compressible Multifluid Flows},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {21},
  number  = {3},
  pages   = {1115--1145},
  year    = {1999},
  doi     = {10.1137/S1064827597323749}
}

@article{JAIN2020,
  author  = {Jain, Suhas S. and Mani, Ali and Moin, Parviz},
  title   = {A Conservative Diffuse-Interface Method for Compressible
             Two-Phase Flows},
  journal = {Journal of Computational Physics},
  volume  = {418},
  pages   = {109606},
  year    = {2020},
  doi     = {10.1016/j.jcp.2020.109606}
}

@article{OSHER1988,
  author  = {Osher, Stanley and Sethian, James A.},
  title   = {Fronts Propagating with Curvature-Dependent Speed: Algorithms
             Based on Hamilton-Jacobi Formulations},
  journal = {Journal of Computational Physics},
  volume  = {79},
  number  = {1},
  pages   = {12--49},
  year    = {1988},
  doi     = {10.1016/0021-9991(88)90002-2}
}

@article{Haas_Sturtevant_1987,
  author  = {Haas, J.-F. and Sturtevant, B.},
  title   = {Interaction of Weak Shock Waves with Cylindrical and Spherical
             Gas Inhomogeneities},
  journal = {Journal of Fluid Mechanics},
  volume  = {181},
  pages   = {41--76},
  year    = {1987},
  doi     = {10.1017/S0022112087002003}
}

@article{Quirk1996,
  author  = {Quirk, James J. and Karni, S.},
  title   = {On the Dynamics of a Shock-Bubble Interaction},
  journal = {Journal of Fluid Mechanics},
  volume  = {318},
  pages   = {129--163},
  year    = {1996},
  doi     = {10.1017/S0022112096007069}
}
```
