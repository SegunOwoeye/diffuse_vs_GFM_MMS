"""Compatibility CLI for multidimensional solver plots."""

from pathlib import Path

from multid_io import build_output_name, time_tagged_csv_files
from multid_render import (
    plot_multiple_cpp_solutions,
    plot_pressure_contours_from_times,
    plot_schlieren_sequence_from_times,
)


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/graphing/plot_multid.py [--schlieren] [--pressure-contours] file.csv")
        print("python src/graphing/plot_multid.py [--schlieren] [--pressure-contours] directory_name")
        print("python src/graphing/plot_multid.py [--schlieren] [--same-material-schlieren] [--interface-overlay] file1.csv file2.csv")
        raise SystemExit(1)

    force_schlieren = False
    pressure_contours = False
    same_material_schlieren = False
    interface_overlay = False
    args = []

    for arg in sys.argv[1:]:
        if arg == "--schlieren":
            force_schlieren = True
        elif arg == "--paper-schlieren":
            force_schlieren = True
        elif arg == "--same-material-schlieren":
            same_material_schlieren = True
        elif arg == "--interface-overlay":
            interface_overlay = True
        elif arg == "--pressure-contours":
            pressure_contours = True
        else:
            args.append(arg)

    if not args:
        raise SystemExit("No input files or directories provided")

    data_root = Path("data/csv")

    if len(args) == 1:
        path_arg = data_root / args[0]

        if path_arg.is_dir():
            if pressure_contours:
                save_path = path_arg / f"{path_arg.name}_pressure_contours.png"
                plot_pressure_contours_from_times(path_arg, save_path=save_path)
                raise SystemExit(0)

            csv_files = sorted([f.name for f in path_arg.glob("*.csv") if "_N" in f.name])

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path_arg}")

            filenames = [str(Path(args[0]) / f) for f in csv_files]

            output_name = build_output_name(args[0])
            save_path = path_arg / f"{output_name}.png"

            if force_schlieren and time_tagged_csv_files(path_arg):
                plot_schlieren_sequence_from_times(
                    path_arg,
                    data_root,
                    save_path=save_path,
                    same_material_schlieren=same_material_schlieren,
                    interface_overlay=interface_overlay,
                )
                raise SystemExit(0)

            plot_multiple_cpp_solutions(
                filenames,
                title=output_name,
                save_path=save_path,
                force_schlieren=force_schlieren,
                same_material_schlieren=same_material_schlieren,
                interface_overlay=interface_overlay,
            )
        else:
            plot_multiple_cpp_solutions(
                [args[0]],
                force_schlieren=force_schlieren,
                same_material_schlieren=same_material_schlieren,
                interface_overlay=interface_overlay,
            )

    else:
        plot_multiple_cpp_solutions(
            args,
            force_schlieren=force_schlieren,
            same_material_schlieren=same_material_schlieren,
            interface_overlay=interface_overlay,
        )


if __name__ == "__main__":
    main()
