"""From a PDB, rasterize a voxel grid template."""

import argparse
import logging
import toml
from pathlib import Path
from typing import Any

import cryojax.simulator as cxs
import jax
import numpy as np
from cryojax.image import downsample_with_fourier_cropping
from cryojax.io import read_atoms_from_pdb, write_volume_to_mrc
from jaxtyping import Float
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)-6s [%(filename)s:%(lineno)d] %(message)s")


# Template generation settings
VOXEL_SIZE: float = 2.0
TEMPLATE_SHAPE: tuple[int, int, int] = (250, 250, 250)
B_FACTOR_SCALE_FACTOR: float = 0.0
MINIMUM_B_FACTOR: float = 0.0

# Finer details of template generation
UPSAMPLING_FACTOR: int = 1
Z_PLANES_IN_PARALLEL: int = 1
ATOM_GROUPS_IN_SERIES: int = 1


def build_atom_potential(path_to_mt_pdb: Path) -> cxs.PengAtomicPotential:
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        path_to_mt_pdb,
        get_b_factors=True,
        center=True,
        assemble=False,  # TODO: What does this do?
    )
    b_factors = MINIMUM_B_FACTOR + B_FACTOR_SCALE_FACTOR * b_factors
    return cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors=b_factors)


def compute_and_write_template(
    atom_potential: cxs.PengAtomicPotential,
    path_to_pdb: Path,
    output_folder: Path,
) -> tuple[Float[np.ndarray, "_ _ _"], dict[str, Any]]:
    # Build the voxel grid
    logger.info(
        f"Computing template of shape {TEMPLATE_SHAPE} upsampled by a factor of "
        f"{UPSAMPLING_FACTOR}..."
    )
    upsampled_shape = (
        TEMPLATE_SHAPE[0] * UPSAMPLING_FACTOR,
        TEMPLATE_SHAPE[1] * UPSAMPLING_FACTOR,
        TEMPLATE_SHAPE[2] * UPSAMPLING_FACTOR,
    )
    upsampled_voxel_size = VOXEL_SIZE / UPSAMPLING_FACTOR
    upsampled_potential_as_voxel_grid = atom_potential.as_real_voxel_grid(
        upsampled_shape,
        upsampled_voxel_size,
        z_planes_in_parallel=Z_PLANES_IN_PARALLEL,
        atom_groups_in_series=ATOM_GROUPS_IN_SERIES,
    )
    if UPSAMPLING_FACTOR == 1:
        potential_as_voxel_grid = upsampled_potential_as_voxel_grid
    else:
        potential_as_voxel_grid = downsample_with_fourier_cropping(
            upsampled_potential_as_voxel_grid, UPSAMPLING_FACTOR
        )
    # Write the voxel grid to MRC format
    path_to_template = Path(
        output_folder,
        f"{path_to_pdb.stem}-"
        f"vs{str(VOXEL_SIZE).replace('.', '_')}-"
        f"bf{str(B_FACTOR_SCALE_FACTOR).replace('.', '_')}.mrc",
    )
    write_volume_to_mrc(
        potential_as_voxel_grid, VOXEL_SIZE, path_to_template, overwrite=True
    )
    # Compile metadata of the template into a dictionary and return
    template_metadata = dict(
        path_to_template=str(path_to_template.resolve().absolute()),
        path_to_pdb=str(path_to_pdb.resolve().absolute()),
        voxel_size=VOXEL_SIZE,
        shape=TEMPLATE_SHAPE,
        b_factor_scale_factor=B_FACTOR_SCALE_FACTOR,
        b_factor_minimum_value=MINIMUM_B_FACTOR,
        upsampling_factor=UPSAMPLING_FACTOR,
    )
    return np.asarray(potential_as_voxel_grid), template_metadata


def write_template_metadata(template_metadata: dict[str, Any]):
    # Write to file
    path_to_template = Path(template_metadata["path_to_template"])
    path_to_metadata = Path(path_to_template.parent, f"{path_to_template.stem}.toml")
    with open(path_to_metadata, "w") as toml_file:
        toml.dump(template_metadata, toml_file)


def display_projection_plots(potential_as_voxel_grid: Float[np.ndarray, "_ _ _"]):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3), dpi=100)
    axis_index_to_name = {0: "z", 1: "y", 2: "x"}
    for idx in range(3):
        im = axes[idx].imshow(
            np.sum(potential_as_voxel_grid, axis=idx),
            cmap="gray",
            origin="lower",
            interpolation=None,
        )
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        axes[idx].set(title=f"{axis_index_to_name[idx]} projection")
    plt.tight_layout()
    plt.show()


def main(
    path_to_pdb: Path,
    write_metadata: bool,
    plot_projections: bool,
    output_folder: Path,
):
    # Build potential and write template
    atom_potential = build_atom_potential(path_to_pdb)
    potential_as_voxel_grid, template_metadata = compute_and_write_template(
        atom_potential,
        path_to_pdb,
        output_folder,
    )
    # Write the metadata from the computation
    if write_metadata:
        write_template_metadata(template_metadata)
    # Display central projections
    if plot_projections:
        display_projection_plots(potential_as_voxel_grid)


if __name__ == "__main__":
    # Setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_pdb", type=str, help="Path to PDB file")
    parser.add_argument(
        "--write-metadata",
        action=argparse.BooleanOptionalAction,
        help="Write the details of template generation to a '.toml' file.",
    )
    parser.add_argument(
        "--plot-projections",
        action=argparse.BooleanOptionalAction,
        help="Display projection plots of the template central cross-sections.",
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, help="Output folder to write results."
    )
    parser.add_argument("-l", "--log", type=str, help="Set level of logger")
    # Defaults
    parser.set_defaults(plot_projections=False, flip_polarity=False)
    # Unpack arg parser
    args = parser.parse_args()
    log_level = args.log or "INFO"
    write_metadata = args.write_metadata
    plot_projections = args.plot_projections
    path_to_pdb = Path(args.path_to_pdb)
    output_folder = (
        Path(args.output_folder) if args.output_folder is not None else path_to_pdb.parent
    )
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    # Run
    main(path_to_pdb, write_metadata, plot_projections, output_folder)
