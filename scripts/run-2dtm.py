import argparse
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import toml
from cryojax.image import operators as op
from cryojax.io import read_array_from_mrc
from cryojax.rotations import convert_quaternion_to_euler_angles, SO3
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)-6s [%(filename)s:%(lineno)d] %(message)s")


# Simulated image parameters
PADDING_FRACTION_OF_3D_TEMPLATE: float = 0.25
# Search parameters. First, parameters for the grid of poses
RNG_SEED: int = 1234
NUMBER_OF_POSES_TO_SEARCH: int = 4_000_000
# ... now, parameters for the pixel size search
NUMBER_OF_PIXEL_SIZES_TO_SEARCH: int = 5
PIXEL_SIZE_SEARCH_RANGE_AS_A_FRACTION: float = 0.01
# ... and finally the defocus search
NUMBER_OF_DEFOCUS_OFFSETS_TO_SEARCH: int = 5
DEFOCUS_OFFSET_SEARCH_RANGE_IN_ANGSTROMS: float = 100.0
# Control flow specification for the search
BATCH_SIZE_OVER_GRID_POINTS: int = 100
NUMBER_OF_PROGRESS_BAR_UPDATES: int = 10


def load_template_and_metadata(
    path_to_template_metadata: Path,
) -> tuple[Float[Array, "_ _ _"], dict[str, Any]]:
    # Load metadata
    with open(path_to_template_metadata, "r") as metadata_file:
        template_metadata = toml.load(metadata_file)
    template = read_array_from_mrc(template_metadata["path_to_template"])
    return (
        jnp.asarray(template),
        dict(
            voxel_size=template_metadata["voxel_size"],
        ),
    )


@eqx.filter_jit
def build_whitening_filter(
    noise_image_stack: Float[Array, "n_images y_dim x_dim"], shape: tuple[int, int]
) -> op.WhiteningFilter:

    @jax.vmap
    def normalize_image_stack(image_stack):
        return (image_stack - jnp.mean(image_stack)) / jnp.std(
            image_stack,
        )

    whitening_filter = op.WhiteningFilter(
        normalize_image_stack(noise_image_stack),
        shape,
        interpolation_mode="linear",
    )

    return whitening_filter


@eqx.filter_jit
def build_uniform_orientation_grid(
    rng_key: PRNGKeyArray,
) -> tuple[
    Float[Array, " n_angles"],
    Float[Array, " n_angles"],
    Float[Array, " n_angles"],
]:
    rng_key_per_pose = jr.split(rng_key, NUMBER_OF_POSES_TO_SEARCH)
    # Create uniform grid over SO3
    uniform_phi_angles, uniform_theta_angles, uniform_psi_angles = eqx.filter_vmap(
        lambda k: convert_quaternion_to_euler_angles(SO3.sample_uniform(k).wxyz),
    )(rng_key_per_pose).T

    return uniform_phi_angles, uniform_theta_angles, uniform_psi_angles


@eqx.filter_jit
def build_pixel_size_grid(mean_pixel_size: Float[Array, ""]) -> Float[Array, " _"]:
    pixel_size_search_range_in_angstroms = (
        mean_pixel_size * PIXEL_SIZE_SEARCH_RANGE_AS_A_FRACTION
    )
    return jnp.linspace(
        mean_pixel_size - pixel_size_search_range_in_angstroms / 2,
        mean_pixel_size + pixel_size_search_range_in_angstroms / 2,
        NUMBER_OF_PIXEL_SIZES_TO_SEARCH,
    )


@eqx.filter_jit
def build_defocus_grid(
    mean_defocus_per_particle: Float[Array, " n_images"]
) -> Float[Array, "_ n_images"]:
    defocus_offset_search_values = jnp.linspace(
        -DEFOCUS_OFFSET_SEARCH_RANGE_IN_ANGSTROMS / 2,
        DEFOCUS_OFFSET_SEARCH_RANGE_IN_ANGSTROMS / 2,
        NUMBER_OF_DEFOCUS_OFFSETS_TO_SEARCH,
    )
    return defocus_offset_search_values[:, None] + mean_defocus_per_particle[None, :]


def display_test_image_and_cross_correlation(
    simulated_image: Float[Array, "y_dim x_dim"],
    observed_image: Float[Array, "y_dim x_dim"],
    cross_correlation: Float[Array, "y_dim x_dim"],
):
    # Plotting test simulated image
    aspect_ratio = observed_image.shape[2] / observed_image.shape[1]
    fig, axes = plt.subplots(figsize=(3 * 5 * aspect_ratio, 5), dpi=100, ncols=3)
    plot_image_with_colorbar(simulated_image, fig, axes[0], cmap="gray")
    axes[0].set(title="Test simulated image")
    # ... and observed image
    plot_image_with_colorbar(observed_image, fig, axes[1], cmap="gray")
    axes[1].set(title="Test observed image")
    # ... and cross-correlation
    plot_image_with_colorbar(cross_correlation, fig, axes[2], cmap="plasma")
    axes[2].set(title="Test cross-correlation")
    plt.tight_layout()
    plt.show()


def display_whitening_filter(whitening_filter: op.WhiteningFilter):
    whitening_filter_array = jnp.fft.fftshift(
        jnp.asarray((whitening_filter.get().at[0, 0].set(jnp.nan))), axes=(0,)
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    plot_image_with_colorbar(whitening_filter_array.T, fig, ax, cmap="gray")
    ax.set(title="Whitening filter computed from noise images.")
    plt.tight_layout()
    plt.show()


def display_maximized_cross_correlation(maximized_cc: Float[np.ndarray, "y_dim x_dim"]):
    # Plotting test simulated image
    simulated_aspect_ratio = maximized_cc.shape[2] / maximized_cc.shape[1]
    fig, ax = plt.subplots(figsize=(5 * simulated_aspect_ratio, 5), dpi=100)
    plot_image_with_colorbar(maximized_cc, fig, ax, cmap="grey")
    ax.set(title="Test maximized cross-correlation")
    plt.tight_layout()
    plt.show()


def plot_image_with_colorbar(image, fig, ax, cmap="gray"):
    im = ax.imshow(image, cmap=cmap, origin="lower", interpolation=None)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)


def main(
    path_to_micrographs: Path,
    path_to_ctffind_results: Path,
    paths_to_template_metadata: tuple[Path, ...],
    search_pixel_size: bool,
    search_defocus: bool,
    use_whitening_filter: bool,
    plot_test_image: bool,
    plot_maximized_cc: bool,
    plot_whitening_filter: bool,
    show_progress_bar: bool,
    output_folder: Path,
):
    raise NotImplementedError


if __name__ == "__main__":
    # Setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_micrographs",
        type=str,
        help="Path to folder of micrographs.",
    )
    parser.add_argument(
        "path_to_ctffind_results",
        type=str,
        help="Path to the results of CTFFIND search.",
    )
    parser.add_argument(
        "paths_to_template_metadata",
        type=str,
        nargs="+",
        help="Path to metadata '.toml' files for MT templates.",
    )
    parser.add_argument(
        "--use-whitening-filter",
        action=argparse.BooleanOptionalAction,
        help=(
            "Whiten images using the noise particle stack given by the "
            "'--path-to-noise-starfile' flag."
        ),
    )
    parser.add_argument(
        "--search-pixel-size",
        action=argparse.BooleanOptionalAction,
        help="Search over pixel sizes within specified range.",
    )
    parser.add_argument(
        "--search-defocus",
        action=argparse.BooleanOptionalAction,
        help="Search over defocus values within specified range.",
    )
    parser.add_argument(
        "--plot-test-image",
        action=argparse.BooleanOptionalAction,
        help="Plot a test simulated image before running the search.",
    )
    parser.add_argument(
        "--plot-maximized-cc",
        action=argparse.BooleanOptionalAction,
        help="Plot the maximum cross-correlation-per-pixel after running the search.",
    )
    parser.add_argument(
        "--plot-whitening-filter",
        action=argparse.BooleanOptionalAction,
        help="Plot the whitening filter used for computing the cross-correlation score.",
    )
    parser.add_argument(
        "--show-progress-bar",
        action=argparse.BooleanOptionalAction,
        help="Show a tqdm progress bar during the cross-correlation search.",
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, help="Output folder to write results."
    )
    parser.add_argument("-l", "--log", type=str, help="Set level of logger.")
    # Set defaults
    parser.set_defaults(
        search_pixel_size=False,
        search_defocus=False,
        plot_test_image=False,
        plot_maximized_cc=False,
        plot_whitening_filter=False,
        use_whitening_filter=False,
        show_progress_bar=False,
    )
    # Parse arguments
    args = parser.parse_args()
    # Unpack parser
    path_to_micrographs = Path(args.path_to_micrographs)
    path_to_ctffind_results = Path(args.path_to_ctffind_results)
    paths_to_template_metadata = tuple([Path(p) for p in args.paths_to_template_metadata])
    log_level = args.log or "INFO"
    output_folder = Path(args.output_folder or ".")
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    # Run
    main(
        path_to_micrographs,
        path_to_ctffind_results,
        paths_to_template_metadata,
        args.search_pixel_size,
        args.search_defocus,
        args.use_whitening_filter,
        args.plot_test_image,
        args.plot_maximized_cc,
        args.plot_whitening_filter,
        args.show_progress_bar,
        output_folder,
    )
