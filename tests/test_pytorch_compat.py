import sys
from pathlib import Path

import pytest
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLES_ROOT))

import airlab as al
from airlab.loss import pairwise as pairwise_loss
from airlab.regulariser import demons as demons_reg
from airlab.utils import matrix as matrix_utils
from airlab.utils.imageFilters import remove_bed_filter
from create_test_image_data import create_C_2_O_test_images


@pytest.fixture()
def image_triplet():
    th.manual_seed(0)
    return create_C_2_O_test_images(32, dtype=th.float32, device=th.device("cpu"))


def test_expm_eig_matches_matrix_exp_for_symmetric_input():
    matrix = th.tensor(
        [[-2.0, 0.3, 0.0], [0.3, -1.5, 0.2], [0.0, 0.2, -1.0]],
        dtype=th.float32,
    )

    actual = matrix_utils.expm_eig(matrix)
    expected = th.linalg.matrix_exp(matrix)

    assert th.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pairwise_losses_and_warping_run_on_zero_displacement(image_triplet):
    fixed_image, moving_image, shaded_image = image_triplet
    zero_disp = th.zeros(*fixed_image.size, len(fixed_image.size), dtype=th.float32)

    losses = [
        pairwise_loss.MSE(fixed_image, moving_image),
        pairwise_loss.NCC(fixed_image, moving_image),
        pairwise_loss.LCC(fixed_image, moving_image, sigma=[1, 1], kernel_type="box"),
        pairwise_loss.MI(fixed_image, moving_image, bins=16, sigma=2, spatial_samples=1.0),
        pairwise_loss.NGF(fixed_image, moving_image),
        pairwise_loss.SSIM(fixed_image, moving_image, sigma=[1, 1], dim=2, kernel_type="box"),
    ]

    for loss in losses:
        value = loss(zero_disp)
        assert th.isfinite(value).item(), loss.name

    warped = al.transformation.utils.warp_image(shaded_image, zero_disp)
    assert warped.image.shape == shaded_image.image.shape


def test_masked_similarity_losses_accept_masks_in_torch_2_x(image_triplet):
    fixed_image, moving_image, _ = image_triplet
    zero_disp = th.zeros(*fixed_image.size, len(fixed_image.size), dtype=th.float32)

    mask_tensor = th.ones_like(fixed_image.image)
    mask_tensor[..., :4, :4] = 0

    fixed_mask = al.Image(mask_tensor.squeeze(), fixed_image.size, fixed_image.spacing, fixed_image.origin)
    moving_mask = al.Image(mask_tensor.squeeze(), moving_image.size, moving_image.spacing, moving_image.origin)

    losses = [
        pairwise_loss.LCC(
            fixed_image,
            moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            sigma=[1, 1],
        ),
        pairwise_loss.SSIM(
            fixed_image,
            moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            sigma=[1, 1],
            dim=2,
        ),
    ]

    for loss in losses:
        value = loss(zero_disp)
        assert th.isfinite(value).item(), loss.name


def test_ssim_supports_3d_images():
    fixed_tensor = th.zeros(8, 8, 8, dtype=th.float32)
    moving_tensor = th.zeros(8, 8, 8, dtype=th.float32)
    fixed_tensor[2:6, 2:6, 2:6] = 1.0
    moving_tensor[1:5, 2:6, 2:6] = 1.0

    fixed_image = al.Image(fixed_tensor, fixed_tensor.shape, [1, 1, 1], [0, 0, 0])
    moving_image = al.Image(moving_tensor, moving_tensor.shape, [1, 1, 1], [0, 0, 0])
    zero_disp = th.zeros(*fixed_image.size, len(fixed_image.size), dtype=th.float32)

    loss = pairwise_loss.SSIM(fixed_image, moving_image, sigma=[1, 1, 1], dim=3)
    value = loss(zero_disp)

    assert th.isfinite(value).item()


def test_pairwise_losses_support_3d_images():
    fixed_tensor = th.zeros(8, 8, 8, dtype=th.float32)
    moving_tensor = th.zeros(8, 8, 8, dtype=th.float32)
    fixed_tensor[2:6, 2:6, 2:6] = 1.0
    moving_tensor[1:5, 2:6, 2:6] = 1.0

    fixed_image = al.Image(fixed_tensor, fixed_tensor.shape, [1, 1, 1], [0, 0, 0])
    moving_image = al.Image(moving_tensor, moving_tensor.shape, [1, 1, 1], [0, 0, 0])
    zero_disp = th.zeros(*fixed_image.size, len(fixed_image.size), dtype=th.float32)

    losses = [
        pairwise_loss.MSE(fixed_image, moving_image),
        pairwise_loss.NCC(fixed_image, moving_image),
        pairwise_loss.LCC(fixed_image, moving_image, sigma=[1, 1, 1]),
        pairwise_loss.MI(fixed_image, moving_image, bins=8, sigma=1, spatial_samples=1.0),
        pairwise_loss.NGF(fixed_image, moving_image),
        pairwise_loss.SSIM(fixed_image, moving_image, sigma=[1, 1, 1], dim=3),
    ]

    for loss in losses:
        value = loss(zero_disp)
        assert th.isfinite(value).item(), loss.name


def test_demons_regularisers_and_short_registration_smoke_test(image_triplet):
    fixed_image, moving_image, _ = image_triplet

    transformation = al.transformation.pairwise.NonParametricTransformation(
        moving_image.size,
        dtype=th.float32,
        device=th.device("cpu"),
        diffeomorphic=True,
    )
    parameter = next(transformation.parameters())
    with th.no_grad():
        parameter.normal_(mean=0.0, std=1e-3)

    demons_reg.GaussianRegulariser(
        moving_image.spacing, sigma=[1, 1], dtype=th.float32, device=th.device("cpu")
    ).regularise(transformation.parameters())

    edge_updater = demons_reg.EdgeUpdaterIntensities(moving_image.spacing, fixed_image.image, scale=1)
    demons_reg.GraphDiffusionRegulariser(
        moving_image.size,
        moving_image.spacing,
        edge_updater,
        phi=1,
        dtype=th.float32,
        device=th.device("cpu"),
    ).regularise(transformation.parameters())

    edge_updater_disp = demons_reg.EdgeUpdaterDisplacementIntensities(moving_image.spacing, fixed_image.image)
    demons_reg.GraphDiffusionRegulariser(
        moving_image.size,
        moving_image.spacing,
        edge_updater_disp,
        phi=1,
        dtype=th.float32,
        device=th.device("cpu"),
    ).regularise(transformation.parameters())

    assert th.isfinite(parameter).all().item()

    transformation = al.transformation.pairwise.NonParametricTransformation(
        moving_image.size,
        dtype=th.float32,
        device=th.device("cpu"),
        diffeomorphic=True,
    )
    image_loss = pairwise_loss.MSE(fixed_image, moving_image)
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

    registration = al.PairwiseRegistration(verbose=False)
    registration.set_transformation(transformation)
    registration.set_image_loss([image_loss])
    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(5)

    initial_loss = image_loss(transformation()).item()
    registration.start()
    final_loss = image_loss(transformation()).item()

    assert final_loss <= initial_loss + 1e-6


def test_remove_bed_filter_executes_with_binary_morphological_closing():
    image_array = -1024 * th.ones(16, 16, dtype=th.float32).numpy()
    image_array[4:12, 4:12] = 100

    image = al.image_from_numpy(image_array, [1, 1], [0, 0])
    filtered_image, body_mask = remove_bed_filter(image, cropping=False)

    assert filtered_image.image.shape == image.image.shape
    assert body_mask.image.shape == image.image.shape
