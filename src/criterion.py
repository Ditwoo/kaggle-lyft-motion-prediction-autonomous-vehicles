import torch
import numpy as np


def neg_multi_log_likelihood_batch(
    ground_truth: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    avails: torch.Tensor,
) -> torch.Tensor:
    """Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow.
    For more information about it see:
        - https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
        - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        - https://leimao.github.io/blog/LogSumExp/

    Args:
        ground_truth (torch.Tensor): array of shape (bs)x(time)x(2D coords)
        predicted (torch.Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (torch.Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        availability (torch.Tensor): array of shape (bs)x(time) with the availability for each gt timestep

    Returns:
        torch.Tensor: negative log-likelihood for this example, a single float number

    Source:
        https://www.kaggle.com/huanvo/lyft-complete-train-and-prediction-pipeline
    """
    if len(predictions.shape) != 4:
        raise ValueError(f"expected 3D (MxTxC) array for pred, got {predictions.shape}")

    batch_size, num_modes, future_len, num_coords = predictions.shape

    if ground_truth.shape != (batch_size, future_len, num_coords,):
        raise ValueError(
            f"expected 2D (Time x Coords) array for gt, got {ground_truth.shape}"
        )

    if confidences.shape != (batch_size, num_modes,):
        raise ValueError(f"expected 1D (Modes) array for gt, got {confidences.shape}")

    if not torch.allclose(
        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))
    ):
        raise ValueError("confidences should sum to 1!")

    if avails.shape != (batch_size, future_len):
        raise ValueError(f"expected 1D (Time) array for gt, got {avails.shape}")

    # assert all data are valid
    if not torch.isfinite(predictions).all():
        raise ValueError("invalid value found in pred")

    if not torch.isfinite(ground_truth).all():
        raise ValueError("invalid value found in gt")

    if not torch.isfinite(confidences).all():
        raise ValueError("invalid value found in confidences")

    if not torch.isfinite(avails).all():
        raise ValueError("invalid value found in avails")

    # convert to (batch_size, num_trajectories, future_len, num_coords)
    ground_truth = torch.unsqueeze(ground_truth, 1)  # add trajectories
    avails = avails[:, None, :, None]  # add trajectories and cords

    # error (batch_size, num_trajectories, future_len)
    # reduce coords and use availability
    error = torch.sum(((ground_truth - predictions) * avails) ** 2, dim=-1)

    with np.errstate(divide="ignore"):
        # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_trajectories)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_trajectories)
    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)
    error = (
        -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True))
        - max_value
    )  # reduce trajectories

    return torch.mean(error)
