import torch
import torch.nn as nn


class ModelWithConfidence(nn.Module):
    def __init__(self, backbone, future_num_frames, num_trajectories=3):
        super().__init__()
        self.backbone = backbone
        self.future_num_frames = future_num_frames
        self.num_trajectories = num_trajectories
        self.num_preds = 2 * future_num_frames * num_trajectories

    def forward(self, *batch):
        """
        Args:
            batch (torch.Tensor): input data,
                should have shape - (batch size)x(time)x(height)x(width)

        Returns:
            predictions with shape - (batch size)x(trajectories)x(time)x(2D coordinates)
            and
            confidences with shape - (batch size)x(trajectories)
        """
        x = self.backbone(*batch)

        bs, _ = x.shape
        preds, conf = torch.split(x, self.num_preds, dim=1)
        preds = preds.view(bs, self.num_trajectories, self.future_num_frames, 2)

        assert conf.shape == (bs, self.num_trajectories)
        conf = torch.softmax(conf, dim=1)
        return preds, conf


class SegmentationModelWithConfidence(ModelWithConfidence):
    def forward(self, *batch):
        """
        Args:
            batch (torch.Tensor): input data,
                should have shape - (batch size)x(time)x(height)x(width)

        Returns:
            predictions with shape - (batch size)x(trajectories)x(time)x(2D coordinates)
            and
            confidences with shape - (batch size)x(trajectories)
        """
        x, mask = self.backbone(*batch)

        bs, _ = x.shape
        preds, conf = torch.split(x, self.num_preds, dim=1)
        preds = preds.view(bs, self.num_trajectories, self.future_num_frames, 2)

        assert conf.shape == (bs, self.num_trajectories)
        conf = torch.softmax(conf, dim=1)
        return preds, conf, mask
