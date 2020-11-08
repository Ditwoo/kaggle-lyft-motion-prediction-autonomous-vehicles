import os

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE

from l5kit.data.filter import filter_agents_by_labels, filter_agents_by_track_id
from l5kit.geometry.transform import yaw_as_rotation33, transform_points, rotation33_as_yaw
from l5kit.rasterization.rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from l5kit.rasterization.box_rasterizer import get_ego_as_agent, draw_boxes

from tqdm.auto import tqdm
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer


from typing import List, Optional

import numpy as np

from l5kit.rasterization.box_rasterizer import BoxRasterizer
from l5kit.rasterization.rasterizer import Rasterizer
from l5kit.rasterization.render_context import RenderContext
from l5kit.rasterization.semantic_rasterizer import SemanticRasterizer



class BoxRasterizerCompressed(BoxRasterizer):


    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # all frames are drawn relative to this one"
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history

        # TODO: Change shape
        # out_shape = (self.raster_size[1], self.raster_size[0], self.history_num_frames + 1)
        out_shape = (self.raster_size[1], self.raster_size[0], 1)
        # agents_images = np.zeros(out_shape, dtype=np.uint8)
        # ego_images = np.zeros(out_shape, dtype=np.uint8)
        agents_images = np.zeros(out_shape, dtype=np.float32)
        ego_images = np.zeros(out_shape, dtype=np.float32)

        n = history_frames.size
        weights = (np.logspace(0, 1, n, base=15) / n)[::-1][:, None, None]

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)

            # agents_images[..., i] = agents_image
            # ego_images[..., i] = ego_image
            # print(i)
            if i == 0:
                heads = np.expand_dims(agents_image + ego_image, axis=2)

            agents_images[..., 0] += (agents_image * weights[i]).astype(np.float32)
            ego_images[..., 0] += (ego_image * weights[i]).astype(np.float32)



        # ego_heads =
        # agents_images = np.expand_dims((weights * agents_images).max(2), axis=2)
        # ego_images = np.expand_dims((weights * ego_images).max(2), axis=2)

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images, heads), -1)

        # return out_im.astype(np.float32) / 255
        return out_im.astype(np.float32) / np.amax(out_im)

class SemBoxRasterizerCompressed(Rasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class
    """

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        semantic_map_path: str,
        world_to_ecef: np.ndarray,
    ):
        super(SemBoxRasterizerCompressed, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

        self.box_rast = BoxRasterizerCompressed(render_context, filter_agents_threshold, history_num_frames)
        # self.box_rast = BoxRasterizer(render_context, filter_agents_threshold, history_num_frames)
        self.sat_rast = SemanticRasterizer(render_context, semantic_map_path, world_to_ecef)

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        im_out_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        im_out_sat = self.sat_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)

        return np.concatenate([im_out_box, im_out_sat], -1)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        im_out_box = self.box_rast.to_rgb(in_im[..., :-3], **kwargs)
        im_out_sat = self.sat_rast.to_rgb(in_im[..., -3:], **kwargs)
        # merge the two together
        mask_box = np.any(im_out_box > 0, -1)
        im_out_sat[mask_box] = im_out_box[mask_box]
        return im_out_sat
