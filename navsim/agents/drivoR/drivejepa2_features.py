"""
DriveJEPA 数据特征构建器
图像处理逻辑完全对齐 Drive-JEPA 参考代码
（/home/dataset-assist-0/yinhongbo/code/Drive-JEPA/navsim_v1/
  navsim/agents/drive_jepa_perception_based/drive_jepa_features.py）

核心逻辑：
  - 使用 cam_f0/cam_l0/cam_r0/cam_b0 四个相机
  - 取当前帧（cameras[-1]）和上一帧（cameras[-2]），共 2 帧
  - 裁剪上下各 28 像素 -> image[28:-28]（保证 2:1 宽高比）
  - cv2.resize 到 (512, 256)（W, H）
  - torchvision.transforms.ToTensor() 转换为 [C, H, W]，归一化在 model.forward 中完成
  - 输出：camera_feature_1/camera_feature_2（4 相机）以及 LSS 所需内外参
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from torchvision import transforms
import torch

cv2.setNumThreads(0)

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from PIL import Image


class DriveJEPA2FeatureBuilder(AbstractFeatureBuilder):
    """
    DriveJEPA2 特征构建器：
    - 使用 4 个相机的当前帧与上一帧
    - 裁剪 + resize 到 (512, 256)
    - ToTensor 输出，归一化延迟到模型 forward 中完成
    """

    CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")

    def __init__(self, config: Dict):
        """
        @param {dict|OmegaConf} config - 来自 drivejepa.yaml 的配置
        """
        self._config = config
        # 图像 resize 目标尺寸（W, H），与 Drive-JEPA 参考代码保持一致
        self._resize = (512, 256)
        # 上下裁剪像素数（保证 2:1 宽高比）
        self._crop_rows = 28

    def get_unique_name(self) -> str:
        """@returns {str} - 唯一标识符"""
        return "drivejepa2_feature"

    def _resize_camera_image(self, image: np.ndarray) -> torch.Tensor:
        image = image[self._crop_rows:-self._crop_rows]
        resized = cv2.resize(image, self._resize)
        return transforms.ToTensor()(resized)

    def _resize_intrinsic(self, cam) -> np.ndarray:
        image_h, image_w = cam.image.shape[:2]
        resize_w, resize_h = self._resize
        cropped_h = image_h - 2 * self._crop_rows

        intrinsic = np.asarray(cam.intrinsics, dtype=np.float32).copy()
        intrinsic[1, 2] -= self._crop_rows
        scale_x = resize_w / image_w
        scale_y = resize_h / cropped_h
        intrinsic[0, 0] *= scale_x
        intrinsic[0, 2] *= scale_x
        intrinsic[1, 1] *= scale_y
        intrinsic[1, 2] *= scale_y
        return intrinsic

    @staticmethod
    def _camera_to_ego_extrinsic(cam) -> np.ndarray:
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = np.asarray(cam.sensor2lidar_rotation, dtype=np.float32)
        extrinsic[:3, 3] = np.asarray(cam.sensor2lidar_translation, dtype=np.float32)
        return extrinsic

    def _get_camera_feature(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        提取 4 相机两帧图像（当前帧 + 上一帧）：
          1. 取 cameras[-1] / cameras[-2] 的 cam_f0/cam_l0/cam_r0/cam_b0
          2. 裁剪：image[28:-28]（去掉上下各 28 行）
          3. cv2.resize 到 (512, 256)（W×H 顺序）
          4. ToTensor 转换为 float32 [C, H, W]，值域 [0,1]

        @param {AgentInput} agent_input - 传感器数据输入
        @returns {Dict[str, Tensor]} - 4 相机图像和 LSS metadata
        """
        current_cameras = agent_input.cameras[-1]
        previous_cameras = agent_input.cameras[-2]

        current_images = []
        previous_images = []
        intrinsics = []
        extrinsics = []

        for camera_name in self.CAMERA_ORDER:
            current_cam = getattr(current_cameras, camera_name)
            previous_cam = getattr(previous_cameras, camera_name)
            if current_cam.image is None or previous_cam.image is None:
                raise ValueError(f"DriveJEPA2 LSS path requires camera {camera_name}")

            current_images.append(self._resize_camera_image(current_cam.image))
            previous_images.append(self._resize_camera_image(previous_cam.image))
            intrinsics.append(torch.from_numpy(self._resize_intrinsic(current_cam)))
            extrinsics.append(torch.from_numpy(self._camera_to_ego_extrinsic(current_cam)))

        return {
            "camera_feature_1": torch.stack(current_images),   # [N_cam, C, H, W]
            "camera_feature_2": torch.stack(previous_images),  # [N_cam, C, H, W]
            "intrinsics": torch.stack(intrinsics),             # [N_cam, 3, 3]
            "extrinsics": torch.stack(extrinsics),             # [N_cam, 4, 4]
            "future_egomotion": torch.zeros(1, 6, dtype=torch.float32),
        }

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        从 AgentInput 提取所有训练特征。

        @param {AgentInput} agent_input - 原始传感器输入
        @returns {Dict[str, Tensor]} - 特征字典
        """
        features = {}

        # 相机特征
        features.update(self._get_camera_feature(agent_input))

        # Ego 状态特征（与 DrivoR 完全相同）
        ego_feature_list = []
        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose = torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
            ego_feature = torch.cat([pose, velocity, acceleration, driving_command], dim=-1)
            ego_feature_list.append(ego_feature)

        features["ego_status"] = torch.stack(ego_feature_list)
        return features


    def _get_lidar_feature(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        计算 LiDAR BEV 直方图特征（与 DrivoRFeatureBuilder 完全相同）。

        @param {AgentInput} agent_input
        @returns {Dict[str, Tensor]} - {"lidar_feature": Tensor}
        """
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        def splat_points(point_cloud):
            xbins = np.linspace(
                self._config.lidar_min_x, self._config.lidar_max_x,
                self._config.lidar_image_size[0] + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y, self._config.lidar_max_y,
                self._config.lidar_image_size[1] + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.lidar_hist_max_per_pixel] = self._config.lidar_hist_max_per_pixel
            return hist / self._config.lidar_hist_max_per_pixel

        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.lidar_max_height]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)

        if self._config.lidar_use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)

        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        features = np.expand_dims(features, axis=0)
        return {"lidar_feature": torch.tensor(features)}
