"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets.camera_only import base_camera_dataset
from opencood.utils import common_utils


class CamIntermediateFusionDataset(base_camera_dataset.BaseCameraDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamIntermediateFusionDataset, self).__init__(params,
                                                           visualize,
                                                           train,
                                                           validate)
        self.visible = params['train_params']['visible']
        fusion_args = params.get('fusion', {}).get('args', {})
        if isinstance(fusion_args, list):
            fusion_args = {}
        self.single_vehicle = bool(fusion_args.get('single_vehicle', False))

    def __getitem__(self, idx):
        # retrieve_base_data 现在返回 list 或 dict
        result = self.get_sample_random(idx)

        # 兼容单帧（queue_length=1）
        if isinstance(result, dict):
            result = [result]   # 包装成列表统一处理

        # result: List[processed_data_dict]，长度 = queue_length
        # 当前帧（用于 GT 提取）= result[0]
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = OrderedDict()

        # ── 从当前帧获取 ego 信息和 GT ──────────────────────────
        cur_frame = result[0]
        ego_id = -999
        ego_lidar_pose = []
        for cav_id, cav_content in cur_frame.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(cur_frame.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        valid_cav_ids = []
        for cav_id, cav_content in cur_frame.items():
            distance = common_utils.cav_distance_cal(cav_content, ego_lidar_pose)
            if distance <= opencood.data_utils.datasets.COM_RANGE:
                valid_cav_ids.append(cav_id)
        if self.single_vehicle:
            valid_cav_ids = [ego_id]

        pairwise_t_matrix = self.get_pairwise_transformation(
            cur_frame, self.params['train_params']['max_cav'])

        # ── 遍历时间序列，收集图像和位姿偏移 ─────────────────────
        camera_data_seq = []
        camera_intrinsic_seq = []
        camera_extrinsic_seq = []
        prev_pose_offset_seq = []

        for frame_data in result:
            cam_data_t, cam_int_t, cam_ext_t, pose_off_t = [], [], [], []
            for cav_id in valid_cav_ids:
                cav_content = frame_data.get(cav_id, None)
                if cav_content is not None:
                    distance = common_utils.cav_distance_cal(cav_content, ego_lidar_pose)
                if cav_content is None or distance > opencood.data_utils.datasets.COM_RANGE:
                    cav_content = cur_frame[cav_id]
                    pose_offset = np.eye(4)
                else:
                    pose_offset = cav_content.get('prev_pose_offset', np.eye(4))
                proc = self.get_single_cav(cav_content)
                cam_data_t.append(proc['camera']['data'])
                cam_int_t.append(proc['camera']['intrinsic'])
                cam_ext_t.append(proc['camera']['extrinsic'])
                pose_off_t.append(pose_offset)

            camera_data_seq.append(np.stack(cam_data_t))
            camera_intrinsic_seq.append(np.stack(cam_int_t))
            camera_extrinsic_seq.append(np.stack(cam_ext_t))
            prev_pose_offset_seq.append(np.stack(pose_off_t))

        camera_data      = np.stack(camera_data_seq)       # [T, L, M, H, W, C]
        camera_intrinsic = np.stack(camera_intrinsic_seq)  # [T, L, M, 3, 3]
        camera_extrinsic = np.stack(camera_extrinsic_seq)  # [T, L, M, 4, 4]
        prev_pose_offset = np.stack(prev_pose_offset_seq)  # [T, L, 4, 4]

        # GT 从当前帧提取（仅 ego）
        cur_ego_proc = self.get_single_cav(cur_frame[ego_id])
        gt_dynamic = cur_ego_proc['gt']['dynamic_bev']
        gt_static  = cur_ego_proc['gt']['static_bev']

        # transformation_matrix 从当前帧提取
        transformation_matrix = []
        for cav_id in valid_cav_ids:
            transformation_matrix.append(cur_frame[cav_id]['params']['transformation_matrix'])
        
        transformation_matrix = np.stack(transformation_matrix)
        padding_eye = np.tile(
            np.eye(4)[None],
            (self.max_cav - len(transformation_matrix), 1, 1)
        )
        transformation_matrix = np.concatenate(
            [transformation_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update({
            'transformation_matrix': transformation_matrix,
            'pairwise_t_matrix':     pairwise_t_matrix,
            'camera_data':           camera_data,        
            'camera_intrinsic':      camera_intrinsic,   
            'camera_extrinsic':      camera_extrinsic,   
            'prev_pose_offset':      prev_pose_offset,   
            'gt_dynamic': np.expand_dims(gt_dynamic, 0),
            'gt_static':  np.expand_dims(gt_static, 0),
        })
        return processed_data_dict

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()

        # update the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({
            'transformation_matrix': transformation_matrix
        })

        # for intermediate fusion, we only need ego's gt
        if selected_cav_base['ego'] and 'bev_static.png' in selected_cav_base:
            # process the groundtruth
            if self.visible:
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['bev_visibility_corp.png'])
            else:
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['bev_dynamic.png'])
            road_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['bev_static.png'])
            lane_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['bev_lane.png'])
            static_bev = self.post_processor.merge_label(road_bev, lane_bev)

            gt_dict = {'static_bev': static_bev,
                       'dynamic_bev': dynamic_bev}

            selected_cav_processed.update({'gt': gt_dict})

        all_camera_data = []
        all_camera_origin = []
        all_camera_intrinsic = []
        all_camera_extrinsic = []

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_camera_origin.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)
            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)

        camera_dict = {
            'origin_data': np.stack(all_camera_origin),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic)
        }

        selected_cav_processed.update({'camera': camera_dict})

        return selected_cav_processed

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        if not self.train:
            assert len(batch) == 1

        output_dict = {'ego': {}}

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cam_intrinsic_all_batch = []
        prev_pose_offset_all_batch = []

        gt_static_all_batch = []
        gt_dynamic_all_batch = []

        transformation_matrix_all_batch = []
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']

            camera_data = ego_dict['camera_data']
            camera_intrinsic = ego_dict['camera_intrinsic']
            camera_extrinsic = ego_dict['camera_extrinsic']
            prev_pose_offset = ego_dict['prev_pose_offset']

            assert camera_data.shape[1] == \
                   camera_intrinsic.shape[1] == \
                   camera_extrinsic.shape[1] == \
                   prev_pose_offset.shape[1]

            record_len.append(camera_data.shape[1])

            cam_rgb_all_batch.append(camera_data)
            cam_intrinsic_all_batch.append(camera_intrinsic)
            cam_to_ego_all_batch.append(camera_extrinsic)
            prev_pose_offset_all_batch.append(prev_pose_offset)

            # ground truth
            gt_dynamic_all_batch.append(ego_dict['gt_dynamic'])
            gt_static_all_batch.append(ego_dict['gt_static'])

            # transformation matrix
            transformation_matrix_all_batch.append(
                ego_dict['transformation_matrix'])
            # pairwise matrix
            pairwise_t_matrix_all_batch.append(ego_dict['pairwise_t_matrix'])

        # (T, B*L, M, H, W, C) 或者保持 (B*L, T, M, H, W, C)
        # 这里为了兼容性，按第一维度展开 B*L，即 concat(axis=1) => 保持 batch*cav 维度，然后可能是 T
        # np.concatenate(axis=1) => [T, B*L, M, ...] 但是 camera_data 是 [T, L, M, H, W, C]
        # concatenate(..., axis=1) 将产生 [T, \sum L, M, H, W, C]
        cam_rgb_all_batch = torch.from_numpy(
            np.concatenate(cam_rgb_all_batch, axis=1)).float()
        cam_intrinsic_all_batch = torch.from_numpy(
            np.concatenate(cam_intrinsic_all_batch, axis=1)).float()
        cam_to_ego_all_batch = torch.from_numpy(
            np.concatenate(cam_to_ego_all_batch, axis=1)).float()
        
        # 对于 [T, L, 4, 4] -> np.concatenate(axis=1) -> [T, \sum L, 4, 4]
        prev_pose_offset_all_batch = torch.from_numpy(
            np.concatenate(prev_pose_offset_all_batch, axis=1)).float()
            
        # 补充为了兼容单帧：如果需要模型期望 shape 为 [B*L, T, ...]。这里 [T, B*L, ...] 可以通过 permute 变成 [B*L, T, ...]
        # [B*L, T, M, H, W, C]
        cam_rgb_all_batch = cam_rgb_all_batch.permute(1, 0, 2, 3, 4, 5)
        # FAX/CVT geometry always uses the current-frame calibration. Keep the
        # legacy FAX shape [N, 1, M, ...] no matter whether queue_length is 1 or 2.
        cam_intrinsic_all_batch = cam_intrinsic_all_batch[0].unsqueeze(1)
        cam_to_ego_all_batch = cam_to_ego_all_batch[0].unsqueeze(1)
        prev_pose_offset_all_batch = prev_pose_offset_all_batch.permute(1, 0, 2, 3)

        # (B,)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # (B, 1, H, W)
        gt_static_all_batch = \
            torch.from_numpy(np.concatenate(gt_static_all_batch, axis=0)).long()
        gt_dynamic_all_batch = \
            torch.from_numpy(np.concatenate(gt_dynamic_all_batch, axis=0)).long()
        if gt_static_all_batch.dim() == 3:
            gt_static_all_batch = gt_static_all_batch.unsqueeze(1)
        if gt_dynamic_all_batch.dim() == 3:
            gt_dynamic_all_batch = gt_dynamic_all_batch.unsqueeze(1)

        # (B,max_cav,4,4)
        transformation_matrix_all_batch = \
            torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()
        pairwise_t_matrix_all_batch = \
            torch.from_numpy(np.stack(pairwise_t_matrix_all_batch)).float()

        # convert numpy arrays to torch tensor
        output_dict['ego'].update({
            'inputs': cam_rgb_all_batch,
            'extrinsic': cam_to_ego_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'prev_pose_offset': prev_pose_offset_all_batch,
            'gt_static': gt_static_all_batch,
            'gt_dynamic': gt_dynamic_all_batch,
            'transformation_matrix': transformation_matrix_all_batch,
            'pairwise_t_matrix': pairwise_t_matrix_all_batch,
            'record_len': record_len
        })

        return output_dict

    def post_process(self, batch_dict, output_dict):
        output_dict = self.post_processor.post_process(batch_dict,
                                                       output_dict)

        return output_dict
