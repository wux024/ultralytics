# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Sequence, Union, List
import numpy as np
from keypoint_2d_metrics import (keypoint_auc, keypoint_epe, keypoint_mpjpe,
                          keypoint_pck_accuracy)


class BaseMetric:
    def __init__(self) -> None:
        self.results = []

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        raise NotImplementedError

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute metrics based on the processed results.

        Args:
            results (list): The processed results.

        Returns:
            Dict[str, float]: The computed metrics.
        """
        raise NotImplementedError


class CocoMetric(BaseMetric):
    
    pass

class MPJPE(BaseMetric):

    def __init__(self) -> None:
        super().__init__()

    def process(self, data_samples: Sequence[dict]) -> None:

        for data_sample in data_samples:
            # predicted keypoints coordinates, [T, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            if pred_coords.ndim == 4:
                pred_coords = np.squeeze(pred_coords, axis=0)
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [T, K, D]
            gt_coords = gt['lifting_target']
            # ground truth keypoints_visible, [T, K, 1]
            mask = gt['lifting_target_visible'].astype(bool).reshape(
                gt_coords.shape[0], -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:


        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        error_name = self.mode.upper()

        return {
            error_name:
            keypoint_mpjpe(pred_coords, gt_coords, mask)
        }
    
class PCKAccuracy(BaseMetric):

    def __init__(self, thr: float = 0.05) -> None:
        super().__init__()
        self.thr = thr

    def process(self, data_samples: Sequence[dict]) -> None:

        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }
            bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
            bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
            result['bbox_size'] = bbox_size
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metrics = dict()
        norm_size_bbox = np.concatenate(
            [result['bbox_size'] for result in results])
        
        _, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                              self.thr, norm_size_bbox)
        metrics['PCK'] = pck

        return metrics



class AUC(BaseMetric):
    def __init__(self,
                 norm_factor: float = 30,
                 num_thrs: int = 20,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.norm_factor = norm_factor
        self.num_thrs = num_thrs

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:

        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:


        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        auc = keypoint_auc(pred_coords, gt_coords, mask, self.norm_factor,
                           self.num_thrs)

        metrics = dict()
        metrics['AUC'] = auc

        return metrics

class EPE(BaseMetric):

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        epe = keypoint_epe(pred_coords, gt_coords, mask)

        metrics = dict()
        metrics['EPE'] = epe

        return metrics