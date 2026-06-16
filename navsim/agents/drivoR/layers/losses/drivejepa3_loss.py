from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss, _agent_loss


class DriveJEPA3Loss(DrivoRLoss):
    """DrivoR loss with PAD/VeteranAD-style decoder trajectory supervision."""

    def forward(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config, scoring_function=None):
        proposals = pred.get("refined_proposals", pred["proposals"])
        proposal_list = list(pred.get("proposal_list", [proposals]))

        target_trajectory = targets["trajectory"]
        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = scoring_function(
            targets, proposals, test=False
        )

        target_trajectory_long = targets.get("trajectory_long")
        trajectory_loss = 0
        min_loss_list = []
        inter_loss_list = []
        for proposals_i in proposal_list:
            min_loss = torch.linalg.norm(
                proposals_i - target_trajectory[:, None],
                dim=-1,
                ord=1,
            ).mean(-1).amin(1).mean()
            if target_trajectory_long is not None:
                min_loss = min_loss + torch.linalg.norm(
                    proposals_i - target_trajectory_long[:, None],
                    dim=-1,
                    ord=1,
                ).mean(-1).amin(1).mean()

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = self.prev_weight * trajectory_loss + min_loss + inter_loss * self.inter_weight
            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        l2_distance = -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5

        if "pred_logit" in pred:
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],
                pred["pred_logit2"],
                pred["pred_agents_states"],
                pred["pred_area_logit"],
                target_scores,
                gt_states,
                gt_valid,
                gt_ego_areas,
                l2_distance.detach(),
            )
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(
                targets,
                pred,
                self.agent_class_weight,
                self.agent_box_weight,
            )
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
            self.trajectory_weight * trajectory_loss
            + self.final_score_weight * final_score_loss
            + self.pred_ce_weight * pred_ce_loss
            + self.pred_l1_weight * pred_l1_loss
            + self.pred_area_weight * pred_area_loss
            + self.agent_class_weight * agent_class_loss
            + self.agent_box_weight * agent_box_loss
            + self.bev_semantic_weight * bev_semantic_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()
        da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss = sub_score_loss

        return {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            "da_loss": da_loss,
            "ttc_loss": ttc_loss,
            "noc_loss": noc_loss,
            "progress_loss": progress_loss,
            "ddc_loss": ddc_loss,
            "comfort_loss": comfort_loss,
            "final_score_loss": final_score_loss,
            "pred_ce_loss": pred_ce_loss,
            "pred_l1_loss": pred_l1_loss,
            "pred_area_loss": pred_area_loss,
            "agent_class_loss": agent_class_loss,
            "agent_box_loss": agent_box_loss,
            "bev_semantic_loss": bev_semantic_loss,
            "inter_loss0": inter_loss0,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            "min_loss": min_loss_list[-1],
            "score": score,
            "best_score": best_score,
        }
