"""
    Pointscore model
"""

import torch
import torch.nn as nn
import os

from aograsp.models.modules import PointNet2SemSegSSG, PointScore


class Model_PointScore(nn.Module):
    def __init__(self, conf):
        super(Model_PointScore, self).__init__()
        self.conf = conf
        self.feat_dim = conf.feat_dim

        self.pointnet2 = PointNet2SemSegSSG(
            {
                "feat_dim": conf.feat_dim,
                "pn_radius": conf.pn_radius,
                "pn_nsample": conf.pn_nsample,
            }
        )

        self.point_score = PointScore(
            self.feat_dim,
            dropout_p=self.conf.dropout_p,
            k=conf.pointscore_k,
            weight_loss=conf.pointscore_weight_loss,
        )

    def forward(self, input_dict):
        pcs = input_dict["pcs"]
        batch_size = pcs.shape[0]
        pcs = pcs.repeat(1, 1, 2)

        # push through PointNet++
        whole_feats = self.pointnet2(pcs)
        net = whole_feats.permute(0, 2, 1)
        point_score_heatmap = self.point_score(net).reshape(batch_size, -1)
        # [B, N, 1] --> [B, N]

        output_dict = {
            "whole_feats": net,
            "point_score_heatmap": point_score_heatmap,
        }
        return output_dict

    def loss(self, output_dict, input_dict, gt_labels):
        pred_heatmap = output_dict["point_score_heatmap"]
        gt_heatmap = input_dict["heatmap"]

        # Compute point score loss
        point_score_loss = self.point_score.get_topk_mse_loss(pred_heatmap, gt_heatmap)
        point_score_loss = point_score_loss.mean()
        return {
            "total_loss": point_score_loss,
            "point_score_loss": point_score_loss,
        }

    def test(self, input_dict, gt_labels):
        """Run inference with model and get error between predicted and gt heatmaps"""

        pcs = input_dict["pcs"]
        batch_size = pcs.shape[0]
        pcs = pcs.repeat(1, 1, 2)

        with torch.no_grad():
            whole_feats = self.pointnet2(pcs)
            net = whole_feats.permute(0, 2, 1)
            pred_point_score_map = self.point_score(net).reshape(batch_size, -1)

            output_dict = {
                "whole_feats": net,
                "point_score_heatmap": pred_point_score_map,
            }

        return output_dict
