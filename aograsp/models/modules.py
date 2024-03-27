import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        radius = self.hparams["pn_radius"]
        nsample = self.hparams["pn_nsample"]

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=radius[0],
                nsample=nsample[0],
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=radius[1],
                nsample=nsample[1],
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=radius[2],
                nsample=nsample[2],
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=radius[3],
                nsample=nsample[3],
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams["feat_dim"], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams["feat_dim"]),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
        Forward pass of the network

        Parameters
        ----------
        pointcloud: Variable(torch.cuda.FloatTensor)
            (B, N, 3 + input_channels) tensor
            Point cloud to run predicts on
            Each point in the point-cloud MUST
            be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class PointScore(nn.Module):
    def __init__(self, feat_dim, dropout_p=0.0, k=128, weight_loss=True):
        super(PointScore, self).__init__()

        self.mlp1 = nn.Linear(feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

        self.MSELoss = nn.MSELoss(reduction="none")
        self.K = k
        self.weight_loss = weight_loss

    # feats B x F
    # output: B
    def forward(self, feats):
        net = self.dropout(F.leaky_relu(self.mlp1(feats)))
        net = torch.sigmoid(self.mlp2(net))
        return net

    def get_topk_mse_loss(self, pred_logits, heatmap):
        loss = self.MSELoss(pred_logits, heatmap.float())
        weights = torch.exp(heatmap)
        if self.weight_loss:
            loss = loss * weights
        total_loss = torch.topk(loss, self.K, dim=1).values
        return total_loss

    def get_all_pts_err(self, pred_heatmap, gt_heatmap):
        """
        Compute error between predicted and gt labels of all points
        Args:
            pred_labels: predicted labels [B, N]
            gt_labels: ground truth heatmap labels [B, N]
        """
        err = self.MSELoss(pred_heatmap, gt_heatmap).mean()

        return err
