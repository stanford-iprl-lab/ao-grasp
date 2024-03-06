import torch
import os

from aograsp.models.model_pointscore import Model_PointScore

"""
Helper functions for model
"""

def load_model(
    model_conf_path="aograsp/aograsp_model/conf.pth",
    ckpt_path="aograsp/aograsp_model/770-network.pth",
):
    """Load pointscore model and restore checkpoint"""

    # Load model
    model_conf = torch.load(model_conf_path)
    model = Model_PointScore(model_conf)

    # Check if checkpoint exists
    state_exists = os.path.exists(
        os.path.join(ckpt_path)
    )

    # Load states for network, optimizer, lr_scheduler
    if state_exists:
        print(
            f"\n--------------------------------------------------------------------------------"
        )
        print(
            f"------ Restoring model to {ckpt_path} ----------------"
        )
        print(
            f"--------------------------------------------------------------------------------\n"
        )

        data_to_restore = torch.load(ckpt_path)

        # Remove training-only parameters
        layers_to_remove = []
        for key in data_to_restore:
            if "siamese" in key:
                layers_to_remove.append(key)
        for key in layers_to_remove:
            del data_to_restore[key]

        model.load_state_dict(data_to_restore)
    else:
        raise ValueError("Specified checkpoint cannot be found.")
    
    return model