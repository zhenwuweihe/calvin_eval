import logging
from pathlib import Path

from calvin_agent.models.play_lmp import PlayLMP
from calvin_agent.utils.utils import add_text, get_all_checkpoints, get_last_checkpoint
import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning import seed_everything
import torch

logger = logging.getLogger(__name__)


def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device("cuda:0")
    if env is None:
        env = hydra.utils.instantiate(cfg.callbacks.rollout.env_cfg, dataset, device, show_gui=False)

    print(f"Loading model from {checkpoint}")
    model = PlayLMP.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class DefaultLangEmbeddings:
    def __init__(self, dataset_path):
        embeddings = np.load(
            Path(dataset_path) / "validation/lang_annotations/embeddings.npy", allow_pickle=True
        ).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = torch.device("cuda:0")

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0)}


def get_eval_env_state():
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    scene_obs = np.array(
        [
            -1.04781977e-06,
            0.00000000e00,
            0.00000000e00,
            2.04697370e-17,
            0.00000000e00,
            0.00000000e00,
            5.00000896e-02,
            -1.20000177e-01,
            4.59990009e-01,
            -2.33735409e-08,
            -1.62393945e-08,
            1.57000011e00,
            2.29995412e-01,
            -1.19995140e-01,
            4.59990010e-01,
            5.67408891e-09,
            2.11727851e-08,
            -1.52919521e-06,
            1.00341633e-01,
            8.52660710e-02,
            4.60981664e-01,
            -2.84972036e-04,
            1.40530077e-04,
            1.55882276e00,
        ]
    )
    return robot_obs, scene_obs


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        f"Average Success Rate {mod} = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


def format_sftp_path(cfg):
    """
    When using network mount from nautilus, format path
    """
    if cfg.train_folder.startswith("sftp"):
        cfg.train_folder = "/run/user/9984/gvfs/sftp:host=" + cfg.train_folder[7:]
