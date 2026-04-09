# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.5",
#     "torchvision",
#     "safetensors",
#     "tokenizers",
#     "pandas",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "einops",
#     "diffusers",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../.." }
# ///
"""GR00T-Base (N1.5) model server.

Uses the LLVM/susser-tod GR00T-Base model for inference.  Requires a local
clone of the LLVM repo (``--llvm_dir``) with the ``external/susser-tod``
submodule initialised.

Model code is imported directly from ``llvm_dir/external/susser-tod/src``
and ``llvm_dir/src`` at runtime.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms import InterpolationMode

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

# Default llvm_dir: external/LLVM submodule relative to repo root
_DEFAULT_LLVM_DIR = str(Path(__file__).resolve().parents[3] / "external" / "LLVM")

# Camera keys matching RoboTwin benchmark output order
_CAM_KEYS = ["head_camera", "left_camera", "right_camera"]

_EMBODIMENT_TAG_TO_ID = {
    "robotwin/agilex_clean": 31,
    "robotwin/agilex_random": 31,
}


# ---------------------------------------------------------------------------
# Environment setup: sys.path + dependency stubs
# ---------------------------------------------------------------------------

_llvm_env_ready = False


def _stub(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:
    """Register a module stub with valid ``__spec__`` if not already present."""
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m
    elif sys.modules[name].__spec__ is None:
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, None)
    return sys.modules[name]


def _stub_pkg(name: str, real_dir: str) -> types.ModuleType:
    """Register a package stub whose ``__path__`` points to a real directory.

    Bypasses the package's ``__init__.py`` while still allowing Python to
    locate submodules on the file system.
    """
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [real_dir]
        m.__package__ = name
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m
    return sys.modules[name]


def _import_file(mod_name: str, path: str) -> types.ModuleType:
    """Import a single .py file by path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_llvm_environment(llvm_dir: str) -> None:
    """Add susser-tod and llvm source trees to ``sys.path`` and register stubs.

    Mirrors the setup in the reference ``deploy_policy.py`` so that
    ``susser_tod`` and ``llvm`` packages are importable without pulling in
    heavy training-only dependencies (flash_attn, lakestream, prospec, …).
    """
    global _llvm_env_ready  # noqa: PLW0603
    if _llvm_env_ready:
        return

    susser_src = os.path.join(llvm_dir, "external", "susser-tod", "src")
    llvm_src = os.path.join(llvm_dir, "src")
    for p in [susser_src, llvm_src]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Source directory not found: {p}")
        if p not in sys.path:
            sys.path.insert(0, p)

    # flash_attn stubs (patched by susser_tod/__init__, checked by diffusers)
    _stub("flash_attn_interface")
    _stub("flash_attn_3")
    _stub("flash_attn_3.flash_attn_interface")

    # Training-only libraries
    _stub("lakestream", {"Batch": dict})
    _stub("prospec", {"BaseSpec": object, "Narrow": lambda *a, **kw: None})

    # Skip heavy __init__ files that import all model variants
    _stub_pkg("susser_tod.models", os.path.join(susser_src, "susser_tod/models"))
    _stub_pkg(
        "susser_tod.models.gr00t_n1_5",
        os.path.join(susser_src, "susser_tod/models/gr00t_n1_5"),
    )

    # llvm.specs stubs (pydantic-based spec system, training only)
    _stub_pkg("llvm.specs", os.path.join(llvm_src, "llvm/specs"))
    gr00t_specs = _stub("llvm.specs.gr00t")
    for nm in [
        "GR00TDataSpec",
        "GR00TDatasetSpec",
        "EagleBatchTransformSpec",
        "GR00TStateTransformSpec",
        "GR00TVideoTransformSpec",
        "GR00TActionTransformSpec",
        "GR00TLanguageTransformSpec",
        "GR00TPVIVideoPreTransformSpec",
        "GR00TPVIVideoPostTransformSpec",
        "VJEPAPreprocessTransformSpec",
        "VideoSliceTransformSpec",
        "StateSliceTransformSpec",
    ]:
        setattr(gr00t_specs, nm, type(nm, (), {}))

    # Trigger susser_tod's flash_attn compat patch
    import susser_tod  # noqa: F401

    _llvm_env_ready = True


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_model_config(config_path: str) -> Any:
    """Construct ``GR00T_N1_5_BaseConfig`` from a ``config.json`` file."""
    from susser_tod.models.gr00t_n1_5.modeling import GR00T_N1_5_BaseConfig
    from susser_tod.models.gr00t_n1_5.modeling.action_head.action_head_config import ActionHeadConfig
    from susser_tod.models.gr00t_n1_5.modeling.action_head.dit import DiTConfig, SelfAttentionConfig
    from susser_tod.models.gr00t_n1_5.modeling.backbone.backbone_config import BackBoneConfig
    from susser_tod.models.gr00t_n1_5.modeling.backbone.eagle2 import Eagle2Config
    from susser_tod.models.gr00t_n1_5.modeling.backbone.eagle2.language import Qwen3Config
    from susser_tod.models.gr00t_n1_5.modeling.backbone.eagle2.vision import SiglipVisionConfig

    with open(config_path) as f:
        data = json.load(f)

    bb = data.get("backbone_cfg", {})
    e2 = bb.get("eagle2_config", {})
    vis = e2.get("vision_config", {})
    lang = e2.get("language_config", {})
    ah = data.get("action_head_cfg", {})
    dit = ah.get("diffusion_model_cfg", {})
    vla = ah.get("vl_self_attention_cfg", {})

    vision_cfg = SiglipVisionConfig(
        hidden_size=vis.get("hidden_size", 1152),
        intermediate_size=vis.get("intermediate_size", 4304),
        num_hidden_layers=vis.get("num_hidden_layers", 27),
        num_attention_heads=vis.get("num_attention_heads", 16),
        num_channels=vis.get("num_channels", 3),
        image_size=vis.get("image_size", 224),
        patch_size=vis.get("patch_size", 14),
        hidden_act=vis.get("hidden_act", "gelu_pytorch_tanh"),
        layer_norm_eps=vis.get("layer_norm_eps", 1e-6),
        attention_dropout=vis.get("attention_dropout", 0.0),
    )
    language_cfg = Qwen3Config(
        hidden_size=lang.get("hidden_size", 2048),
        intermediate_size=lang.get("intermediate_size", 6144),
        num_hidden_layers=lang.get("num_hidden_layers", 28),
        num_attention_heads=lang.get("num_attention_heads", 16),
        num_key_value_heads=lang.get("num_key_value_heads", 8),
        head_dim=lang.get("head_dim", 128),
        vocab_size=lang.get("vocab_size", 151680),
        max_position_embeddings=lang.get("max_position_embeddings", 40960),
        rope_theta=lang.get("rope_theta", 1000000.0),
        rope_scaling=lang.get("rope_scaling", None),
        hidden_act=lang.get("hidden_act", "silu"),
        rms_norm_eps=lang.get("rms_norm_eps", 1e-6),
        initializer_range=lang.get("initializer_range", 0.02),
        attention_bias=lang.get("attention_bias", False),
        attention_dropout=lang.get("attention_dropout", 0.0),
        max_window_layers=lang.get("max_window_layers", 28),
        sliding_window=lang.get("sliding_window", None),
        use_sliding_window=lang.get("use_sliding_window", False),
    )
    eagle2_cfg = Eagle2Config(
        vision_config=vision_cfg,
        language_config=language_cfg,
        downsample_ratio=e2.get("downsample_ratio", 0.5),
        select_layer=e2.get("select_layer", -1),
        force_image_size=e2.get("force_image_size", 224),
        use_pixel_shuffle=e2.get("use_pixel_shuffle", False),
        mlp_connector_layers=e2.get("mlp_connector_layers", 1),
        image_token_index=e2.get("image_token_index", 151669),
        mlp_checkpoint=e2.get("mlp_checkpoint", False),
        initializer_range=e2.get("initializer_range", 0.02),
    )
    backbone_cfg = BackBoneConfig(
        tune_llm=bb.get("tune_llm", False),
        tune_visual=bb.get("tune_visual", True),
        project_to_dim=bb.get("project_to_dim", None),
        select_layer=bb.get("select_layer", 12),
        eagle2_config=eagle2_cfg,
    )
    dit_cfg = DiTConfig(
        num_attention_heads=dit.get("num_attention_heads", 32),
        attention_head_dim=dit.get("attention_head_dim", 48),
        num_layers=dit.get("num_layers", 16),
        cross_attention_dim=dit.get("cross_attention_dim", 2048),
        dropout=dit.get("dropout", 0.2),
        final_dropout=dit.get("final_dropout", True),
        interleave_self_attention=dit.get("interleave_self_attention", True),
        norm_type=dit.get("norm_type", "ada_norm"),
        output_dim=dit.get("output_dim", 1024),
        positional_embeddings=dit.get("positional_embeddings", None),
        attention_bias=dit.get("attention_bias", True),
        activation_fn=dit.get("activation_fn", "gelu-approximate"),
        num_embeds_ada_norm=dit.get("num_embeds_ada_norm", 1000),
        upcast_attention=dit.get("upcast_attention", False),
        norm_elementwise_affine=dit.get("norm_elementwise_affine", False),
        norm_eps=dit.get("norm_eps", 1e-5),
        max_num_positional_embeddings=dit.get("max_num_positional_embeddings", 512),
    )
    vl_attn_cfg = SelfAttentionConfig(
        num_attention_heads=vla.get("num_attention_heads", 32),
        attention_head_dim=vla.get("attention_head_dim", 64),
        num_layers=vla.get("num_layers", 4),
        dropout=vla.get("dropout", 0.2),
        final_dropout=vla.get("final_dropout", True),
        positional_embeddings=vla.get("positional_embeddings", None),
        attention_bias=vla.get("attention_bias", True),
        activation_fn=vla.get("activation_fn", "gelu-approximate"),
        upcast_attention=vla.get("upcast_attention", False),
        max_num_positional_embeddings=vla.get("max_num_positional_embeddings", 512),
    )
    action_head_cfg = ActionHeadConfig(
        action_dim=ah.get("action_dim", 32),
        action_horizon=ah.get("action_horizon", 16),
        add_pos_embed=ah.get("add_pos_embed", True),
        backbone_embedding_dim=ah.get("backbone_embedding_dim", 2048),
        diffusion_model_cfg=dit_cfg,
        hidden_size=ah.get("hidden_size", 1024),
        input_embedding_dim=ah.get("input_embedding_dim", 1536),
        max_state_dim=ah.get("max_state_dim", 64),
        noise_beta_alpha=ah.get("noise_beta_alpha", 1.5),
        noise_beta_beta=ah.get("noise_beta_beta", 1.0),
        noise_s=ah.get("noise_s", 0.999),
        num_inference_timesteps=ah.get("num_inference_timesteps", 4),
        num_target_vision_tokens=ah.get("num_target_vision_tokens", 32),
        num_timestep_buckets=ah.get("num_timestep_buckets", 1000),
        tune_projector=ah.get("tune_projector", True),
        tune_diffusion_model=ah.get("tune_diffusion_model", True),
        use_vlln=ah.get("use_vlln", True),
        vl_self_attention_cfg=vl_attn_cfg,
        max_num_embodiments=ah.get("max_num_embodiments", 32),
        max_seq_len=ah.get("max_seq_len", 1024),
        expand_batch=ah.get("expand_batch", None),
    )
    return GR00T_N1_5_BaseConfig(
        backbone_cfg=backbone_cfg,
        action_head_cfg=action_head_cfg,
        action_dim=data.get("action_dim", 32),
        action_horizon=data.get("action_horizon", 16),
        hidden_size=data.get("hidden_size", 2048),
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_checkpoint(model: Any, ckpt_path: str) -> None:
    """Load model weights from a DCP or safetensors checkpoint."""
    ckpt = Path(ckpt_path)
    sf_files = list(ckpt.glob("*.safetensors"))
    has_dcp = list(ckpt.glob("*.distcp")) or (ckpt / ".metadata").exists()

    if sf_files:
        from safetensors.torch import load_file

        state_dict: dict[str, Any] = {}
        for sf in sf_files:
            state_dict.update(load_file(str(sf)))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s...", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s...", len(unexpected), unexpected[:5])
        logger.info("Loaded safetensors checkpoint from %s", ckpt)
    elif has_dcp:
        import torch.distributed as dist
        import torch.distributed.checkpoint as dcp

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29501")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        model_sd = model.state_dict()
        dcp.load({"model": model_sd}, checkpoint_id=str(ckpt))
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s...", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s...", len(unexpected), unexpected[:5])
        logger.info("Loaded DCP checkpoint from %s", ckpt)
    else:
        raise FileNotFoundError(f"No checkpoint files in {ckpt}. Expected *.safetensors or *.distcp.")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _preprocess_frame(frame: np.ndarray) -> Image.Image:
    """Convert uint8 HWC frame to 224×224 PIL Image.

    Matches the GR00TVideoTransform pipeline (augment=False):
    float [0,1] → CenterCrop(95%) → Resize(224) → uint8 → PIL.
    """
    t = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
    t = t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    h, w = t.shape[-2:]
    t = T.CenterCrop((int(h * 0.95), int(w * 0.95)))(t)
    t = T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True)(t)
    np_frame = (t[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(np_frame)


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------


class GR00TBaseModelServer(PredictModelServer):
    """GR00T-Base (N1.5) model server using susser-tod flow-matching action head.

    Args:
        llvm_dir: Path to the LLVM repo root (contains ``external/susser-tod``
            and ``src/llvm``).
        model_path: Path to checkpoint directory (safetensors or DCP format).
        train_data_path: Path to training data root used for loading
            normalisation statistics (``meta/stats.json`` per task).
        embodiment_tag: Embodiment identifier, e.g. ``"robotwin/agilex_random"``.
        denoising_steps: Number of Euler integration steps for the flow-matching
            action head (default 8, model default is 4).
        tokenizer_dir: Optional path to Eagle tokenizer data.  If ``None``,
            uses the bundled tokenizer in the LLVM repo.
    """

    def __init__(
        self,
        llvm_dir: str,
        model_path: str,
        train_data_path: str,
        embodiment_tag: str = "robotwin/agilex_random",
        denoising_steps: int = 8,
        tokenizer_dir: str | None = None,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.llvm_dir = llvm_dir
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.embodiment_tag = embodiment_tag
        self.denoising_steps = denoising_steps
        self.tokenizer_dir = tokenizer_dir

        self._model: Any = None
        self._eagle_processor: Any = None
        self._stats: dict[str, Any] | None = None
        self._action_inverse: Any = None
        self._normalize_fn: Any = None
        self._embodiment_id: int = 0
        self._action_horizon: int = 16
        self._max_state_dim: int = 64
        self._device: torch.device = torch.device("cpu")

    def _load_model(self) -> None:
        if self._model is not None:
            return

        _setup_llvm_environment(self.llvm_dir)

        from susser_tod.models.gr00t_n1_5.modeling import GR00T_N1_5_Base

        # --- Config ---
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            model_cfg = _load_model_config(config_path)
        else:
            from susser_tod.models.gr00t_n1_5.modeling import GR00T_N1_5_BaseConfig

            logger.warning("config.json not found in %s; using defaults", self.model_path)
            model_cfg = GR00T_N1_5_BaseConfig()

        model_cfg.action_head_cfg.num_inference_timesteps = self.denoising_steps
        self._action_horizon = model_cfg.action_head_cfg.action_horizon
        self._max_state_dim = model_cfg.action_head_cfg.max_state_dim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eid = _EMBODIMENT_TAG_TO_ID.get(self.embodiment_tag)
        if eid is None:
            raise ValueError(f"Unknown embodiment_tag: {self.embodiment_tag}")
        self._embodiment_id = eid

        # --- Model ---
        model = GR00T_N1_5_Base(model_cfg)
        model.to(device=self._device, dtype=torch.bfloat16)
        model.eval()
        _load_checkpoint(model, self.model_path)
        self._model = model

        # --- Statistics ---
        llvm_src = os.path.join(self.llvm_dir, "src")
        stats_mod = _import_file("_gr00t_statistics", os.path.join(llvm_src, "llvm/data/gr00t_statistics.py"))
        embodiment_path = Path(self.train_data_path) / self.embodiment_tag
        if not embodiment_path.exists():
            raise FileNotFoundError(f"Embodiment path not found: {embodiment_path}")
        task_dirs = sorted([p for p in embodiment_path.iterdir() if p.is_dir()])
        self._stats = stats_mod.get_merged_statistics(task_dirs)

        # --- Transforms (via importlib to bypass heavy __init__) ---
        transforms_mod = _import_file(
            "_gr00t_transforms", os.path.join(llvm_src, "llvm/transforms/gr00t_transforms.py")
        )
        self._normalize_fn = transforms_mod._normalize_min_max
        self._action_inverse = transforms_mod.GR00TActionInverseTransform(self._stats)

        # --- Eagle processor ---
        eagle_mod = _import_file(
            "_eagle_batch_transform", os.path.join(llvm_src, "llvm/transforms/eagle_batch_transform.py")
        )
        self._eagle_processor = eagle_mod.EagleProcessor(model_dir=self.tokenizer_dir)

        logger.info(
            "GR00T-Base loaded: action_horizon=%d, max_state_dim=%d, device=%s",
            self._action_horizon,
            self._max_state_dim,
            self._device,
        )

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._model is not None and self._stats is not None

        # --- Images: HWC uint8 → 224×224 PIL ---
        images_dict = obs.get("images", {})
        images = [_preprocess_frame(images_dict[k]) for k in _CAM_KEYS if k in images_dict]

        # --- Text: ChatML with image placeholders ---
        instruction = obs.get("task_description", "")
        text = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n"
        for i in range(1, len(images) + 1):
            text += f"<image-{i}>"
        text += f"{instruction}<|im_end|>\n<|im_start|>assistant\n"

        # --- State: normalize + pad ---
        joint_state = obs.get("joint_state", obs.get("state"))
        if joint_state is None:
            joint_vec = np.zeros(14, dtype=np.float32)
        else:
            joint_vec = np.asarray(joint_state, dtype=np.float32)

        state_dim = len(joint_vec)
        state_t = torch.from_numpy(joint_vec[np.newaxis]).float()
        state_norm = self._normalize_fn(state_t, self._stats["observation.state"]).numpy()
        if state_dim < self._max_state_dim:
            state_norm = np.pad(state_norm, ((0, 0), (0, self._max_state_dim - state_dim)))
        else:
            state_norm = state_norm[:, : self._max_state_dim]

        # --- Eagle processor (tokenize text + tile images) ---
        eagle_out = self._eagle_processor(text=text, images=images, return_tensors="pt", padding=False)

        inputs = {
            "eagle_input_ids": eagle_out["input_ids"].to(self._device),
            "eagle_attention_mask": eagle_out["attention_mask"].to(self._device),
            "eagle_pixel_values": eagle_out["pixel_values"].to(self._device),
            "embodiment_id": torch.tensor([self._embodiment_id], dtype=torch.long, device=self._device),
            "state": torch.from_numpy(state_norm).unsqueeze(0).to(self._device),
        }

        # --- Inference ---
        actions_padded = self._model.predict(inputs)  # (1, action_horizon, max_action_dim)
        actions_denorm = self._action_inverse(actions_padded[0].float())  # (action_horizon, action_dim)
        return {"actions": actions_denorm.cpu().numpy().astype(np.float32)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T-Base (N1.5) model server (uv script)")
    parser.add_argument(
        "--llvm_dir", default=_DEFAULT_LLVM_DIR, help="Path to the LLVM repo root (default: external/LLVM submodule)"
    )
    parser.add_argument("--model_path", required=True, help="Path to checkpoint directory (safetensors or DCP)")
    parser.add_argument("--train_data_path", required=True, help="Training data root (for normalisation stats)")
    parser.add_argument("--embodiment_tag", default="robotwin/agilex_random")
    parser.add_argument("--denoising_steps", type=int, default=8, help="Euler steps for flow-matching (default 8)")
    parser.add_argument("--tokenizer_dir", default=None, help="Eagle tokenizer data directory (default: bundled)")
    parser.add_argument("--chunk_size", type=int, default=16, help="Action horizon / chunk size (default 16)")
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = GR00TBaseModelServer(
        llvm_dir=args.llvm_dir,
        model_path=args.model_path,
        train_data_path=args.train_data_path,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        tokenizer_dir=args.tokenizer_dir,
        chunk_size=args.chunk_size,
        action_ensemble=args.action_ensemble,
    )

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
