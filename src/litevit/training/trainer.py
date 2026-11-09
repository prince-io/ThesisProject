"""Custom Ultralytics trainers for LiteViT architectures."""

from __future__ import annotations

import traceback

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

from litevit.models import wrap_v1_p5
from litevit.models.build import P5_LAYER_INDEX


class V1DetectionTrainer(DetectionTrainer):
    """Build V1 inside the trainer so Ultralytics cannot discard the wrapper.

    The default trainer rebuilds the model from ``model.yaml`` when training
    starts. This override wraps the freshly built model before the optimizer,
    EMA, and training loop are created.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        if getattr(self, "resume", False) and weights is not None:
            model = getattr(weights, "ema", weights)
            layers = getattr(model, "model", None)
            if layers is None:
                raise RuntimeError("resume checkpoint has no detector layer sequence")
            if not hasattr(layers[P5_LAYER_INDEX], "block"):
                wrap_v1_p5(model)
            return model
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return wrap_v1_p5(model)

    def _log_post_train_error(self, name: str) -> None:
        path = self.save_dir / f"{name}_error.txt"
        path.write_text(traceback.format_exc(), encoding="utf-8")
        LOGGER.warning(f"{name} failed; traceback saved to {path}")

    def final_eval(self):
        """Run final validation, falling back to the live FP32 model if checkpoint reload fails."""
        try:
            super().final_eval()
        except Exception:
            self._log_post_train_error("final_eval")
            self.validator.args.plots = self.args.plots
            self.metrics, self.fitness = self.validate()
            if self.metrics:
                self.metrics.pop("fitness", None)

    def plot_metrics(self):
        """Do not fail a completed training run because a diagnostic plot failed."""
        try:
            super().plot_metrics()
        except Exception:
            self._log_post_train_error("plot_metrics")
