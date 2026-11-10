"""Custom Ultralytics trainers for LiteViT architectures."""

from __future__ import annotations

import traceback

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

from litevit.models import (
    wrap_gate_edge_cbam,
    wrap_gate_only,
    wrap_gate_p2,
    wrap_gate_p5,
    wrap_v1_p5,
)
from litevit.models.build import EDGE_LAYER_INDICES, GATE_LAYER_INDICES, P5_LAYER_INDEX
from litevit.models.p2 import DETECT_LAYER_INDEX


class LiteDetectionTrainer(DetectionTrainer):
    """Build a LiteViT variant inside trainer setup.

    Ultralytics rebuilds the model from ``model.yaml`` when training starts. The
    variant wrapper must therefore be applied after that rebuild, before the
    optimizer, EMA, and training loop are created.
    """

    def _is_wrapped(self, model) -> bool:
        raise NotImplementedError

    def _wrap_model(self, model):
        raise NotImplementedError

    def get_model(self, cfg=None, weights=None, verbose=True):
        if getattr(self, "resume", False) and weights is not None:
            model = getattr(weights, "ema", weights)
            if getattr(model, "model", None) is None:
                raise RuntimeError("resume checkpoint has no detector layer sequence")
            if not self._is_wrapped(model):
                model = self._wrap_model(model)
            return model
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return self._wrap_model(model)

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


class V1DetectionTrainer(LiteDetectionTrainer):
    """Build P5 LiteViT inside trainer setup."""

    def _is_wrapped(self, model) -> bool:
        return hasattr(model.model[P5_LAYER_INDEX], "block")

    def _wrap_model(self, model):
        return wrap_v1_p5(model)


class GateDetectionTrainer(LiteDetectionTrainer):
    """Build P3/P4/P5 weather gates inside trainer setup."""

    def _is_wrapped(self, model) -> bool:
        return all(hasattr(model.model[index], "gate") for index in GATE_LAYER_INDICES)

    def _wrap_model(self, model):
        return wrap_gate_only(model)


class GateP2DetectionTrainer(LiteDetectionTrainer):
    """Build P3/P4/P5 weather gates plus a stride-4 P2 detection head."""

    def _is_wrapped(self, model) -> bool:
        return (
            len(model.model) == DETECT_LAYER_INDEX + 1
            and getattr(model.model[DETECT_LAYER_INDEX], "nl", 0) == 4
            and all(hasattr(model.model[i], "gate") for i in GATE_LAYER_INDICES)
        )

    def _wrap_model(self, model):
        return wrap_gate_p2(model)


class GateP5DetectionTrainer(LiteDetectionTrainer):
    """Build regularized P5 attention plus P3/P4/P5 gates inside trainer setup."""

    def _is_wrapped(self, model) -> bool:
        return hasattr(model.model[P5_LAYER_INDEX], "block") and all(
            hasattr(model.model[index], "gate") for index in GATE_LAYER_INDICES
        )

    def _wrap_model(self, model):
        return wrap_gate_p5(model)


class GateEdgeCBAMTrainer(LiteDetectionTrainer):
    """Build Sobel edge + channel gate + spatial CBAM inside trainer setup."""

    def _is_wrapped(self, model) -> bool:
        return all(hasattr(model.model[i], "edge") for i in EDGE_LAYER_INDICES) and all(
            hasattr(model.model[i], "gate") for i in GATE_LAYER_INDICES
        )

    def _wrap_model(self, model):
        return wrap_gate_edge_cbam(model)
