---
title: Invalid V1 Run — Diagnosis and Lesson
status: draft
date: 2026-11-08
sources:
  - runs/train/litevit_v1_p5_seed42
  - runs/eval/litevit_v1_p5_seed42
  - src/litevit/models/build.py
  - src/litevit/training/train.py
---

# Invalid V1 Run — Diagnosis and Lesson

## Observation

The first `litevit_v1_p5_seed42` run produced metrics almost identical to the
`EXP-0001` baseline: test mAP50 0.5418 vs 0.5428 and test mAP50-95 0.3271 vs
0.3287. The raw artifacts are retained under `runs/train/litevit_v1_p5_seed42`
and `runs/eval/litevit_v1_p5_seed42`.

The saved checkpoint explains why:

- layer 9 is `SPPF`, not `FeatureWithBlock`;
- parameter count is 3,138,975, the baseline count;
- GFLOPs is 8.754, the baseline value;
- the expected V1 count is 3,674,271 parameters and 9.206 GFLOPs.

## Root cause

`build_v1_p5()` wrapped layer 9 correctly before training. However, Ultralytics
`Model.train()` rebuilds the trainer model from `self.model.yaml` when a checkpoint
is loaded:

```python
self.trainer.model = self.trainer.get_model(
    weights=weights,
    cfg=self.model.yaml,
)
```

That rebuild creates a fresh stock YOLOv8n and loads the pretrained weights. The
Python-level wrapper is not in `model.yaml`, so it was silently discarded before
the optimizer and training loop were created.

The builder tests passed because they checked `build_v1_p5()` in isolation. They
did not check the model that Ultralytics actually trained.

## Interpretation

- This run is a **baseline repeat**, not a V1 experiment.
- Its small differences from `EXP-0001` are run-to-run noise, not a ViT effect.
  Overall differences are ~0.1–0.5 mAP point; per-condition differences reach
  ~2–3 points. That is a warning for interpreting future single-seed results.
- No conclusion about P5 global context, linear attention, or the LiteViT block can
  be drawn from this run.

## Corrective action

1. Move V1 construction into `V1DetectionTrainer.get_model()` so the wrapper is
   created inside Ultralytics trainer setup.
2. Add post-training verification: load `best.pt`, require `FeatureWithBlock` at
   layer 9, and require about 3.674 M parameters.
3. Rerun as `litevit_v1_p5_seed42_v2`.
4. Keep this failed run for audit; do not delete it.

## Follow-up 10-epoch test

After the trainer fix and the FP32 block, a 10-epoch test completed with finite
losses and a correct 3,674,271-parameter block (epoch 10: val mAP50 0.4082,
mAP50-95 0.2367). The wrapper script then failed before writing `run_meta.json` and
`vram.json`, likely in final validation or plotting. Fallback logging was added; a
repeat test is pending.

## Lesson

A model builder test is not enough. Every architecture run must verify the
**saved checkpoint**, not only the object returned by the builder.
