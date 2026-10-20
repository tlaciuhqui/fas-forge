# Unified_FAS — General Face Anti-Spoofing for CelebA-Spoof

A general facial presentation attack detection (PAD) project built from the
*shared* components of the existing [`DCN`](../DCN) and [`ANRL`](../ANRL)
tasks, plus the common infrastructure under `security/common`.

It targets the official
[CelebA-Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof) dataset
layout instead of the OULU/CASIA/Replay list-file layout used by DCN/ANRL.

## What is reused (general components)

| Component | Source | Where it lives here |
| --- | --- | --- |
| `BaseTask` (env, DDP, optimizer, scheduler, checkpointing) | `common/task/base_task.py` | reused as-is |
| `test_module` image/video evaluation helper | `common/task/fas/modules.py` | reused for image-level eval |
| Base transforms / dataloader | `common/data` | reused in `datasets/factory.py` |
| Meters, metrics, seeding, face-box utils | `common/utils` | reused in `train.py` |
| Balanced pos/neg training loop (zip real + spoof loaders) | shared by DCN & ANRL | `train.py` |
| Multi-scale CNN encoder → classifier → auxiliary depth head | shared by DCN & ANRL | `models/framework.py` |
| Adaptive BN/IN normalization (`AdapNorm`) | generalized from ANRL | `models/base_block.py` |
| `ConvBlock`/`BasicBlock` downsampling unit | generalized from DCN/ANRL | `models/base_block.py` |
| HSV jitter + BGR→RGB color handling | shared by DCN & ANRL | `datasets/` |

## What is intentionally left out (method-specific components)

- **DCN specifics** (`../DCN`): jigsaw patch destruction, patch permutations,
  cutmix across subdomains/classes, and the patch-similarity head.
- **ANRL specifics** (`../ANRL`): meta-train/meta-test optimization, fast
  weights, and the dual calibration constraints (Inter-Domain Compatible /
  Inter-Class Separable losses).

Those remain in their own projects. This project only reuses the parts that
are general across face anti-spoofing models.

## CelebA-Spoof dataset structure

The dataset follows the official CelebA-Spoof layout:

```
CelebA_Spoof/
    Data/
        train/
            {subject_id}/
                live/{image_id}.jpg
                spoof/{image_id}.jpg
        test/
            {subject_id}/
                live/{image_id}.jpg
                spoof/{image_id}.jpg
    metainfo/
        train_label.json
        test_label.json
```

- `metainfo/{split}_label.json` maps a relative image path to a 44-element
  annotation vector; index `43` is the Live/Spoof label (`0` = live, `1` = spoof),
  matching the official `intra_dataset_code/client.py`.
- Each image may have a sibling `{image_id}_BB.txt` bounding-box file used for
  face cropping (`crop_face: bb`).

If the metainfo JSON is missing, the dataset falls back to scanning
`Data/{split}/{subject_id}/{live,spoof}/*.jpg`.

## Project layout

```
Unified_FAS/
    README.md
    train.py
    train.sh
    configs/
        celeba_spoof.yaml
        celeba_spoof_dinov2.yaml
    datasets/
        __init__.py
        factory.py
        dataset.py
    models/
        __init__.py
        base_block.py
        framework.py
        demo_dinov2_registers.py
```

## Quickstart

1. Edit `configs/celeba_spoof.yaml`:
   - `dataset.CelebASpoofDataset.data_root` → your `CelebA_Spoof` root.
   - Optionally adjust `img_mode`, `crop_face`, `aux_head`, batch size, etc.

2. Single-GPU training (disable DDP):

```bash
cd security/tasks/Face-Anti-Spoofing/Unified_FAS
python train.py -c configs/celeba_spoof.yaml --distributed 0
```

3. Multi-GPU training:

```bash
sh train.sh 4 configs/celeba_spoof.yaml 12345
```

## DINOv2 with registers backbone

`models/framework.py` also includes `DINOv2RegFeatureExtractor`, which wraps a
timm DINOv2 model that uses *register tokens* (`vit_*_patch14_reg4_dinov2`).

In timm these models expose:

- `backbone.cls_token` — the `[CLS]` token, shape `[1, 1, embed_dim]`
- `backbone.reg_token` — the learnable register tokens, shape
  `[1, num_registers, embed_dim]` (4 registers for `*_reg4_*` models)
- `backbone.num_prefix_tokens` — `1 + num_registers`

`forward_features` therefore returns `[B, 1 + num_registers + num_patches, C]`.
The extractor drops the `[CLS]` and register tokens and reshapes the remaining
patch tokens into a `[B, C, H/14, W/14]` feature map that the CNN
classifier/aux heads can consume.

### Demo

Run the standalone timm demo (does not import the `common`/wandb stack):

```bash
cd security/tasks/Face-Anti-Spoofing/Unified_FAS
python models/demo_dinov2_registers.py
```

### Train with DINOv2 + registers

```bash
python train.py -c configs/celeba_spoof_dinov2.yaml --distributed 0
```

Notes:

- `img_mode` must be `rgb` and `in_ch` must be `3` (DINOv2 is a 3-channel ViT).
- `transform.image_size` must be divisible by the patch size (`14`); the
  example uses `224`.
- `backbone_pretrained: True` uses DINOv2 weights from the timm hub, and
  `backbone_freeze: True` keeps them frozen while the FAS heads train.
- The DINOv2 config uses the DINOv2/ImageNet mean/std normalization.

## Config notes

- `split_data.val: train` + `val_from_train_ratio: 0.05` builds a deterministic
  5% subject-level validation split from the training set, because
  CelebA-Spoof has no official validation split.
- `aux_map: null` and `aux_head: False` are the defaults for CelebA-Spoof,
  since the dataset does not provide depth/reflection ground truth. The aux
  head infrastructure is kept so users can re-enable it with a dataset that
  provides such maps.
- `crop_face: bb` uses the official `_BB.txt` files; set it to `none` to skip
  cropping.
