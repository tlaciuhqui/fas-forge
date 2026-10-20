"""Standalone demo: DINOv2 with register tokens via ``timm``.

This script intentionally avoids importing ``common.utils`` (which pulls in
``wandb``), so it can be run directly to inspect how timm represents the
[CLS] token and the learnable register tokens inside a DINOv2 model.

Run from the Unified_FAS directory:

    python models/demo_dinov2_registers.py
"""
import torch
import timm


def main():
    model_name = 'vit_small_patch14_reg4_dinov2'
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0,
                                 img_size=224, dynamic_img_size=True)

    cls_token = backbone.cls_token
    reg_token = backbone.reg_token

    print(f'{model_name}:')
    print(f'  embed_dim          = {backbone.embed_dim}')
    print(f'  num_prefix_tokens  = {backbone.num_prefix_tokens}')
    print(f'  cls_token shape    = {tuple(cls_token.shape) if cls_token is not None else None}')
    print(f'  reg_token shape    = {tuple(reg_token.shape) if reg_token is not None else None}')
    print(f'  num_registers      = {reg_token.shape[1] if reg_token is not None else 0}')

    # [B, 1 + num_registers + num_patches, embed_dim]
    x = torch.randn(2, 3, 224, 224)
    tokens = backbone.forward_features(x)
    print(f'  forward_features   = {tuple(tokens.shape)}  # [B, 1 + R + patches, C]')

    # Drop [CLS] + register tokens and keep only the patch tokens, then
    # reshape them into a 2-D feature map for downstream CNN heads.
    patch_tokens = tokens[:, backbone.num_prefix_tokens:]
    b, n, c = patch_tokens.shape
    h = w = int(n ** 0.5)
    feat = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
    print(f'  patch token map    = {tuple(feat.shape)}  # [B, C, H/14, W/14]')


if __name__ == '__main__':
    main()
