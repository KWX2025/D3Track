# -*- coding: utf-8 -*-
"""
Profile / FPS test for OStrack & DFTrack (4-frame template update).
- OStrack: 原流程（THOP + FPS）
- DFTrack: 仅 FPS（模板为 5D [4,B,3,H,W]，匹配 VisionTransformerCE.forward_features 的写法）
"""
import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import time
import importlib
import torch
from thop import profile
from thop.utils import clever_format
from lib.utils.misc import NestedTensor


# ------------------------- CLI -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for profiling / FPS test')
    parser.add_argument('--script', type=str, default='dftrack', choices=['ostrack', 'dftrack'],
                        help='which model family to run')
    parser.add_argument('--config', type=str, default='dftrack_4b_dropmae_dftrack_tf_cc_mask.25',
                        help='yaml configure file name, under experiments/<script>/*.yaml')

    # 仅影响测速，不影响模型
    parser.add_argument('--tpl_frames', type=int, default=4,
                        help='(DFTrack) number of template frames (must be 4 because backbone uses view(4,...))')
    parser.add_argument('--warmup', type=int, default=50,
                        help='warmup iterations for FPS test')
    parser.add_argument('--timing', type=int, default=200,
                        help='timing iterations for FPS test')

    return parser.parse_args()


# ------------------------- OStrack: 原有单帧评测（含 FLOPs/Params + FPS） -------------------------
def evaluate_vit(model, template, search):
    """Speed + THOP (OStrack 单帧)."""
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        # warmup
        for _ in range(T_w):
            _ = model(template, search)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(T_t):
            _ = model(template, search)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
    avg_lat = (end - start) / T_t
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    print("FPS is %.2f fps" % (1. / avg_lat))


def evaluate_vit_separate(model, template, search):
    """分步测速（OStrack 单帧，先 backbone 再融合）"""
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # warmup
        for _ in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for _ in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
    avg_lat = (end - start) / T_t
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))


# ------------------------- DFTrack: 多帧双模态的 FPS 测试（仅测速，不做 THOP） -------------------------
def _pack_dftrack_io_5d(tpl_rgb_4d, tpl_tir_4d, sch_rgb_4d, sch_tir_4d, T=4):
    """
    构造符合 VisionTransformerCE.forward_features 的输入：
      template_list = [z_rgb_5d, z_tir_5d]，每个形状 [4, B, 3, H, W]
      search_list   = [x_rgb_4d, x_tir_4d]，每个形状 [B, 3, H, W]
    仅用于 FPS 测试。
    """
    assert tpl_rgb_4d.dim() == 4 and tpl_tir_4d.dim() == 4, "tpl_* 必须是 [B,3,H,W]"
    assert sch_rgb_4d.dim() == 4 and sch_tir_4d.dim() == 4, "sch_* 必须是 [B,3,H,W]"
    B, C, H, W = tpl_rgb_4d.shape
    assert C == 3, f"in_chans must be 3, got {C}"

    # 按时间维堆叠成 [4, B, 3, H, W]（这里只是复制同一帧做 4 帧，用于测速）
    z_rgb_5d = tpl_rgb_4d.unsqueeze(0).repeat(T, 1, 1, 1, 1).contiguous()
    z_tir_5d = tpl_tir_4d.unsqueeze(0).repeat(T, 1, 1, 1, 1).contiguous()

    template_list = [z_rgb_5d, z_tir_5d]
    search_list = [sch_rgb_4d, sch_tir_4d]
    return template_list, search_list


def evaluate_vit_dftrack(model, tpl_frames=4, T_w=50, T_t=200, z_sz=128, x_sz=256, bs=1, device='cuda'):
    """
    DFTrack 的整体 FPS 测试：
      - 模板：5D [4,B,3,z_sz,z_sz]
      - 搜索：4D [B,3,x_sz,x_sz]
      - 仅测 FPS（不做 THOP）
    """
    if tpl_frames != 4:
        print(f"[DFTrack] WARNING: backbone uses view(4,...), force tpl_frames=4 (got {tpl_frames})")
        tpl_frames = 4

    # 构造两模态的随机输入
    tpl_rgb = torch.randn(bs, 3, z_sz, z_sz, device=device)
    tpl_tir = torch.randn(bs, 3, z_sz, z_sz, device=device)
    sch_rgb = torch.randn(bs, 3, x_sz, x_sz, device=device)
    sch_tir = torch.randn(bs, 3, x_sz, x_sz, device=device)

    template_list, search_list = _pack_dftrack_io_5d(tpl_rgb, tpl_tir, sch_rgb, sch_tir, T=tpl_frames)
    z_rgb_5d, z_tir_5d = template_list
    x_rgb_4d, x_tir_4d = search_list
    print(f"testing speed ... (DFTrack) "
          f"z_rgb={tuple(z_rgb_5d.shape)}, z_tir={tuple(z_tir_5d.shape)}, "
          f"x_rgb={tuple(x_rgb_4d.shape)}, x_tir={tuple(x_tir_4d.shape)}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        # warmup
        for _ in range(T_w):
            _ = model(template_list, search_list)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(T_t):
            _ = model(template_list, search_list)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
    avg_lat = (end - start) / T_t
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
    print("FPS is %.2f fps" % (1. / avg_lat))


# ------------------------- 其他辅助 -------------------------
def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


# ------------------------- Main -------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    args = parse_args()

    # === 读配置 ===
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)

    # === 基本尺寸 ===
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "ostrack":
        # ---------- 原 OStrack 分支：保持不变 ----------
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_ostrack
        model = model_constructor(cfg, training=False)
        template = torch.randn(bs, 3, z_sz, z_sz).to(device)
        search = torch.randn(bs, 3, x_sz, x_sz).to(device)
        model = model.to(device)

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search)
        else:
            evaluate_vit_separate(model, template, search)

    elif args.script == "dftrack":
        # ---------- DFTrack 分支：仅 FPS 测试（模板 5D，搜索 4D） ----------
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_ostrack_dftrack
        model = model_constructor(cfg, training=False).to(device)
        model.eval()

        evaluate_vit_dftrack(model,
                         tpl_frames=args.tpl_frames,
                         T_w=args.warmup,
                         T_t=args.timing,
                         z_sz=z_sz,
                         x_sz=x_sz,
                         bs=bs,
                         device=device)
        print("[Note] DFTrack 分支未打印 FLOPs/Params（THOP 在多帧多模态场景下易失真/报错）。")

    else:
        raise NotImplementedError