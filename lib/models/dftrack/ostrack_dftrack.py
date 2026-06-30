
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.IRIS import IRIS
from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224, resize_pos_embed
from .vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from greenlet import greenlet

class OSTrack_DFTrack(nn.Module):
    

    def __init__(self, rgb_branch, tir_branch, teacher_rgb, teacher_tir, box_head, box_head_v, box_head_i, \
                 aux_loss=False, head_type="CORNER",mask_probability=0.0, mask_ratio=0.0, training=True):
        
        super().__init__()
        self.igf_module = IRIS(embedding_dim=rgb_branch.embed_dim)
        self.rgb_branch = rgb_branch
        self.tir_branch = tir_branch
        self.rgb_branch.mask_probability = mask_probability
        self.tir_branch.mask_probability = mask_probability
        self.rgb_branch.mask_ratio = mask_ratio
        self.tir_branch.mask_ratio = mask_ratio
        self.box_head = box_head
        
        self.align_rgb2tir = nn.Linear(self.rgb_branch.embed_dim, self.tir_branch.embed_dim, bias=False)
        self.align_tir2rgb = nn.Linear(self.tir_branch.embed_dim, self.rgb_branch.embed_dim, bias=False)
        with torch.no_grad():
            if self.rgb_branch.embed_dim == self.tir_branch.embed_dim:
                nn.init.eye_(self.align_rgb2tir.weight)
                nn.init.eye_(self.align_tir2rgb.weight)
            else:
                nn.init.xavier_uniform_(self.align_rgb2tir.weight)
                nn.init.xavier_uniform_(self.align_tir2rgb.weight)
        if training:
            self.teacher_rgb = teacher_rgb
            self.teacher_tir = teacher_tir
            self.box_head_v = box_head_v
            self.box_head_i = box_head_i
        else:
            self.teacher_rgb = None
            self.teacher_tir = None
            self.box_head_v = None
            self.box_head_i = None

        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)



    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        if self.training:

            teacher_rgb_gr = greenlet(self.teacher_rgb)
            teacher_tir_gr = greenlet(self.teacher_tir)
            rgb_branch_gr = greenlet(self.rgb_branch)
            tir_branch_gr = greenlet(self.tir_branch)
            self.teacher_rgb.next_gr[0] = teacher_tir_gr
            self.teacher_tir.next_gr[0] = rgb_branch_gr
            self.rgb_branch.next_gr[0] = tir_branch_gr
            self.tir_branch.next_gr[0] = teacher_rgb_gr

            t_z_rgb, t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(z_li=template, x_li=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

            t_z_tir,t_x_tir, t_aux_dict_tir = teacher_tir_gr.switch()

            z_rgb,x_rgb, aux_dict_rgb = rgb_branch_gr.switch()

            z_tir, x_tir, aux_dict_tir = tir_branch_gr.switch()

            t_x_rgb1 = torch.cat([t_z_rgb,t_x_rgb],dim=1)
            t_x_tir1 = torch.cat([t_z_tir,t_x_tir],dim=1)
            x_rgb1 = torch.cat([z_rgb,x_rgb],dim=1)
            x_tir1 = torch.cat([z_tir,x_tir],dim=1)


            aux_dict = {
                'x_rgb':x_rgb1,
                'x_tir':x_tir1,
                't_x_rgb':t_x_rgb1,
                't_x_tir':t_x_tir1,
                'aux_dict_rgb':aux_dict_rgb,
                'aux_dict_tir':aux_dict_tir,
                'aux_dict_t_rgb':t_aux_dict_rgb,
                'aux_dict_t_tir':t_aux_dict_tir,}



            x_fused, lambda_tokens = self.igf_module(x_rgb, x_tir, return_lambda=True)  

            x_rgb = torch.cat([z_rgb, x_fused], dim=1)
            x_tir = torch.cat([z_tir, x_fused], dim=1)
            x = torch.cat([x_rgb, x_tir], 2)

            feat_last = x[-1] if isinstance(x, list) else x
            out = self.forward_head(feat_last, None)


            out_t_tir = self.forward_head(t_x_tir1, None, head=self.box_head_i)
            out['out_t_tir'] = out_t_tir
            out_t_rgb = self.forward_head(t_x_rgb1, None, head=self.box_head_v)
            out['out_t_rgb'] = out_t_rgb

            
            B = lambda_tokens.shape[0]
            H = W = self.feat_sz_s
            lambda_map = lambda_tokens.permute(0, 2, 1).contiguous().view(B, 1, H, W)
            out['lambda_factor'] = lambda_map

            
            out.update(aux_dict)
            out['backbone_feat'] = x
            return out

        else:

            rgb_branch_gr = greenlet(self.rgb_branch)
            tir_branch_gr = greenlet(self.tir_branch)
            self.rgb_branch.next_gr[0] = tir_branch_gr
            self.tir_branch.next_gr[0] = rgb_branch_gr
            z_rgb, x_rgb, aux_dict_rgb = rgb_branch_gr.switch(z_li=template[:2], x_li=search[:2],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

            z_tir, x_tir, aux_dict_tir = tir_branch_gr.switch()

            aux_dict = {
                'aux_dict_rgb':aux_dict_rgb,
                'aux_dict_tir':aux_dict_tir}



            x_fused, lambda_tokens = self.igf_module(x_rgb, x_tir, return_lambda=True)  

            x_rgb = torch.cat([z_rgb, x_fused], dim=1)
            x_tir = torch.cat([z_tir, x_fused], dim=1)
            x = torch.cat([x_rgb, x_tir], 2)


            feat_last = x[-1] if isinstance(x, list) else x
            out = self.forward_head(feat_last, None)

            B = lambda_tokens.shape[0]
            H = W = self.feat_sz_s
            lambda_map = lambda_tokens.permute(0, 2, 1).contiguous().view(B, 1, H, W)
            out['lambda_factor'] = lambda_map

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out

    def forward_head(self, cat_feature, gt_score_map=None, head=None):
        
        if head==None:
            box_head = self.box_head
        else:
            box_head = head
        enc_opt = cat_feature[:, -self.feat_len_s:]  
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            
            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack_dftrack(cfg, training=True):
    patch_start_index = 1


    rgb_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
    rgb_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    if cfg.MODEL.SHARE_STUDENT:
        tir_branch = rgb_branch
    else:
        tir_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                             ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                             ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        tir_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    if cfg.MODEL.RGB_TEACHER:
        teacher_rgb = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                              ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_rgb.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_v = build_box_head(cfg, teacher_rgb.embed_dim)
    else:
        teacher_rgb = None
        box_head_v = None

    if cfg.MODEL.TIR_TEACHER:
        teacher_tir = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                              ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_tir.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_i = build_box_head(cfg, teacher_tir.embed_dim)
    else:
        teacher_tir = None
        box_head_i = None

    box_head = build_box_head(cfg, teacher_tir.embed_dim * 2)

    backbone_weight_filter = lambda param_dict: {k.replace("backbone.", ""): v for k, v in param_dict.items() if
                                                 'backbone' in k}
    boxhead_weight_filter = lambda param_dict: {k.replace("box_head.", ""): v for k, v in param_dict.items() if
                                                'box_head' in k}

    def pos_embed_filter(param):
        param['backbone.pos_embed_z'] = resize_pos_embed(param['backbone.pos_embed_z'],
                                                         posemb_new=torch.zeros(1, 64, 768), num_tokens=0)
        param['backbone.pos_embed_x'] = resize_pos_embed(param['backbone.pos_embed_x'],
                                                         posemb_new=torch.zeros(1, 256, 768), num_tokens=0)
        param['backbone.pos_embed_z'] += param['backbone.temporal_pos_embed_z']
        param['backbone.pos_embed_x'] += param['backbone.temporal_pos_embed_x']
        return param

    if training:
        print("load RGB parameters:", cfg.MODEL.RGB_BRANCH)
        rgb_param = torch.load(cfg.MODEL.RGB_BRANCH, map_location="cpu")['net']
        if "DropTrack" in cfg.MODEL.RGB_BRANCH:
            rgb_param = pos_embed_filter(rgb_param)
        rgb_branch.load_state_dict(backbone_weight_filter(rgb_param), strict=False)

        if not cfg.MODEL.SHARE_STUDENT:
            print("load TIR parameters:", cfg.MODEL.TIR_BRANCH)
            tir_param = torch.load(cfg.MODEL.TIR_BRANCH, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_BRANCH:
                tir_param = pos_embed_filter(tir_param)
            tir_branch.load_state_dict(backbone_weight_filter(tir_param), strict=False)

        print("Tracking head type: concat")
        head_param = boxhead_weight_filter(rgb_param)
        for k, v in list(head_param.items()):
            if k in ['conv1_ctr.0.weight', 'conv1_offset.0.weight', 'conv1_size.0.weight']:
                head_param[k] = torch.cat([v, v], 1)
        box_head.load_state_dict(head_param, strict=False)

        if teacher_rgb is not None:
            print("load rgb teacher parameters:", cfg.MODEL.RGB_TEACHER)
            rgbTeacher_param = torch.load(cfg.MODEL.RGB_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.RGB_TEACHER:
                rgbTeacher_param = pos_embed_filter(rgbTeacher_param)
            teacher_rgb.load_state_dict(backbone_weight_filter(rgbTeacher_param), strict=False)
        if box_head_v is not None:
            box_head_v.load_state_dict(boxhead_weight_filter(rgbTeacher_param), strict=False)

        if teacher_tir is not None:
            print("load tir teacher parameters:", cfg.MODEL.TIR_TEACHER)
            tirTeacher_param = torch.load(cfg.MODEL.TIR_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_TEACHER:
                tirTeacher_param = pos_embed_filter(tirTeacher_param)
            teacher_tir.load_state_dict(backbone_weight_filter(tirTeacher_param), strict=False)
        if box_head_i is not None:
            box_head_i.load_state_dict(boxhead_weight_filter(tirTeacher_param), strict=False)

    model = OSTrack_DFTrack(
        rgb_branch,
        tir_branch,
        teacher_rgb,
        teacher_tir,
        box_head=box_head,
        box_head_v=box_head_v,
        box_head_i=box_head_i,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        mask_ratio=cfg.TRAIN.INPUT_MASK_RATIO,
        mask_probability=cfg.TRAIN.MASK_PROBABILITY,
        training=training,
    )

    return model