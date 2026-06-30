import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from lib.models.layers.EMA import EMA
from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import MCEBlock, CEBlock
from greenlet import greenlet
_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None
                 ):
        
        
        super().__init__()
        self.next_gr = [None]
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.ema = EMA(channels=embed_dim, factor=8)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                MCEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i, next_gr=self.next_gr, layer_idx=i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
        

    def forward_features(self, z, x, mask_z=1., mask_x=1.,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, attn_li=[]
                         ):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        
        x = self.patch_embed(x)
        z = z.flatten(0, 1) 
        z = self.patch_embed(z) 
        z = mask_z*z
        x = mask_x*x
        z += self.pos_embed_z
        x += self.pos_embed_x
        N = z.shape[1]
        C = z.shape[2]
        z = z.view(4, B, N, C).permute(1, 0, 2, 3).flatten(1, 2)
        
        
        
        
        
        

        x = combine_tokens(z, x, mode=self.cat_mode)


        x = self.pos_drop(x)

        lens_z = z.shape[1]  
        lens_x = x.shape[1] - lens_z  

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        x_list = {}
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn, attn_li = \
                blk(x, global_index_t, global_index_s, None, ce_template_mask, ce_keep_rate, attn_li)
            if self.training:
                x_list[i] = x.clone()

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            
            C = x.shape[-1]
            
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        
        
        
        

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
            "global_index_s": global_index_s,
            "x_list": x_list
        }
        self.next_gr[0].switch(attn_li)

        return z,x,aux_dict




    def forward(self, z_li, x_li, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, mask_probability=0.0, mask_ratio=0.0,
                return_last_attn=False, mask_x=1., mask_z=1.):
        z = z_li[0]
        x = x_li[0]

        if len(z_li)>1:
            
            if len(z_li)==2 and self.training:
                
                N_x = 256
                B = z.shape[0]
                if self.mask_probability<1. and self.mask_probability>0.:
                    mask_fun = lambda flag: torch.rand(N_x)<self.mask_ratio if flag else torch.ones(N_x)
                    mask_x = torch.stack([mask_fun(torch.rand(1).item()>self.mask_probability) for _ in range(B)]).unsqueeze(-1).to(z.device)
                elif self.mask_probability>=1.:

                    mask_x = torch.stack([torch.rand(N_x)>self.mask_ratio for _ in range(B)], dim=0).unsqueeze(-1).to(z.device)
                else:
                    mask_x = 1.



            attn_li = self.next_gr[0].switch(z_li[1:], x_li[1:], ce_template_mask, ce_keep_rate, tnc_keep_rate, 
                                mask_probability, mask_ratio,return_last_attn, mask_x)
        else:
            attn_li = self.next_gr[0].switch((None, [], []))     

        z1, x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            mask_x=mask_x, attn_li=attn_li, mask_z=mask_z)
        return z1,x, aux_dict



def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            except:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model





def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model



def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
