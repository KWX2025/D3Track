from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from .dftrack_loss import get_dftrack_loss


class OSTrack_DFTrack_Actor(BaseActor):
    

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.dftrack_loss = get_dftrack_loss(cfg)

        
        self._proj_use = True                                
        self._diag_enable = True                             
        self._diag_layers = getattr(cfg.TRAIN, "DIAG_LAYERS", [9])  
        self._diag_max_tokens = int(getattr(cfg.TRAIN, "DIAG_MAX_TOKENS", 2048))  

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        
        assert len(data['template_images']) >= 16
        assert len(data['search_images']) >= 2

        template_list = []
        for i in range(16):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        template_tensor = torch.stack(template_list, dim=0).view(4, 4, 4, 3, 128, 128)

        search_list = []
        for i in range(len(data['search_images'])):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, self.cfg.TRAIN.BATCH_SIZE, template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'],
                                            warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=template_tensor,
                            search=torch.stack(search_list),
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        return out_dict

    
    def _align_to_student(self, feat_teacher, feat_student, projector):
        
        t = feat_teacher.detach()
        if projector is not None:
            y = projector(t)
            
            if y.shape[-1] != feat_student.shape[-1]:
                Cs = feat_student.shape[-1]
                if y.shape[-1] > Cs:
                    y = y[..., :Cs]
                else:
                    pad = torch.zeros(y.shape[:-1] + (Cs - y.shape[-1],),
                                      device=y.device, dtype=y.dtype)
                    y = torch.cat([y, pad], dim=-1)
            return y
        
        c_src, c_dst = t.shape[-1], feat_student.shape[-1]
        if c_src == c_dst:  return t
        if c_src > c_dst:   return t[..., :c_dst]
        pad = torch.zeros(t.shape[:-1] + (c_dst - c_src,), device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=-1)

    
    @torch.no_grad()
    def _diag_basis(self, Ft, Fs, projector, max_tokens=2048):
        
        B, N, Ct = Ft.shape
        Cs = Fs.shape[-1]
        Ft2 = Ft.reshape(B * N, Ct)
        Fs2 = Fs.reshape(B * N, Cs)

        M = Ft2.shape[0]
        if M > max_tokens:
            idx = torch.randperm(M, device=Ft2.device)[:max_tokens]
            Ft2 = Ft2[idx]; Fs2 = Fs2[idx]

        def _norm_ch(X):
            return (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)

        
        Ft2n = _norm_ch(Ft2)
        Fs2n = _norm_ch(Fs2)

        
        C_pre = (Ft2n.T @ Fs2n) / (Ft2n.shape[0] - 1)            
        rowmax_pre = C_pre.abs().max(dim=1).values.mean()

        
        cos_base = F.cosine_similarity(F.normalize(Ft2n, dim=1), F.normalize(Fs2n, dim=1), dim=1).mean()

        
        Qa, _ = torch.linalg.qr(Ft2n, mode='reduced')
        Qb, _ = torch.linalg.qr(Fs2n, mode='reduced')
        S = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
        angle_base = torch.arccos(S).mean() * (180.0 / 3.141592653589793)

        
        if projector is not None:
            Ft2a = projector(Ft2n)
            if Ft2a.shape[1] != Cs:
                if Ft2a.shape[1] > Cs:
                    Ft2a = Ft2a[:, :Cs]
                else:
                    pad = torch.zeros(Ft2a.shape[0], Cs - Ft2a.shape[1], device=Ft2a.device, dtype=Ft2a.dtype)
                    Ft2a = torch.cat([Ft2a, pad], dim=1)
            Ft2a = _norm_ch(Ft2a)
        else:
            
            Ft2a = self._align_to_student(Ft2n, Fs2n, None)

        C_proj = (Ft2a.T @ Fs2n) / (Ft2a.shape[0] - 1)
        rowmax_proj = C_proj.abs().max(dim=1).values.mean()

        cos_proj = F.cosine_similarity(F.normalize(Ft2a, dim=1), F.normalize(Fs2n, dim=1), dim=1).mean()

        Qa2, _ = torch.linalg.qr(Ft2a, mode='reduced')
        S2 = torch.linalg.svdvals(Qa2.T @ Qb).clamp(0, 1)
        angle_proj = torch.arccos(S2).mean() * (180.0 / 3.141592653589793)

        return {
            "Diag/rowmax_pre":  float(rowmax_pre.item()),
            "Diag/rowmax_proj": float(rowmax_proj.item()),
            "Diag/cos_base":    float(cos_base.item()),
            "Diag/cos_proj":    float(cos_proj.item()),
            "Diag/angle_base_deg": float(angle_base.item()),
            "Diag/angle_proj_deg": float(angle_proj.item()),
        }

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        
        gt_bbox = gt_dict['search_anno'][-1]  
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'],
                                            self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  

        
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(0.0, 1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss, iou = torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        
        loss_t_rgb = torch.tensor(0.0, device=l1_loss.device)
        loss_t_tir = torch.tensor(0.0, device=l1_loss.device)
        iou_t_rgb, iou_t_tir = None, None

        if "out_t_tir" in pred_dict:
            pred_boxes_t = pred_dict['out_t_tir']['pred_boxes']
            if torch.isnan(pred_boxes_t).any():
                raise ValueError("Teacher TIR outputs is NAN! Stop Training")
            pv = box_cxcywh_to_xyxy(pred_boxes_t).view(-1, 4)
            try:
                giou_loss_t_tir, iou_t_tir = self.objective['giou'](pv, gt_boxes_vec)
            except Exception:
                giou_loss_t_tir, iou_t_tir = torch.tensor(0.0, device=l1_loss.device), torch.tensor(0.0, device=l1_loss.device)
            l1_loss_t_tir = self.objective['l1'](pv, gt_boxes_vec)
            loc_t = self.objective['focal'](pred_dict['out_t_tir']['score_map'], gt_gaussian_maps) if 'score_map' in pred_dict['out_t_tir'] else torch.tensor(0.0, device=l1_loss.device)
            loss_t_tir = self.loss_weight['giou']*giou_loss_t_tir + self.loss_weight['l1']*l1_loss_t_tir + self.loss_weight['focal']*loc_t

        if "out_t_rgb" in pred_dict:
            pred_boxes_t = pred_dict['out_t_rgb']['pred_boxes']
            if torch.isnan(pred_boxes_t).any():
                raise ValueError("Teacher RGB outputs is NAN! Stop Training")
            pv = box_cxcywh_to_xyxy(pred_boxes_t).view(-1, 4)
            try:
                giou_loss_t_rgb, iou_t_rgb = self.objective['giou'](pv, gt_boxes_vec)
            except Exception:
                giou_loss_t_rgb, iou_t_rgb = torch.tensor(0.0, device=l1_loss.device), torch.tensor(0.0, device=l1_loss.device)
            l1_loss_t_rgb = self.objective['l1'](pv, gt_boxes_vec)
            loc_t = self.objective['focal'](pred_dict['out_t_rgb']['score_map'], gt_gaussian_maps) if 'score_map' in pred_dict['out_t_rgb'] else torch.tensor(0.0, device=l1_loss.device)
            loss_t_rgb = self.loss_weight['giou']*giou_loss_t_rgb + self.loss_weight['l1']*l1_loss_t_rgb + self.loss_weight['focal']*loc_t

        
        
        loss_cross = torch.tensor(0.0, device=l1_loss.device)

        if ('aux_dict_rgb' in pred_dict and 'aux_dict_tir' in pred_dict and
            'aux_dict_t_rgb' in pred_dict and 'aux_dict_t_tir' in pred_dict):

            
            x_s_rgb = pred_dict['aux_dict_rgb']['x_list']
            x_s_tir = pred_dict['aux_dict_tir']['x_list']
            x_t_rgb = pred_dict['aux_dict_t_rgb']['x_list']
            x_t_tir = pred_dict['aux_dict_t_tir']['x_list']

            
            
            
            
            

            
            proj_rgb2tir = getattr(self.net, 'align_rgb2tir', None)
            proj_tir2rgb = getattr(self.net, 'align_tir2rgb', None)

            
            diag_stats = {}
            if self._diag_enable and len(self._diag_layers) > 0:
                try:
                    lyr = int(self._diag_layers[0])
                    Ft = x_t_tir[lyr]  
                    Fs = x_s_rgb[lyr]  
                    diag_stats = self._diag_basis(Ft, Fs, proj_tir2rgb, max_tokens=self._diag_max_tokens)
                except Exception:
                    diag_stats = {}



            
            cross_layers_high = [7, 8, 9, 10, 11]
            for i in cross_layers_high:
                feat_t_tir = x_t_tir[i]
                feat_t_rgb = x_t_rgb[i]
                feat_s_rgb = x_s_rgb[i]
                feat_s_tir = x_s_tir[i]

                if self._proj_use:
                    t_tir_to_rgb = self._align_to_student(feat_t_tir, feat_s_rgb, proj_tir2rgb)
                    t_rgb_to_tir = self._align_to_student(feat_t_rgb, feat_s_tir, proj_rgb2tir)
                else:
                    
                    t_tir_to_rgb = self._align_to_student(feat_t_tir, feat_s_rgb, None)
                    t_rgb_to_tir = self._align_to_student(feat_t_rgb, feat_s_tir, None)

                loss_cross +=  self.dftrack_loss.cross_distill_high(
                    t_tir_to_rgb, feat_s_rgb, score_map_gt=gt_gaussian_maps)
                loss_cross +=  self.dftrack_loss.cross_distill_high(
                    t_rgb_to_tir, feat_s_tir, score_map_gt=gt_gaussian_maps)

            
            cross_layers_low = [0, 1, 2]
            for i in cross_layers_low:
                feat_t_tir = x_t_tir[i]
                feat_t_rgb = x_t_rgb[i]
                feat_s_rgb = x_s_rgb[i]
                feat_s_tir = x_s_tir[i]
                loss_cross += self.dftrack_loss.cross_distill_low(
                    feat_t_tir.detach(), feat_s_rgb, score_map_gt=gt_gaussian_maps)
                loss_cross +=  self.dftrack_loss.cross_distill_low(
                    feat_t_rgb.detach(), feat_s_tir, score_map_gt=gt_gaussian_maps)

        
        loss_total = self.loss_weight['giou'] * giou_loss + \
                     self.loss_weight['l1']   * l1_loss   + \
                     self.loss_weight['focal']* location_loss

        
        
        loss = loss_total + loss_t_rgb + loss_t_tir  + 0.01 * loss_cross
        
        lam_map = pred_dict.get('lambda_factor', None)
        if return_status:
            status = {
                "Loss/total": float(loss_total.item()),
                
                "Loss/cross": float(loss_cross.item()),
                "Loss/giou":  float(giou_loss.item()),
                "Loss/l1":    float(l1_loss.item()),
                "Loss/location": float(location_loss.item()),
                "IoU":        float(iou.detach().mean().item()),
            }
            if lam_map is not None:
                status["Gate/mean"] = float(lam_map.mean().item())
            
            if 'diag_stats' in locals() and isinstance(diag_stats, dict):
                status.update(diag_stats)
            return loss, status
        else:
            return loss