import copy
from torchvision.ops import box_iou
from lib.models.dftrack import build_ostrack_dftrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.template_update import TemplateUpdateParams
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_dftrack(params.cfg, training=False)
        try:
            m, n = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
            if m != []:
                raise m
        except:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes

        self.template_update = TemplateUpdateParams.from_cfg(self.cfg.TEST.TEMPLATE_UPDATE)
        self.event_score_history = self.template_update.new_event_score_history()
        self._cooldown = 0
        self._last_best = -10**9
        self._diverse_cooldown = 0
        self.prev_state_for_iou = None
        self.prev_score_map = None
        self.best_heatmap = None
        self.best_state = None
        self.slot_scores = [1.0, 1.0, -1e9, -1e9]

    def _xywh_to_xyxy(self, box):
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def _frame_iou(self, box_a, box_b):
        if box_a is None or box_b is None:
            return 1.0
        a = torch.tensor([self._xywh_to_xyxy(box_a)], dtype=torch.float32)
        b = torch.tensor([self._xywh_to_xyxy(box_b)], dtype=torch.float32)
        return float(box_iou(a, b)[0, 0])

    def _heatmap_peakiness(self, hm: torch.Tensor, topk: int = 5):
        v = hm.reshape(-1).float()
        k = min(topk, v.numel())
        topk_vals, _ = torch.topk(v, k=k)
        top1 = topk_vals[0]
        mean_topk = topk_vals.mean()
        return float(top1 / (mean_topk + 1e-6))

    def _hm_cosine(self, a: torch.Tensor, b: torch.Tensor):
        if a is None or b is None:
            return 0.0
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(F.cosine_similarity(a, b, dim=0))

    def _quality_score(self, conf: float, peak: float):
        return conf * peak

    def _novelty_score(self, response: torch.Tensor, red_prev: float):
        novelty_values = []
        if self.prev_score_map is not None:
            sim_prev = max(min(red_prev, 0.999), -1.0)
            novelty_values.append(max(0.0, 1.0 - sim_prev))

        if self.best_heatmap is not None:
            red_best = self._hm_cosine(response, self.best_heatmap)
            sim_best = max(min(red_best, 0.999), -1.0)
            novelty_values.append(max(0.0, 1.0 - sim_best))

        return max(novelty_values) if len(novelty_values) > 0 else 0.0

    def _adaptive_event_threshold(self):
        template_update = self.template_update
        if len(self.event_score_history) < template_update.min_event_history:
            return float("inf")
        hist = torch.tensor(list(self.event_score_history), dtype=torch.float32)
        return float(torch.quantile(hist, template_update.event_quantile_q))

    def initialize(self, image_v, image_i, info: dict):
        self.temps_queue = []
        self.temps_score = []

        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(
            image_v, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(
            image_i, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)

        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)

        with torch.no_grad():
            base = [template_rgb, template_tir]
            self.temps_queue = [
                base,
                copy.deepcopy(base),
                copy.deepcopy(base),
                copy.deepcopy(base)
            ]

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor_rgb, template_rgb.tensors.device).squeeze(1)
            self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

            template_bbox_tir = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor_tir, template_tir.tensors.device).squeeze(1)
            self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            self.box_mask_z = [self.box_mask_z_rgb, self.box_mask_z_tir]
            self.box_mask_z = self.box_mask_z[0]

        self.state = info['init_bbox']
        self.frame_id = 0
        self.prev_state_for_iou = info['init_bbox']
        self.prev_score_map = None
        self.best_heatmap = None
        self.best_state = None
        self.slot_scores = [1.0, 1.0, -1e9, -1e9]
        self._cooldown = 0
        self._diverse_cooldown = 0
        self._last_best = -10**9
        self.event_score_history = self.template_update.new_event_score_history()

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def update_template(self, image_v, image_i, new_bbox):
        z_patch_arr_rgb, _, z_amask_arr_rgb = sample_target(
            image_v, new_bbox, self.params.template_factor, output_sz=self.params.template_size)
        z_patch_arr_tir, _, z_amask_arr_tir = sample_target(
            image_i, new_bbox, self.params.template_factor, output_sz=self.params.template_size)

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)

        with torch.no_grad():
            self.temps_queue[1] = [template_rgb, template_tir]
        return template_rgb, template_tir

    def compute_confidence_score(self, pred_score_map):
        return pred_score_map.max().item()

    def track(self, image_v, image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        template_update = self.template_update

        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(
            image_v, self.state, self.params.search_factor, output_sz=self.params.search_size)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(
            image_i, self.state, self.params.search_factor, output_sz=self.params.search_size)

        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

        with torch.no_grad():
            templates_rgb = torch.stack([t[0].tensors for t in self.temps_queue], dim=0)
            templates_tir = torch.stack([t[1].tensors for t in self.temps_queue], dim=0)

            out_dict = self.network.forward(
                template=[templates_rgb, templates_tir],
                search=[search_rgb.tensors, search_tir.tensors],
                ce_template_mask=self.box_mask_z
            )

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor_rgb).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)

        conf = float(response.max())
        peak = self._heatmap_peakiness(response, template_update.heatmap_peak_topk)
        iou_stable = self._frame_iou(self.state, self.prev_state_for_iou)
        red_prev = self._hm_cosine(response, self.prev_score_map)

        if self._cooldown > 0:
            self._cooldown -= 1
        if self._diverse_cooldown > 0:
            self._diverse_cooldown -= 1

        q = self._quality_score(conf, peak)
        novelty = self._novelty_score(response, red_prev)
        stability_gate = 1.0 if iou_stable >= template_update.tau_iou_evt else 0.0

        event_score = q * novelty * stability_gate
        event_threshold = self._adaptive_event_threshold()

        scheduled_st = (self.frame_id % template_update.interval_st == 0) and (conf > template_update.tau_conf_sched)
        cooldown_gate = (self._cooldown == 0)
        quality_gate = (conf > template_update.tau_conf_evt) and (peak > template_update.tau_peak_evt)
        stable_gate = (iou_stable >= template_update.tau_iou_evt)
        novelty_gate = (red_prev < template_update.tau_red_evt)

        fixed_event_st = cooldown_gate and quality_gate and stable_gate and novelty_gate

        adaptive_ready = len(self.event_score_history) >= template_update.min_event_history
        adaptive_gate = (
            adaptive_ready
            and (event_score > event_threshold * template_update.adaptive_event_margin)
        )
        adaptive_novelty_gate = (red_prev < template_update.adaptive_tau_red_evt)
        adaptive_event_st = (
            cooldown_gate
            and quality_gate
            and stable_gate
            and adaptive_novelty_gate
            and adaptive_gate
        )

        event_st = fixed_event_st or adaptive_event_st

        updated_st, tpl_rgb, tpl_tir = False, None, None
        if scheduled_st or event_st:
            tpl_rgb, tpl_tir = self.update_template(image_v, image_i, self.state)
            updated_st = True

            if event_st:
                self._cooldown = template_update.cooldown_frames

        self.event_score_history.append(float(event_score))

        if self.slot_scores[2] > -1e8:
            self.slot_scores[2] *= (1.0 - template_update.best_decay)

        best_updated = False
        if updated_st and (self.best_heatmap is None):
            self.temps_queue[2] = [tpl_rgb, tpl_tir]
            self.slot_scores[2] = q
            self.best_heatmap = response.detach()
            self.best_state = list(self.state)
            self._last_best = self.frame_id
            best_updated = True

        if updated_st and (not best_updated) and (self.frame_id - self._last_best >= template_update.best_cooldown):
            force_due = (self.frame_id - self._last_best >= template_update.best_force_interval)
            better = (q >= self.slot_scores[2] * (1.0 + template_update.best_eps))
            if better or force_due:
                self.temps_queue[2] = [tpl_rgb, tpl_tir]
                self.slot_scores[2] = q
                self.best_heatmap = response.detach()
                self.best_state = list(self.state)
                self._last_best = self.frame_id
                best_updated = True

        if updated_st and (self.best_heatmap is not None) and (self._diverse_cooldown == 0):
            red_vs_best = self._hm_cosine(response, self.best_heatmap)
            iou_vs_best = self._frame_iou(self.state, self.best_state) if self.best_state is not None else 1.0
            thr = template_update.tau_div_red_strict if best_updated else template_update.tau_div_red
            cond_diff_hm = (red_vs_best < thr)
            cond_diff_box = (iou_vs_best < template_update.tau_div_iou)
            cond_q = (q >= self.slot_scores[2] * template_update.tau_div_q_ratio)
            cond_div = (cond_q and (cond_diff_hm or cond_diff_box))

            if cond_div:
                self.temps_queue[3] = [tpl_rgb, tpl_tir]
                self.slot_scores[3] = q * (1.0 - min(red_vs_best, 0.999))
                self._diverse_cooldown = template_update.diverse_cooldown_frames

            elif template_update.verbose_update_log and updated_st:
                print(f"[DBG] diverse_skip red={red_vs_best:.3f} thr={thr:.3f} "
                      f"iou_best={iou_vs_best:.3f} q={q:.3f} best_q={self.slot_scores[2]:.3f}")

        if template_update.enable_lt_refresh and updated_st and (self.frame_id % template_update.lt_interval == 0):
            if (q > template_update.lt_q_gain * template_update.tau_conf_sched * template_update.tau_peak_evt) and (
                    iou_stable > template_update.lt_iou_need):
                self.temps_queue[0] = [tpl_rgb, tpl_tir]
                self.slot_scores[0] = q

        self.prev_state_for_iou = self.state
        self.prev_score_map = response.detach()

        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'search_region_t')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'template_t')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz),
                                     'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [ri.cpu().numpy() for ri in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                if 'removed_indexes_s' in out_dict.get('aux_dict_rgb', {}) and out_dict['aux_dict_rgb']['removed_indexes_s']:
                    removed_indexes_s = [ri.cpu().numpy() for ri in out_dict['aux_dict_rgb']['removed_indexes_s']]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search_v')

                if 'removed_indexes_s' in out_dict.get('aux_dict_tir', {}) and out_dict['aux_dict_tir']['removed_indexes_s']:
                    removed_indexes_s = [ri.cpu().numpy() for ri in out_dict['aux_dict_tir']['removed_indexes_s']]
                    masked_search = gen_visualization(x_patch_arr_tir, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search_i')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb,
                                                resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack_twobranch
