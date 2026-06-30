import math
import copy
from collections import deque
from torchvision.ops import box_iou
from lib.models.dftrack import build_ostrack_dftrack
from lib.test.tracker.basetracker import BaseTracker
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

        # 特征图尺寸与 hann 窗
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # debug
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

        # ================= 策略参数 =================
        # ST 主路：每 15 帧 + conf>0.60
        self.interval_st = 15
        self.tau_conf_sched = 0.6

        # ST 事件门（“加分”）：更稳时在间隔外补一次
        self.tau_conf_evt = 0.66
        self.tau_peak_evt = 1.20
        self.tau_iou_evt  = 0.55
        self.tau_red_evt  = 0.990

        # Paper-style adaptive event threshold:
        # tau_t = Quantile_q({Phi_{t-j}}_{j=1}^{M}), with default q=0.70, M=30.
        self.event_window_M = int(getattr(params, "event_window_M", 30))
        self.event_quantile_q = float(getattr(params, "event_quantile_q", 0.70))
        self.event_quantile_q = min(max(self.event_quantile_q, 0.0), 1.0)
        self.min_event_history = int(getattr(params, "min_event_history", 5))
        self.event_score_history = deque(maxlen=self.event_window_M)

        # To avoid over-changing the original update behavior, q/M is used as a
        # light auxiliary event trigger rather than replacing the original hard gates.
        self.adaptive_event_margin = float(getattr(params, "adaptive_event_margin", 1.20))
        self.adaptive_tau_red_evt = float(getattr(params, "adaptive_tau_red_evt", 0.995))

        self.cooldown_frames = 8         # 事件更新后的冷却（不限制主路）
        self._cooldown = 0

        # Best(2)：必种子化 + 时间衰减 + 强制刷新 + 短冷却
        self.best_eps = 0.00             # 相对提升阈值：允许“只要不更差就刷新”（0.00）
        self.best_cooldown = 10          # Best 更新冷却缩短
        self.best_force_interval = 60    # 最长 60 帧强制刷新一次 Best（即使没更好）
        self.best_decay = 0.002      # 每帧 0.5% 衰减，降低越到后期越难刷新的问题
        self._last_best = -10**9

        # Diverse(3)：热图差异 OR 框差异 + 放松质量跟随 + 允许同帧 + 短冷却
        self.tau_div_red = 0.9             # 与 Best 的余弦 < 0.94 视为“不同”（常规）
        self.tau_div_red_strict = 0.94       # 若本帧已更新 Best，则用更严门
        self.tau_div_q_ratio = 0.85          # q ≥ 0.85 × best_q（放松质量跟随）
        self.tau_div_iou = 0.60              # 与 Best 框 IoU < 0.60 也判为“不同”（框差异门）
        self.diverse_cooldown_frames = 10    # 短冷却
        self._diverse_cooldown = 0

        # LT（长期锚）默认关闭；稳定后再开
        self.enable_lt_refresh = False
        self.lt_interval = 240
        self.lt_q_gain = 1.7
        self.lt_iou_need = 0.70

        # 运行期状态
        self.prev_state_for_iou = None
        self.prev_score_map = None      # 上一帧热图（加 hann）
        self.best_heatmap = None        # Best 对应的热图（原始 response）
        self.best_state = None          # Best 对应的 bbox（xywh）
        # 槽分数（Best/Diverse 设极低，确保首次可写）
        self.slot_scores = [1.0, 1.0, -1e9, -1e9]

        # 打印原因（False：只打印 1/2/3；True：额外打印跳过原因）
        self.verbose_update_log = False

    # ---------------- 工具函数 ----------------
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
        """N_t = max(N_short, N_best), where N=1-cos(.,.)."""
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
        """tau_t = Quantile_q({Phi_{t-j}}_{j=1}^{M}).

        The current frame is not included in the history when computing tau_t.
        During the first few frames, only the adaptive auxiliary trigger is
        disabled; the original fixed-threshold event route is still available.
        """
        if len(self.event_score_history) < self.min_event_history:
            return float("inf")
        hist = torch.tensor(list(self.event_score_history), dtype=torch.float32)
        return float(torch.quantile(hist, self.event_quantile_q))

    # ---------------- 初始化 ----------------
    def initialize(self, image_v, image_i, info: dict):
        self.temps_queue = []
        self.temps_score = []

        # 截取模板区域
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
                base,                       # slot0: LT
                copy.deepcopy(base),        # slot1: ST
                copy.deepcopy(base),        # slot2: Best（将被首次 ST 替换）
                copy.deepcopy(base)         # slot3: Diverse（后续替换）
            ]

        # 模板 mask（保持你原来只用 RGB 的做法）
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

        # 状态
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
        self.event_score_history = deque(maxlen=self.event_window_M)

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    # ---------------- 槽位更新 ----------------
    def update_template(self, image_v, image_i, new_bbox):
        """只更新 ST（slot1）；Best/Diverse/LT 在 track() 中按条件处理"""
        z_patch_arr_rgb, _, z_amask_arr_rgb = sample_target(
            image_v, new_bbox, self.params.template_factor, output_sz=self.params.template_size)
        z_patch_arr_tir, _, z_amask_arr_tir = sample_target(
            image_i, new_bbox, self.params.template_factor, output_sz=self.params.template_size)

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)

        with torch.no_grad():
            self.temps_queue[1] = [template_rgb, template_tir]  # slot1=ST
        return template_rgb, template_tir

    def compute_confidence_score(self, pred_score_map):
        return pred_score_map.max().item()

    # ---------------- 跟踪 ----------------
    def track(self, image_v, image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1

        # 搜索区域
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(
            image_v, self.state, self.params.search_factor, output_sz=self.params.search_size)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(
            image_i, self.state, self.params.search_factor, output_sz=self.params.search_size)

        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

        with torch.no_grad():
            templates_rgb = torch.stack([t[0].tensors for t in self.temps_queue], dim=0)  # [4,B,C,H,W]
            templates_tir = torch.stack([t[1].tensors for t in self.temps_queue], dim=0)  # [4,B,C,H,W]

            out_dict = self.network.forward(
                template=[templates_rgb, templates_tir],
                search=[search_rgb.tensors, search_tir.tensors],
                ce_template_mask=self.box_mask_z
            )

        # 得分/回归
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx,cy,w,h)
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)

        # 信号
        conf = float(response.max())
        peak = self._heatmap_peakiness(response)
        iou_stable = self._frame_iou(self.state, self.prev_state_for_iou)
        red_prev = self._hm_cosine(response, self.prev_score_map)

        # 冷却递减
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._diverse_cooldown > 0:
            self._diverse_cooldown -= 1

        # Quality / Novelty / Stability cues
        q = self._quality_score(conf, peak)
        novelty = self._novelty_score(response, red_prev)
        stability_gate = 1.0 if iou_stable >= self.tau_iou_evt else 0.0

        # Paper-style event score and adaptive threshold:
        # Phi_t = Q_t * N_t * g(S_t),
        # tau_t = Quantile_q({Phi_{t-j}}_{j=1}^{M}).
        event_score = q * novelty * stability_gate
        event_threshold = self._adaptive_event_threshold()

        # ST：主路 + 轻量自适应事件更新。
        # Keep the original event gate as the main route, and let q/M only add a
        # conservative auxiliary trigger. This prevents q and M from dominating
        # the template update behavior.
        scheduled_st = (self.frame_id % self.interval_st == 0) and (conf > self.tau_conf_sched)
        cooldown_gate = (self._cooldown == 0)
        quality_gate = (conf > self.tau_conf_evt) and (peak > self.tau_peak_evt)
        stable_gate = (iou_stable >= self.tau_iou_evt)
        novelty_gate = (red_prev < self.tau_red_evt)

        # Original fixed-threshold event update.
        fixed_event_st = cooldown_gate and quality_gate and stable_gate and novelty_gate

        # Adaptive auxiliary event update. It is deliberately conservative:
        # 1) it still requires quality/stability/cooldown gates;
        # 2) it uses a relaxed novelty gate instead of removing novelty;
        # 3) Phi_t must exceed the historical quantile with a margin.
        adaptive_ready = len(self.event_score_history) >= self.min_event_history
        adaptive_gate = (
            adaptive_ready
            and (event_score > event_threshold * self.adaptive_event_margin)
        )
        adaptive_novelty_gate = (red_prev < self.adaptive_tau_red_evt)
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
                self._cooldown = self.cooldown_frames

        # Append current Phi_t after the update decision, so tau_t only uses previous frames.
        self.event_score_history.append(float(event_score))

        # ===== Best：时间衰减 + 强制刷新 + 必种子化 =====
        # 先做时间衰减（每帧让 best 分数略降，降低“越到后期越难刷”的门槛）
        if self.slot_scores[2] > -1e8:
            self.slot_scores[2] *= (1.0 - self.best_decay)

        best_updated = False
        # 种子化：第一次 ST 后一定建立 Best
        if updated_st and (self.best_heatmap is None):
            self.temps_queue[2] = [tpl_rgb, tpl_tir]
            self.slot_scores[2] = q
            self.best_heatmap = response.detach()
            self.best_state = list(self.state)
            self._last_best = self.frame_id
            best_updated = True


        # 常规刷新：相对提升 OR 到达强制刷新间隔（并满足冷却）
        if updated_st and (not best_updated) and (self.frame_id - self._last_best >= self.best_cooldown):
            force_due = (self.frame_id - self._last_best >= self.best_force_interval)
            better = (q >= self.slot_scores[2] * (1.0 + self.best_eps))
            if better or force_due:
                self.temps_queue[2] = [tpl_rgb, tpl_tir]
                self.slot_scores[2] = q
                self.best_heatmap = response.detach()
                self.best_state = list(self.state)
                self._last_best = self.frame_id
                best_updated = True


        # ===== Diverse：热图差异 OR 框差异 + 放松质量跟随 + 独立冷却 =====
        if updated_st and (self.best_heatmap is not None) and (self._diverse_cooldown == 0):
            red_vs_best = self._hm_cosine(response, self.best_heatmap)
            iou_vs_best = self._frame_iou(self.state, self.best_state) if self.best_state is not None else 1.0
            # 基于是否同帧更新 Best 选择阈值
            thr = self.tau_div_red_strict if best_updated else self.tau_div_red
            # 两条通路：热图差异 or 框差异
            cond_diff_hm = (red_vs_best < thr)
            cond_diff_box = (iou_vs_best < self.tau_div_iou)
            # 质量跟随（放松）
            cond_q = (q >= self.slot_scores[2] * self.tau_div_q_ratio)
            cond_div = (cond_q and (cond_diff_hm or cond_diff_box))

            if cond_div:
                self.temps_queue[3] = [tpl_rgb, tpl_tir]
                self.slot_scores[3] = q * (1.0 - min(red_vs_best, 0.999))
                self._diverse_cooldown = self.diverse_cooldown_frames

            elif self.verbose_update_log and updated_st:
                print(f"[DBG] diverse_skip red={red_vs_best:.3f} thr={thr:.3f} "
                      f"iou_best={iou_vs_best:.3f} q={q:.3f} best_q={self.slot_scores[2]:.3f}")

        # ===== LT（默认关闭）=====
        if self.enable_lt_refresh and updated_st and (self.frame_id % self.lt_interval == 0):
            if (q > self.lt_q_gain * self.tau_conf_sched * self.tau_peak_evt) and (iou_stable > self.lt_iou_need):
                self.temps_queue[0] = [tpl_rgb, tpl_tir]
                self.slot_scores[0] = q
                # print(0)

        # 维护上一帧状态
        self.prev_state_for_iou = self.state
        self.prev_score_map = response.detach()

        # debug 可视化（保持不变）
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

    # ---------------- 其它工具 ----------------
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


