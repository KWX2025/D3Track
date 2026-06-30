from collections import deque
from dataclasses import dataclass


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class TemplateUpdateParams:
    interval_st: int
    tau_conf_sched: float
    tau_conf_evt: float
    tau_peak_evt: float
    tau_iou_evt: float
    tau_red_evt: float
    event_window_m: int
    event_quantile_q: float
    min_event_history: int
    adaptive_event_margin: float
    adaptive_tau_red_evt: float
    cooldown_frames: int
    best_eps: float
    best_cooldown: int
    best_force_interval: int
    best_decay: float
    tau_div_red: float
    tau_div_red_strict: float
    tau_div_q_ratio: float
    tau_div_iou: float
    diverse_cooldown_frames: int
    enable_lt_refresh: bool
    lt_interval: int
    lt_q_gain: float
    lt_iou_need: float
    heatmap_peak_topk: int
    verbose_update_log: bool

    @classmethod
    def from_cfg(cls, cfg):
        event_quantile_q = float(cfg.EVENT_QUANTILE_Q)
        event_quantile_q = min(max(event_quantile_q, 0.0), 1.0)
        return cls(
            interval_st=int(cfg.INTERVAL_ST),
            tau_conf_sched=float(cfg.TAU_CONF_SCHED),
            tau_conf_evt=float(cfg.TAU_CONF_EVT),
            tau_peak_evt=float(cfg.TAU_PEAK_EVT),
            tau_iou_evt=float(cfg.TAU_IOU_EVT),
            tau_red_evt=float(cfg.TAU_RED_EVT),
            event_window_m=int(cfg.EVENT_WINDOW_M),
            event_quantile_q=event_quantile_q,
            min_event_history=int(cfg.MIN_EVENT_HISTORY),
            adaptive_event_margin=float(cfg.ADAPTIVE_EVENT_MARGIN),
            adaptive_tau_red_evt=float(cfg.ADAPTIVE_TAU_RED_EVT),
            cooldown_frames=int(cfg.COOLDOWN_FRAMES),
            best_eps=float(cfg.BEST_EPS),
            best_cooldown=int(cfg.BEST_COOLDOWN),
            best_force_interval=int(cfg.BEST_FORCE_INTERVAL),
            best_decay=float(cfg.BEST_DECAY),
            tau_div_red=float(cfg.TAU_DIV_RED),
            tau_div_red_strict=float(cfg.TAU_DIV_RED_STRICT),
            tau_div_q_ratio=float(cfg.TAU_DIV_Q_RATIO),
            tau_div_iou=float(cfg.TAU_DIV_IOU),
            diverse_cooldown_frames=int(cfg.DIVERSE_COOLDOWN_FRAMES),
            enable_lt_refresh=_to_bool(cfg.ENABLE_LT_REFRESH),
            lt_interval=int(cfg.LT_INTERVAL),
            lt_q_gain=float(cfg.LT_Q_GAIN),
            lt_iou_need=float(cfg.LT_IOU_NEED),
            heatmap_peak_topk=int(cfg.HEATMAP_PEAK_TOPK),
            verbose_update_log=_to_bool(cfg.VERBOSE_UPDATE_LOG),
        )

    def new_event_score_history(self):
        return deque(maxlen=self.event_window_m)
