from easydict import EasyDict as edict
import yaml


cfg = edict()


cfg.MODEL = edict()

cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.RGB_BRANCH = False
cfg.MODEL.TRAIN_RGB_BRANCH = True
cfg.MODEL.TIR_BRANCH = False
cfg.MODEL.TRAIN_TIR_BRANCH = True
cfg.MODEL.RGB_TEACHER = False
cfg.MODEL.TRAIN_RGB_TEACHER = True
cfg.MODEL.TIR_TEACHER = False
cfg.MODEL.TRAIN_TIR_TEACHER = True

cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = [2, 5, 8, 11]


cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  


cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

cfg.MODEL.SHARE_STUDENT = False         



cfg.TRAIN = edict()
cfg.TRAIN.DFTrack_LOSS = ""                 
cfg.TRAIN.ENABLE_CONTENT = True         
cfg.TRAIN.ENABLE_STYLE = True
cfg.TRAIN.MASK_PROBABILITY = 0.0        
cfg.TRAIN.INPUT_MASK_RATIO = 0.0        
cfg.TRAIN.PARAM_KEY = False             
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1

cfg.TRAIN.STOP_CONTENT_GRADIENT = True  
cfg.TRAIN.STOP_STYLE_GRADIENT = False
cfg.TRAIN.STYLE_LOSS_TYPE = "channel-level"     
cfg.TRAIN.CONTENT_LOSS_TYPE = "channel-level"

cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.STYLE_DISTILL_WEIGHT = 0.01
cfg.TRAIN.CONTENT_DISTILL_WEIGHT = 0.1

cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 2
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  
cfg.TRAIN.CE_WARM_EPOCH = 80  
cfg.TRAIN.DROP_PATH_RATE = 0.1  


cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1


cfg.DATA = edict()
cfg.DATA.ENABLE_AUG1 = False
cfg.DATA.ENABLE_NOISE = False
cfg.DATA.SAMPLER_MODE = "causal"  
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LasHeR_trainingSet"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["LasHeR_testingSet"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1

cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0


cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.TEMPLATE_UPDATE = edict()
cfg.TEST.TEMPLATE_UPDATE.INTERVAL_ST = 15
cfg.TEST.TEMPLATE_UPDATE.TAU_CONF_SCHED = 0.6
cfg.TEST.TEMPLATE_UPDATE.TAU_CONF_EVT = 0.66
cfg.TEST.TEMPLATE_UPDATE.TAU_PEAK_EVT = 1.20
cfg.TEST.TEMPLATE_UPDATE.TAU_IOU_EVT = 0.55
cfg.TEST.TEMPLATE_UPDATE.TAU_RED_EVT = 0.990
cfg.TEST.TEMPLATE_UPDATE.EVENT_WINDOW_M = 30
cfg.TEST.TEMPLATE_UPDATE.EVENT_QUANTILE_Q = 0.70
cfg.TEST.TEMPLATE_UPDATE.MIN_EVENT_HISTORY = 5
cfg.TEST.TEMPLATE_UPDATE.ADAPTIVE_EVENT_MARGIN = 1.20
cfg.TEST.TEMPLATE_UPDATE.ADAPTIVE_TAU_RED_EVT = 0.995
cfg.TEST.TEMPLATE_UPDATE.COOLDOWN_FRAMES = 8
cfg.TEST.TEMPLATE_UPDATE.BEST_EPS = 0.00
cfg.TEST.TEMPLATE_UPDATE.BEST_COOLDOWN = 10
cfg.TEST.TEMPLATE_UPDATE.BEST_FORCE_INTERVAL = 60
cfg.TEST.TEMPLATE_UPDATE.BEST_DECAY = 0.002
cfg.TEST.TEMPLATE_UPDATE.TAU_DIV_RED = 0.9
cfg.TEST.TEMPLATE_UPDATE.TAU_DIV_RED_STRICT = 0.94
cfg.TEST.TEMPLATE_UPDATE.TAU_DIV_Q_RATIO = 0.85
cfg.TEST.TEMPLATE_UPDATE.TAU_DIV_IOU = 0.60
cfg.TEST.TEMPLATE_UPDATE.DIVERSE_COOLDOWN_FRAMES = 10
cfg.TEST.TEMPLATE_UPDATE.ENABLE_LT_REFRESH = False
cfg.TEST.TEMPLATE_UPDATE.LT_INTERVAL = 240
cfg.TEST.TEMPLATE_UPDATE.LT_Q_GAIN = 1.7
cfg.TEST.TEMPLATE_UPDATE.LT_IOU_NEED = 0.70
cfg.TEST.TEMPLATE_UPDATE.HEATMAP_PEAK_TOPK = 5
cfg.TEST.TEMPLATE_UPDATE.VERBOSE_UPDATE_LOG = False


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
