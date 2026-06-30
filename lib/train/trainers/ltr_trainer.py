
import os
import datetime
from collections import OrderedDict

import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
from lib.utils.misc import get_world_size


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _get_model_and_igf(actor_net):
    
    mod = _unwrap(actor_net)
    igf = getattr(mod, "igf_module", None) or getattr(mod, "igf", None) or getattr(mod, "gate", None)
    return mod, igf


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            os.makedirs(tensorboard_writer_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir,
                                                cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        
        try:
            model = getattr(self.actor, 'net', None)
            if model is not None:
                mod, igf = _get_model_and_igf(model)
                if igf is not None:
                    
                    for _, p in igf.named_parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                    
                    igf_params = [p for _, p in igf.named_parameters() if p.requires_grad]
                    ids_in_opt = set(id(p) for g in self.optimizer.param_groups for p in g['params'])
                    missing = [p for p in igf_params if id(p) not in ids_in_opt]
                    if missing:
                        base_lr = self.optimizer.param_groups[0].get('lr', 1e-4)
                        base_wd = self.optimizer.param_groups[0].get('weight_decay', 1e-4)
                        self.optimizer.add_param_group({'params': missing, 'lr': base_lr, 'weight_decay': base_wd})
        except Exception:
            pass

        
        self._proj_fixed = False  
        
        self._proj_debug = getattr(settings, 'proj_debug', False)

    
    def _proj_modules(self):
        
        net = _unwrap(self.actor.net)
        mods = OrderedDict()
        for name in ("align_rgb2tir", "align_tir2rgb"):
            m = getattr(net, name, None)
            if m is not None and hasattr(m, "weight"):
                mods[name] = m
        return mods

    def _ensure_projector_trainable(self):
        
        mods = self._proj_modules()
        if not mods:
            return
        
        proj_params = []
        for _, m in mods.items():
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                proj_params.append(p)
        
        ids_in_opt = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        to_add = [p for p in proj_params if id(p) not in ids_in_opt]
        if to_add:
            base_lr = float(self.optimizer.param_groups[0].get('lr', 1e-4))
            self.optimizer.add_param_group({"params": to_add, "lr": base_lr * 5.0, "weight_decay": 0.0})
            if self._proj_debug and self.settings.local_rank in [-1, 0]:
                print(f"[PROJ] add {len(to_add)} params to optimizer (lr={base_lr*5.0:.2e}, wd=0.0)", flush=True)
        else:
            if self._proj_debug and self.settings.local_rank in [-1, 0]:
                print("[PROJ] projector params already in optimizer.", flush=True)

    

    def _set_default_settings(self):
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}
        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        
        if loader.training and not self._proj_fixed:
            try:
                self._ensure_projector_trainable()
            finally:
                self._proj_fixed = True

        clean_grad = True

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            if self.move_data_to_gpu:
                data = data.to(self.device)
            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            
            if loader.training:
                if clean_grad:
                    self.optimizer.zero_grad()
                    clean_grad = False

                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                else:
                    self.scaler.scale(loss).backward()

                if not self.use_amp:
                    acc_steps = getattr(self.actor.cfg.TRAIN, 'accumulate_grad_batches', 1)
                    do_step = (acc_steps > 0 and (i % acc_steps == 0))
                    if do_step:
                        self.optimizer.step()
                        clean_grad = True
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            if clean_grad:
                self._print_stats(i, loader, batch_size)

            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        if loader.training:
            if self.lr_scheduler is not None:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            with open(self.settings.log_file, 'a') as f:
                f.write(print_str[:-5] + '\n')

    def _stats_new_epoch(self):
        for loader in self.loaders:
            if loader.training:
                if self.lr_scheduler is not None:
                    try:
                        lr_list = self.lr_scheduler.get_last_lr()
                    except:
                        lr_list = self.lr_scheduler._get_lr(self.epoch)
                    for i, lr in enumerate(lr_list):
                        var_name = 'LearningRate/group{}'.format(i)
                        if var_name not in self.stats[loader.name].keys():
                            self.stats[loader.name][var_name] = StatValue()
                        self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)