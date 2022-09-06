"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import pickle
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.accumulate_grad_batches = 1
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, **extra_state):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        for k,v in extra_state.items():
            setattr(self, k, v)

    def load(self, save_path):
        # set attributes that were saved
        with open(save_path, 'rb') as f:
            save_dict = pickle.load(f)
        self.optimizer_state_dict = save_dict['optimizer']
        del save_dict['optimizer']
        for k,v in save_dict.items():
            setattr(self, k, v)
        return self

    def save(self, save_path):
        # parse out objects that can be pickled and pickle them
        def is_pickleable(obj):
            # quack quack quack
            try:
                pickle.dumps(obj)
                return True
            except AttributeError:
                return False
        save_dict = {k:v for k,v in self.__dict__.items() if is_pickleable(v)}
        del save_dict['train_dataset'] # this is a huge storage waste
        del save_dict['callbacks'] # these are dynamic and can't be pickled
        if 'optimizer' in save_dict:
            save_dict['optimizer'] = self.optimizer.state_dict()
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        if hasattr(self, 'optimizer_state_dict'):
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size//config.accumulate_grad_batches,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            self.grad_accumulate_batch = 0
            self.loss = 0
            model.zero_grad(set_to_none=True)
            while self.grad_accumulate_batch < config.accumulate_grad_batches:
                # fetch the next batch (x, y) and re-init iterator if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                logits, loss = model(x, y)

                # backprop
                loss = loss/config.accumulate_grad_batches
                loss.backward()
                self.loss += loss.detach() # this would cause a memory leak if we didn't detach
                self.grad_accumulate_batch += 1
            # and update the parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
