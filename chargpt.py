"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

import clip

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()
    C.data.clip_token = 1000000

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    C.model.clip_token = C.data.clip_token
    C.model.clip_dim = 512
    C.model.block_size = 512

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 4096
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import wandb
    wandb.init(project="clip-gpt")

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size() + 1
    model = GPT(config.model, train_dataset.itos, train_dataset.stoi)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        wandb.log({'loss': trainer.loss.item(), 'iter': trainer.iter_num, 'iter_dt': trainer.iter_dt})

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            if False:
                with torch.no_grad():
                    # sample from the model...
                    context = "O God, O God!"
                    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                    y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                    completion = ''.join([train_dataset.itos[int(i)] for i in y])
                    print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            state_dict = {k:v for k, v in model.state_dict().items() if 'clip_model' not in k}
            torch.save(state_dict, ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
