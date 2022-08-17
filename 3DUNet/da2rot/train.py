import sys
sys.path.append('../')
from NetworkTrainer.network_trainer import NetworkTrainer
import os
from NetworkTrainer.options.options import Options
from NetworkTrainer.dataloaders.data_kit import *

def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.transform['train'] =[
                RandomCrop(opt.model['input_size']),
                RandomRotation(),
                ToTensor()
            ]
    opt.save_options()

    trainer = NetworkTrainer(opt)
    trainer.set_GPU_device()
    trainer.set_logging()
    trainer.set_randomseed()
    trainer.set_network()
    trainer.set_loss()
    trainer.set_optimizer()
    trainer.set_dataloader()
    trainer.run()

if __name__ == "__main__":
    main()