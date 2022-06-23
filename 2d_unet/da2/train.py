import sys
sys.path.append('../')
from NetworkTrainer.network_trainer import NetworkTrainer
from NetworkTrainer.options import Options

def main():
    opt = Options(isTrain=True)
    opt.parse()
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