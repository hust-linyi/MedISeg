import sys
sys.path.append('../')
from NetworkTrainer.network_trainer import NetworkTrainer
from NetworkTrainer.options.options import Options

def main():
    opt = Options(isTrain=True)
    opt.parse()
    # CHEANGE
    opt.train['save_dir'] = '{:s}/{:s}/fold_{:d}/{:d}'.format(opt.result_dir, opt.task, opt.fold, opt.train['seed'])  # path to save results
    opt.test['save_dir'] = '{:s}/test_results'.format(opt.train['save_dir'])
    opt.test['checkpoint_dir'] = '{:s}/checkpoints/'.format(opt.train['save_dir'])
    opt.test['model_path'] = '{:s}/checkpoint_{:d}.pth.tar'.format(opt.test['checkpoint_dir'], opt.test['test_epoch'])

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