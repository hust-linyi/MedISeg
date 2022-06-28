import sys
sys.path.append('../')
from NetworkTrainer.network_infer import NetworkInfer
from NetworkTrainer.options import Options
import os

def main():
    opt = Options(isTrain=True)
    opt.parse()
    # CHEANGE
    opt.train['save_dir'] = '{:s}/{:s}/{:s}/fold_{:d}/{:d}'.format(opt.result_dir, opt.task, opt.model['name'], opt.fold, opt.train['seed'])  # path to save results
    opt.test['save_dir'] = '{:s}/test_results'.format(opt.train['save_dir'])
    opt.test['checkpoint_dir'] = '{:s}/checkpoints/'.format(opt.train['save_dir'])
    opt.test['model_path'] = '{:s}/checkpoint_{:d}.pth.tar'.format(opt.test['checkpoint_dir'], opt.test['test_epoch'])

    opt.save_options()

    inferencer = NetworkInfer(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.run()

if __name__ == "__main__":
    main()