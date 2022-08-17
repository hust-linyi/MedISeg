import sys
sys.path.append('../')
from NetworkTrainer.network_infer import NetworkInfer
from NetworkTrainer.options.options import Options
import os

def main():
    opt = Options(isTrain=True)
    opt.parse()
    # DEBUG
    opt.test['save_dir'] = os.path.join(opt.result_dir, opt.task, f"fold_{opt.fold}", f"{opt.train['seed']}", 'test_results')
    # opt.test['model_path'] = os.path.join(opt.result_dir, 'ensembleinit', f"fold_{opt.fold}", f"{opt.train['seed']}", 'checkpoints', f"checkpoint_{opt.test['test_epoch']:d}.pth.tar")
    opt.test['model_path'] = os.path.join('/mnt/yfs/ianlin/Experiment/LIVER/', 'ensembleinit', f"fold_{opt.fold}", f"{opt.train['seed']}", 'checkpoints', f"checkpoint_{opt.test['test_epoch']:d}.pth.tar")
    
    opt.save_options()

    inferencer = NetworkInfer(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.run()

if __name__ == "__main__":
    main()