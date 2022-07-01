import sys
sys.path.append('../')
from NetworkTrainer.network_inference import NetworkInference
from NetworkTrainer.options import Options
import os

def main():
    opt = Options(isTrain=True)
    opt.parse()

    # CHEANGE
    opt.test['save_dir'] = os.path.join(opt.result_dir, opt.task, f"fold_{opt.fold}", f"{opt.train['seed']}", 'test_results')
    opt.test['model_path'] = os.path.join(opt.result_dir, 'ensembleinit', opt.model['name'], f"fold_{opt.fold}", f"{opt.train['seed']}", 'checkpoints', f"checkpoint_{opt.test['test_epoch']:d}.pth.tar")
    opt.save_options()

    inferencer = NetworkInference(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.set_save_dir()
    inferencer.run()

if __name__ == "__main__":
    main()