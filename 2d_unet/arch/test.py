import sys
sys.path.append('../')
from NetworkTrainer.network_inference import NetworkInference
from NetworkTrainer.options import Options

def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    inferencer = NetworkInference(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.set_save_dir()
    inferencer.run()

if __name__ == "__main__":
    main()