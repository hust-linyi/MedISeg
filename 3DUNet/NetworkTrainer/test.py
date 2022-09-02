import sys
sys.path.append('../')
from network_infer import NetworkInfer
from options.options import Options

def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    
    inferencer = NetworkInfer(opt)
    inferencer.set_GPU_device()
    inferencer.set_network()
    inferencer.set_dataloader()
    inferencer.run()

if __name__ == "__main__":
    main()