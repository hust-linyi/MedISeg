import sys
sys.path.append('../')
import os
from NetworkTrainer.options import Options
from NetworkTrainer.network_inference import test_calculate_metric

if __name__ == '__main__':
    opt = Options(isTrain=False)
    opt.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])
    metric = test_calculate_metric(opt)
