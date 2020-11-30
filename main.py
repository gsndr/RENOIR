import configparser
import sys
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


np.random.seed(12)
import tensorflow
import tensorflow as tf
tf.random.set_seed(12)
tensorflow.test.is_gpu_available()

#from Siamese import Execution
from Triplet import Execution


def datasetException():
    try:
        dataset=sys.argv[1]

        if (dataset is None) :
            raise Exception()
        if not ((dataset == 'AAGM') or (dataset == 'CICIDS2017')  or (dataset == 'KDDCUP99')):
            raise ValueError()
    except Exception:
        print("The name of dataset is null: use KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    except ValueError:
        print ("Dataset not exist: must be KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    return dataset





def main():

    dataset=datasetException()

    config = configparser.ConfigParser()
    config.read('RENOIR.conf')

    dsConf = config[dataset]
    configuration = config['setting']


    pd.set_option('display.expand_frame_repr', False)



    execution=Execution(dsConf,configuration)

    execution.run()



if __name__ == "__main__":
    main()
