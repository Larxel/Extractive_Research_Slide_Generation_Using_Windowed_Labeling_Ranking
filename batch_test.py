import sys
import time
import numpy
import random
import logging
import tensorflow as tf
from batch_data_utils import *


# Variables for inference
cpt = '1625837699.580477'
metafile_str = 'model-7300.meta'


logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
numpy.set_printoptions(threshold=sys.maxsize)

random.seed(2019)
numpy.random.seed(2019)
tf.compat.v1.set_random_seed(2019)

time1 = time.time()
graph = tf.Graph()
vocab = Vocab(params.max_vocab_size, emb_dim=50, dataset_path='data/', glove_path='glove/glove.6B.50d.txt',
              vocab_path='data_files/vocab.txt', lookup_path='data_files/lookup.pkl')

dg = DataGenerator('data/', params.max_inp_seq_len, params.max_out_seq_len, vocab, use_pgen=False,
                   use_sample=False)
with graph.as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        saver = tf.compat.v1.train.import_meta_graph('./runs/' + cpt + '/checkpoints/' + metafile_str)
        module_file = tf.train.latest_checkpoint("./runs/" + cpt + '/checkpoints/')
        saver.restore(sess, module_file)
        input_x = graph.get_operation_by_name("inputs/x_input").outputs[0]
        predict = graph.get_operation_by_name("score_layer/prediction").outputs[0]
        while True:
            (x, y), filenames = dg.get_batch(split='test')
            print(filenames)
            y_ = np.transpose(sess.run(predict, feed_dict={input_x: x}))
            for i, f in enumerate(filenames):
                f = open(f[:-len('.sents.txt')] + "_SummaRunnerScores.txt", 'w')
                for score in y_[i][:]:
                    f.write(str(score) + '\n')
                f.close()
            if dg.ptr == 0:  # finished epoch
                break
