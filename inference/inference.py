#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os
import argparse
import matplotlib.pyplot as plt
from skimage import io

# User-defined
from network import Network
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def infer(config, directories, ckpt, path, label, args):
    pin_cpu = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':0})
    start = time.time()

    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    # Build graph
    cnn = Model(config, directories, paths=path, labels=label, args=args, single_infer=True)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = cnn.ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=pin_cpu) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        eval_dict = {cnn.training_phase: False, cnn.path: path}

        pred, softmax = sess.run([cnn.pred, cnn.softmax], feed_dict = eval_dict)
        print('Prediction: Class', pred, 'Softmax:', softmax)

        return pred, softmax


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="path to image")
    parser.add_argument("-r", "--restore_path", type=str, help="path to model checkpoint to be restored")
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    args = parser.parse_args()

    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    print(args)
    # Perform inference
    sample_path = pd.Series([args.image_path]).values
    sample_label = pd.Series(['dummy']).values
    pred, score = infer(config_test, directories, ckpt, path=sample_path, label=sample_label, args=args)
    print(pred)
    print(str(score))
    # Temporary 
    # x = io.imread(args.image_path)
    # io.imshow(x, cmap='gray')
    # plt.title('Prediction: {} | Confidence: {}'.format(predicted_string, str(score)))
    # plt.savefig("prediction-{}.pdf".format(predicted_string), format='pdf', dpi=1000)
    # plt.show()

if __name__ == '__main__':
    main()
