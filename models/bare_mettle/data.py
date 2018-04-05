#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_data(filename):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)

        return df['tokens'].values, df['category'].values

    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, batch_size, test=False):
    
        # def _preprocess(tokens, label):
        #     return tokens, label

        dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # dataset = dataset.map(_preprocess)
        dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=(0,0))

        if test:
            dataset = dataset.repeat()

        return dataset
