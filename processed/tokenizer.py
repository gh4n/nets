import pandas as pd
import tensorflow as tf
import argparse, os
from sklearn.model_selection import train_test_split

def tokenize(df):
    tkz =  tf.keras.preprocessing.text.Tokenizer(1200, oov_token=1201)
    tkz.fit_on_texts(df['text'])
    df['tokens'] = tkz.texts_to_sequences(df['text'])
    lengths = df['tokens'].apply(len)
    df = df[lengths <= 15]

    return df

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input csv')
    args = parser.parse_args()

    df = pd.read_hdf(args.input)
    df = tokenize(df)
    out = os.path.splitext(os.path.basename(args.input))[0] + '_tokenized_'

    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_hdf(out+'train.h5', key='df')
    df_test.to_hdf(out+'test.h5', key='df')

if __name__ == '__main__':
    main()
