import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation
import seaborn
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--style', dest='style', type=int, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--save', dest='save', type=str, default=None)
parser.add_argument('--batch', dest='is_batch', action='store_true', default=False)
parser.add_argument('--filepath', dest='csv_path', type=str, default=None)
args = parser.parse_args()


def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)

def sample_text(sess, args_text, translation, style=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0] # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text] # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: args.bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not args.force and finish[0, 0] > 0.8:
                print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords
# honestly, just use this fuction once...
def read_csv(filepath):
    import csv
    sample = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["word"]
            text = text[0].upper() + text[1:]
            sample.append(text)
    return sample

def generate_image(text, style, bias, styles, sess, translation):
    out = os.path.join('samples', text.replace(' ', '_') + '_BIAS{}_STYLE{}.png'.format(bias, style))
    style = [styles[0][style], styles[1][style]]
    phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, text, translation, style)

    strokes = np.array(stroke_data)
    epsilon = 1e-8
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
    minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
    miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

    fig, ax = plt.subplots(1, 1)
    for stroke in split_strokes(cumsum(np.array(coords))):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    # ax.set_title('Handwriting')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig(out, bbox_inches='tight', format='png', pad_inches=0)
    # plt.show()
    


def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    if args.csv_path is not None:
        # read in csv and specify
        sample_input = read_csv(args.csv_path) 
        for string in sample_input:
            args.text = string 


    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)
        
        #load styles file
        with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
            styles = pickle.load(file)
        
        if args.csv_path is not None:
            sample_input = read_csv(args.csv_path)

            # iterate over inputs specifying text and output path
            for x in sample_input:
                for i in range(5):
                    generate_image(x, i, np.random.normal(3, 0.25), styles, sess, translation)


if __name__ == '__main__':
    main()
