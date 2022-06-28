# implementing SMM using FFHT https://github.com/FALCONN-LIB/FFHT
# with much less memory cost than matrix multiplication

import numpy as np
import time

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

# ffht
# need to install the ffht library in
# https://github.com/FALCONN-LIB/FFHT
import ffht

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('d', 63610, 'dimension of the model')
flags.DEFINE_integer('rounds', 1000, 'Number of rounds')
flags.DEFINE_integer('n', 240, 'batch size per round')
flags.DEFINE_float('learning_rate', 0.005, 'learning_rate')
# clipping thresholds
flags.DEFINE_float('c', 2000, 'clipping c')
flags.DEFINE_float('l_inf', 1, 'l_inf bound')
# quantization ratio
flags.DEFINE_integer('gamma', 10, 'quantization ratio')
# skellam noise parameter for each participant, SK(mu,mu) := Pois(mu) - Pois(mu)
flags.DEFINE_float('mu', 5.18, 'mean of poisson for each party')
# communication constraint
flags.DEFINE_integer('bits', 12, 'number of bits per dimension')

"""
python smm_mnist_ffht.py --rounds=1000 --n=240  --c=4096 --gamma=64  --mu=5.95  --bits=8 --l_inf=4.73
"""

# hard code the overall number of training data
N_train = 60000


def generate_train_and_test_images():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.astype(np.float64).reshape(N_train, 784, )
    train_images = train_images / 255
    test_images = test_images.astype(np.float64).reshape(10000, 784, )
    test_images = test_images / 255
    return train_images, train_labels, test_images, test_labels

def smm(rounds, learning_rate,  n, c, d, mu, bits, gamma, l_inf):
    # generate images for training and testing
    train_images, train_labels, test_images, test_labels = generate_train_and_test_images()
    # prepare test dataset for tf
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(test_images.shape[0])

    # GET MODEL
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(80, activation="relu", name="dense_1")(inputs)
    outputs = layers.Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    """ TRAIN """
    start_time = time.time()

    def least_power_2_upper_bound(d):
        upper_bound = 1
        while upper_bound < d:
            upper_bound = upper_bound * 2
        return upper_bound
    d2 = least_power_2_upper_bound(d)

    # communication constraint [0,M-1]
    M = np.power(2, bits)

    # for random Walsh-hadamard transform
    sign_vec = np.random.choice([-1.0,1.0],size=1)

    for round in range(rounds):
        # store flattened gradients for this round
        grads_this_round = []

        # sample a subset of data
        indices = np.random.permutation(N_train)[:n]
        indices_image = [train_images[i] for i in indices]
        indices_label = [train_labels[i] for i in indices]
        train_dataset = tf.data.Dataset.from_tensor_slices((indices_image, indices_label))
        micro_batch_size = 1
        train_dataset = train_dataset.batch(micro_batch_size)

        for party, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # compute gradient
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)

            # flatten
            grads_as_list = tf.nest.flatten(grads)
            shape_as_list_of_np = []
            flattened_as_list_of_np = []
            for layer_step, grad in enumerate(grads_as_list):
                shape_as_list_of_np.append(grad.numpy())
                flattened_as_list_of_np.append(grad.numpy().flatten())
            flattened_as_np = np.concatenate(flattened_as_list_of_np)
            # pad 0 for hadamard transform
            flattened_as_np_d2 = np.pad(flattened_as_np, (0, d2-d), 'constant')
            grads_this_round.append(flattened_as_np_d2)

        # clipping
        clipped_grads_list = []
        for to_clip_grads in grads_this_round:
            def clip_grad_new(input, c):
                sign = np.sign(input)
                abs_input = np.absolute(input)
                float_input = np.subtract(abs_input, np.floor(abs_input))
                sqr_input = np.multiply(abs_input,abs_input)
                binom_input = np.subtract(float_input,
                    np.multiply(float_input,float_input))
                v = np.add(sqr_input,binom_input)
                l1_sum = np.sum(v)
                clip_ratio = min(c/l1_sum, 1)
                clipped_v = v * clip_ratio
                restore_floor = np.floor(np.sqrt(clipped_v))
                floor_sqr = np.multiply(restore_floor,restore_floor)
                enum = np.subtract(clipped_v,floor_sqr)
                denom = np.add(1, np.multiply(2.0,restore_floor))
                restore_float = np.divide(enum, denom)
                restore = np.add(restore_floor,restore_float)
                restore = np.multiply(restore,sign)
                clipped_restore = np.clip(restore, -l_inf, l_inf)
                return clipped_restore

            scaled_grads = np.multiply(to_clip_grads, gamma)
            flip_grads = np.multiply(scaled_grads, sign_vec)
            ffht.fht(flip_grads)
            flip_grads = np.divide(flip_grads, np.sqrt(d2))
            clipped_grads = clip_grad_new(flip_grads, c)
            clipped_grads_list.append(clipped_grads)

        # element wise rounding for gradients
        rounded_grads_list = []
        for clipped_grads in clipped_grads_list:
            floor_grads = np.floor(clipped_grads)
            prob_grads = np.subtract(clipped_grads, floor_grads)
            noise = np.random.binomial(1, prob_grads, prob_grads.shape[0])
            # generate rounded gradients
            rounded_grads = np.add(floor_grads, noise)
            rounded_grads_list.append(rounded_grads)

        # perturbation for rounded gradients
        perturbed_rounded_grads_list = []
        for rounded_grads in rounded_grads_list:
            poisson_samples_1 = np.random.poisson(mu, d2)
            perturbed_rounded_grads = np.add(rounded_grads,poisson_samples_1)
            poisson_samples_2 = np.random.poisson(mu, d2)
            perturbed_rounded_grads = np.subtract(perturbed_rounded_grads,poisson_samples_2)
            # modulo by M
            mod_grads = np.mod(perturbed_rounded_grads, M)
            perturbed_rounded_grads_list.append(mod_grads)

        # sum
        grad_sum = np.sum(perturbed_rounded_grads_list, axis=0, dtype=np.int64)
        grad_sum = np.mod(grad_sum, M)

        # modulo wraping
        def modmap(x):
            if x <= M/2:
                return x
            else:
                return x-M
        grad_sum = [modmap(i) for i in grad_sum]
        grad_avg = np.divide(grad_sum, n*gamma)
        # reverse random Walsh-Hadamard transform
        ffht.fht(grad_avg)
        grad_avg = np.divide(grad_avg, np.sqrt(d2))
        grad_avg = np.multiply(grad_avg, sign_vec)
        grad_avg_as_np = grad_avg[:d]

        # convert back to tensor
        grads_list = []
        current_index = 0
        for i in range(len(shape_as_list_of_np)):
            this_length = flattened_as_list_of_np[i].shape[0]
            flattened_grad_as_np = grad_avg_as_np[current_index : current_index+this_length]
            grad_as_np = np.reshape(flattened_grad_as_np,shape_as_list_of_np[i].shape)
            grads_list.append(tf.convert_to_tensor(grad_as_np).astype('float32'))
            current_index = current_index + this_length

        # unflatten gradient list to gradient
        gradient_avg = tf.nest.pack_sequence_as(grads, grads_list, expand_composites=False)

        # update model
        optimizer.apply_gradients(zip(gradient_avg, model.trainable_weights))

        # print accuracy.
        for x_batch_test, y_batch_test in test_dataset:
            test_logits = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_logits)
        test_acc = test_acc_metric.result()
        test_acc_metric.reset_states()
        print("Round ", round, " Test acc: %.4f" % (float(test_acc),), "Time taken: %.2fs" % (time.time() - start_time))


def main(argv):
    np_config.enable_numpy_behavior()

    del argv  # argv is not used.
    assert FLAGS.rounds is not None, 'Flag rounds is missing.'
    assert FLAGS.learning_rate is not None, 'Flag learning_rate is missing.'
    assert FLAGS.n is not None, 'Flag n is missing.'
    assert FLAGS.c is not None, 'Flag c is missing.'
    assert FLAGS.d is not None, 'Flag d is missing.'
    assert FLAGS.mu is not None, 'Flag mu is missing.'
    assert FLAGS.bits is not None, 'Flag bits is missing.'
    assert FLAGS.gamma is not None, 'Flag gamma is missing.'
    assert FLAGS.l_inf is not None, 'Flag l_inf is missing.'

    rounds = FLAGS.rounds
    learning_rate = FLAGS.learning_rate
    n = FLAGS.n
    c = FLAGS.c
    d = FLAGS.d
    mu = FLAGS.mu
    bits = FLAGS.bits
    gamma = FLAGS.gamma
    l_inf = FLAGS.l_inf

    smm(rounds, learning_rate,  n, c, d,  mu, bits, gamma, l_inf)

if __name__ == '__main__':
    app.run(main)
