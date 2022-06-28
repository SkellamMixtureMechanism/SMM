import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import hadamard
from scipy import sparse

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

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
python smm_mnist.py --rounds=1000 --n=240  --c=4096 --gamma=64  --mu=5.95  --bits=8 --l_inf=4.73
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

def skSGD(rounds, learning_rate,  n, c, d, mu, bits, gamma, l_inf):
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
    matrix_H = np.array(hadamard(d2))
    print("Dimension for Hadamard matrix is ", d2)
    print("Time taken: %.2fs for Hadamard matrix generation" % (time.time() - start_time))
    for (i,h) in enumerate(matrix_H):
        matrix_H[i] = h * np.random.choice([-1.0,1.0],size=1)
    matrix_H = matrix_H  / np.sqrt(d2)
    matrix_H = np.transpose(matrix_H)
    matrix_HA = tf.convert_to_tensor(matrix_H).astype('float64')
    matrix_HA_T = tf.transpose(matrix_HA)
    print("Transform matrix done.")
    print("Time taken: %.2fs for random rotation matrix generation" % (time.time() - start_time))

    # communication constraint [0,M-1]
    M = np.power(2, bits)

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
            flattened_as_np_d2 = np.zeros(d2)
            flattened_as_np_d2[:d] = flattened_as_np
            flattened_as_tf = tf.convert_to_tensor(flattened_as_np_d2)
            grads_this_round.append(flattened_as_tf)

        # stack the grads for random rotation
        stack_grads = tf.stack(grads_this_round)
        # random rotation
        stack_grads = tf.linalg.matmul(matrix_HA, tf.transpose(stack_grads).astype('float64'))
        # scale
        scale_stack_grads = stack_grads * gamma
        # unstack the grads
        unstack_grads = tf.unstack(scale_stack_grads, axis=1)
        # clipping
        clipped_grads_list = []
        for to_clip_grads in unstack_grads:
            def clip_grad_new(input, c):
                sign = tf.math.sign(input)
                # print('input grads is: ', input)
                abs_input = tf.math.abs(input)
                float_input = tf.math.subtract(abs_input,tf.math.floor(abs_input))
                # print('float of input grads is: ', float_input)
                # compute g^2 , p - p^2
                sqr_input = tf.math.multiply(abs_input,abs_input)
                # print('sqr of input is: ', sqr_input)
                binom_input = tf.math.subtract(float_input,
                    tf.math.multiply(float_input,float_input))
                # print('binom part of input is: ', binom_input)
                # construct vector v
                v = tf.math.add(sqr_input,binom_input)
                # print('v : ', v)
                # compute l_1 sum
                l1_sum = tf.math.reduce_sum(v)
                clip_ratio = min(c/l1_sum, 1)
                # print(' clip ratio: ', clip_ratio)
                # clip by v by l_1
                clipped_v = v * clip_ratio
                # print('clipped_v : ', clipped_v)
                # restore integer
                restore_floor = tf.math.floor(tf.math.sqrt(clipped_v))
                # print('restore_floor : ', restore_floor)
                # restore the float part
                floor_sqr = tf.math.multiply(restore_floor,restore_floor)
                enum = tf.math.subtract(clipped_v,floor_sqr)
                denom = tf.math.add(tf.constant(1.0).astype('float64'),
                    tf.math.scalar_mul(2.0,restore_floor).astype('float64'))
                restore_float = tf.math.divide(enum, denom)
                # print('restore_float : ', restore_float)
                restore = tf.math.add(restore_floor,restore_float)
                # sqr_restore = tf.math.multiply(restore,restore)
                # binom_restore = tf.math.subtract(restore_float,tf.math.multiply(restore_float,restore_float))
                # combined_norm_restore = tf.math.add(sqr_restore,binom_restore)
                # print('restore norm+binom: ', combined_norm_restore)
                # print(' validate with clipped v', clipped_v)
                # restore the sign
                restore = tf.math.multiply(restore,sign)
                # print('restore : ', restore)
                # clip l_inf
                clipped_restore = tf.clip_by_value(restore, -l_inf, l_inf)
                return clipped_restore

            clipped_grads = clip_grad_new(to_clip_grads, c)
            clipped_grads_list.append(clipped_grads)

        # element wise rounding for gradients
        rounded_grads_list = []
        for clipped_grads in clipped_grads_list:
            #print(clipped_grads)
            floor_grads = tf.math.floor(clipped_grads)
            prob_grads = tf.math.subtract(clipped_grads, floor_grads).numpy()
            noise = np.random.binomial(1, prob_grads, prob_grads.shape[0])
            # generate rounded gradients
            rounded_grads = tf.math.add(floor_grads, noise)
            rounded_grads_list.append(rounded_grads.astype('int64'))

        # perturbation for rounded gradients
        perturbed_rounded_grads_list = []
        for rounded_grads in rounded_grads_list:
            poisson_samples_1 = tf.random.poisson([d2], mu).astype('int64')
            perturbed_rounded_grads = tf.math.add(rounded_grads,poisson_samples_1).astype('int64')
            poisson_samples_2 = tf.random.poisson([d2], mu).astype('int64')
            perturbed_rounded_grads = tf.math.subtract(perturbed_rounded_grads,poisson_samples_2).astype('int64')
            # modulo by M
            mod_grads = tf.math.floormod(perturbed_rounded_grads, M)
            perturbed_rounded_grads_list.append(mod_grads)

        # sum
        grad_sum = tf.math.add_n(perturbed_rounded_grads_list)
        grad_sum = tf.math.floormod(grad_sum, M)
        grad_sum = grad_sum.numpy()

        # modulo wraping
        def modmap(x):
            if x <= M/2:
                return x
            else:
                return x-M
        grad_sum = [modmap(i) for i in grad_sum]
        grad_sum = tf.convert_to_tensor(grad_sum).astype('float32')

        # scale back
        grad_avg = tf.math.divide_no_nan(grad_sum, n*gamma).astype('float64')
        # tranform back
        grad_avg = tf.linalg.matvec(matrix_HA_T, tf.transpose(grad_avg))
        #print("Reverting transformation for all clients is done, Time taken: %.2fs" % (time.time() - start_time))
        grad_avg_as_np = (grad_avg.numpy())[:d]

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
            # Update val metrics
            test_acc_metric.update_state(y_batch_test, test_logits)
        test_acc = test_acc_metric.result()
        test_acc_metric.reset_states()
        print("Round ", round, " Test acc: %.4f" % (float(test_acc),), "Time taken: %.2fs" % (time.time() - start_time))
        # print("Round ", round, " finished")
        # print(test_acc)
        # print("Time taken: %.2fs" % (time.time() - start_time))


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

    skSGD(rounds, learning_rate,  n, c, d,  mu, bits, gamma, l_inf)

if __name__ == '__main__':
    app.run(main)
