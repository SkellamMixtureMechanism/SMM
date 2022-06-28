import numpy as np
from scipy.linalg import hadamard
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n', 100, 'number of clients(data)')
flags.DEFINE_float('r', 1, 'l2-norm')
flags.DEFINE_integer('d', 65536, 'dimension')
flags.DEFINE_float('mu', None, 'mu on each client')
# scaling parameter
flags.DEFINE_float('gamma', 64, 'scaling parameter')
flags.DEFINE_float('l_inf', 1, 'linf clip')
flags.DEFINE_integer('bits', 32, 'number_of_bits_perdimension')

"""
Example:
 python3 smm_dse.py --gamma=64 --mu=5.95 --l_inf=5.94 --bits=32
"""

def sample_spherical(npoints, ndim, r=1):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec = vec*r
    return vec

def main(argv):
    del argv  # argv is not used.
    assert FLAGS.n is not None, 'Flag n is missing.'
    assert FLAGS.r is not None, 'Flag r is missing.'
    assert FLAGS.d is not None, 'Flag d is missing.'
    assert FLAGS.mu is not None, 'Flag mu is missing.'
    assert FLAGS.gamma is not None, 'Flag gamma is missing.'
    assert FLAGS.l_inf is not None, 'Flag l_inf is missing.'
    assert FLAGS.bits is not None, 'Flag bits is missing.'

    n = FLAGS.n
    r = FLAGS.r
    d = FLAGS.d
    mu = FLAGS.mu
    gamma = FLAGS.gamma
    l_inf = FLAGS.l_inf
    bits = FLAGS.bits

    c = gamma * gamma

    M = np.power(2, bits)
    print('communication constraint m is ', M)

    samples = sample_spherical(n,d,r)
    true_sum = np.sum(np.transpose(samples), axis=0)
    print('true sum shape: ', true_sum.shape)
    print('true sum: ', true_sum)

    # hadamard
    matrix_H = np.array(hadamard(d))
    # print("Dimension for Hadamard matrix is ", d)
    for (i,h) in enumerate(matrix_H):
        matrix_H[i] = h * np.random.choice([-1.0,1.0],size=1)
    matrix_H = matrix_H  / np.sqrt(d)
    matrix_H = np.transpose(matrix_H)
    matrix_HA = tf.convert_to_tensor(matrix_H).astype('float64')
    matrix_HA_T = tf.transpose(matrix_HA)
    # print("Transform matrix done.")

    samples_as_tf = []
    for sample in samples:
        # scale
        sample_as_tf = tf.convert_to_tensor(sample*gamma)
        samples_as_tf.append(sample_as_tf)
    stack_samples = tf.stack(samples_as_tf)
    #print('stack grads', tf.shape(grads_for_a_party))
    stack_samples = tf.linalg.matmul(matrix_HA, stack_samples.astype('float64'))
    #print("Flattening done, Time taken: %.2fs" % (time.time() - start_time))
    unstack_samples = tf.unstack(stack_samples, axis=1)

    clipped_samples_list = []
    for to_clip_grads in unstack_samples:
        # print(to_clip_grads.numpy().shape)
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
            # print(clipped_restore)
            return clipped_restore

        clipped_grads = clip_grad_new(to_clip_grads, c)
        clipped_samples_list.append(clipped_grads)

    # element wise rounding for gradients
    rounded_samples_list = []
    for clipped_grads in clipped_samples_list:
        #print(clipped_grads)
        floor_grads = tf.math.floor(clipped_grads)
        prob_grads = tf.math.subtract(clipped_grads, floor_grads).numpy()
        noise = np.random.binomial(1, prob_grads, prob_grads.shape[0])
        # generate rounded gradients
        rounded_grads = tf.math.add(floor_grads, noise)
        rounded_samples_list.append(rounded_grads.astype('int64'))

    perturbed_samples = []
    for rounded_grads in rounded_samples_list:
        poisson_samples_1 = tf.random.poisson([d], mu).astype('int64')
        # print(poisson_samples_1)
        perturbed_rounded_grads = tf.math.add(rounded_grads,poisson_samples_1).astype('int64')
        poisson_samples_2 = tf.random.poisson([d], mu).astype('int64')
        # print(poisson_samples_2)
        perturbed_rounded_grads = tf.math.subtract(perturbed_rounded_grads,poisson_samples_2).astype('int64')
        mod_grads = tf.math.floormod(perturbed_rounded_grads, M)
        # print('mod grads: ', mod_grads)
        perturbed_samples.append(mod_grads)
    perturbed_sum = tf.math.add_n(perturbed_samples)
    # print('scaled sum: ', perturbed_sum)

    perturbed_sum = tf.math.floormod(perturbed_sum, M)
    perturbed_sum = perturbed_sum.numpy()
    # # modulo wraping
    def modmap(x):
        if x <= M/2:
            return x
        else:
            return x-M
    perturbed_sum = [modmap(i) for i in perturbed_sum]
    # print('unwrapped scaled sum: ', perturbed_sum)
    perturbed_sum = tf.convert_to_tensor(np.divide(perturbed_sum, gamma))
    perturbed_sum = tf.linalg.matvec(matrix_HA_T, tf.transpose(perturbed_sum))
    perturbed_sum_as_np = (perturbed_sum.numpy())
    err_vec =  perturbed_sum_as_np-true_sum
    err = np.sum([i*i for i in err_vec])/d
    print('err is ', err)


if __name__ == '__main__':
    app.run(main)
