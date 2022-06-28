import numpy as np
from log_maths import _log_add
from log_maths import _log_sub
from log_maths import _log_comb
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 4, 'Number of epochs')
flags.DEFINE_integer('N', 60000, 'Overall number of parties')
flags.DEFINE_integer('d', 63610, 'dimension of the gradient')
flags.DEFINE_float('c', None, 'clipping threshold')
flags.DEFINE_integer('n', None, 'number of clients in each round')
flags.DEFINE_float('mu', None, 'mean of poisson noise from each client injected')
flags.DEFINE_float('delta', 1e-5, 'Target delta')
# scaling parameter
flags.DEFINE_float('gamma', 64, 'scaling parameter')

"""
Example:
  python smm_analysis.py --epochs=4  --c=4096  --n=240 --mu=5.95
"""

def rdp_sk(alpha, mu, c):
    eps = (1.2*alpha+1) / 4 / mu * c
    # print("eps is: ", eps)
    return eps

def rdp_sk_sampling(q, alpha, mu, c):
    current_log_sum = (alpha-1)*np.log(1-q) + np.log(alpha*q-q+1)
    #print(np.exp(current_log_sum))
    for j in range(2,alpha+1):
        current_term = _log_comb(alpha, j) + \
                        (alpha-j)*np.log(1-q) + j*np.log(q) + \
                        (j-1) * rdp_sk(j, mu, c)
        current_log_sum = _log_add(current_log_sum, current_term)
    return current_log_sum / (alpha-1)

def main(argv):
    del argv
    assert FLAGS.d is not None, 'Flag d is missing.'
    assert FLAGS.N is not None, 'Flag N is missing.'
    assert FLAGS.c is not None, 'Flag c is missing.'
    assert FLAGS.n is not None, 'Flag n is missing.'
    assert FLAGS.epochs is not None, 'Flag epochs is missing.'
    assert FLAGS.mu is not None, 'Flag mu is missing.'
    assert FLAGS.delta is not None, 'Flag delta is missing.'
    assert FLAGS.gamma is not None, 'Flag gamma is missing.'

    d = FLAGS.d
    N = FLAGS.N
    c = FLAGS.c
    n = FLAGS.n
    mu = FLAGS.mu
    epochs = FLAGS.epochs
    delta = FLAGS.delta
    gamma = FLAGS.gamma

    q = 1.0 * n / N
    rounds = N / n * epochs
    overall_mu = mu * n # overall mu across all clients

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,100))
    print('Privacy analysis for runnning SMM on:', N, ' clients, for',
            epochs, ' epochs,\ni.e., ', rounds, ' rounds with ',
            n, 'clients sampled at each round with probability ',
            q, '\nClipping threshold c is set to: ', c)
    for alpha in alpha_candidates:
        current_eps = rdp_sk_sampling(q,alpha, overall_mu, c)
        current_eps = current_eps*rounds + \
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)
    l_inf = min(2*overall_mu/best_alpha, np.sqrt(4*overall_mu/ \
                            (10.9 * best_alpha*best_alpha -1.8 * best_alpha - 9.1)))
    print('L_inf should be set to: ', l_inf)

if __name__ == '__main__':
    app.run(main)
