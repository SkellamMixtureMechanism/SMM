import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float('r', 1, 'radius')
flags.DEFINE_integer('n', 100, 'number of clients in each round')
flags.DEFINE_float('mu', None, 'mean of poisson noise from each client injected')
flags.DEFINE_float('delta', 1e-5, 'Target delta')
# scaling parameter
flags.DEFINE_float('gamma', 64, 'scaling parameter')

"""
Example:
  python3 smm_dse_analysis.py --gamma=64 --mu=5.95
"""

def rdp_sk(alpha, mu, c):
    eps = (1.2*alpha+1) / 4 / mu * c
    # print("eps is: ", eps)
    return eps

def main(argv):
    del argv
    assert FLAGS.r is not None, 'Flag r is missing.'
    assert FLAGS.n is not None, 'Flag n is missing.'
    assert FLAGS.mu is not None, 'Flag mu is missing.'
    assert FLAGS.delta is not None, 'Flag delta is missing.'
    assert FLAGS.gamma is not None, 'Flag gamma is missing.'

    r = FLAGS.r
    n = FLAGS.n
    mu = FLAGS.mu
    delta = FLAGS.delta
    gamma = FLAGS.gamma
    c = r*r*gamma*gamma

    overall_mu = mu * n # overall mu across all clients

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,100))
    print('Privacy analysis for runnning SMM on:', n, ' clients,')
    for alpha in alpha_candidates:
        current_eps = rdp_sk(alpha, overall_mu, c)
        current_eps = current_eps + \
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)
    l_inf = min(2*overall_mu/best_alpha, np.sqrt(4*overall_mu/ \
                            (10.9 * best_alpha*best_alpha -1.8 * best_alpha - 9.1)))
    print('clipping threshold is set to: ', c)
    print('L_inf should be set to: ', l_inf)

if __name__ == '__main__':
    app.run(main)
