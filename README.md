# Skellam Mixture Mechanism: a Novel Approach to Federated Learning with Differential Privacy

This repository provides Skellam Mixture Mechanism.

## SMM for distributed sum estimation

An example python script for computing the privacy guarantee of SMM for distributed sum estimation task is as follows.

</pre><code>python3 smm_dse_analysis.py --gamma=64 --mu=5.95</code></pre>

By default, we estimate the privacy parameter $\epsilon$ with target \$delta\$ set to $10^{-5}$, as we adopt the classic framework (epsilon,delta)-DP. In addition, the data point is sampled i.i.d. from a unit sphere $r=1$. Finally, each client applies a scale parameter of \$gamma=64\$ and injects Skellam noise sampled from $Sk(mu,mu)$ with $mu=5.95$ to the data. The script also returns the l-infinity clipping bound for the data points. 

By setting the l-infinity clipping bound to $5.94$, we compute the error of SMM under 32 bit communication bandwidth per dimension as follows.

</pre><code>python3 smm_dse.py --gamma=64 --mu=5.95 --l_inf=5.94 --bits=32</code></pre>

## SMM for MNIST

### SMM using fast Walsh-Hadamard transform

## Exact Skellam Sampler

Script </pre><code>skellam.py</code></pre> under folder </pre><code>exact_skellam</code></pre> benchmarks the running time for generating **symmetric Skellam** variates in python. An example script, which repeatedly generates of $100$ samples from $Sk(\frac{8}{3},\frac{8}{3})$ for $10$ times, is as follows.

</pre><code>python3 skellam.py --mx=8 --my=3 --size=100 --m=10</code></pre>

As a **symmetric Skellam** variate is the difference between two independent **Poisson** variates of the same variance, the task of generating exact Skellam variate reduces to generating exact **Poisson** variates, which is based on *Duchon and Duvignau, 2016* (see Algorithm 1 in https://www.combinatorics.org/ojs/index.php/eljc/article/view/v23i4p22/pdf), *Devorye, 1986* (see page 487 in http://www.eirene.de/Devroye.pdf), and the pseudo code by Peter Occil (see https://github.com/peteroupc/peteroupc.github.io/blob/master/randomfunc.md#Poisson_Distribution). In our implementation for the exact Poisson sampler, we adopt the convention that </pre><code>randrange()</code></pre> (see https://docs.python.org/3/library/random.html) is the only accessible randomness. 

Script </pre><code>poisson_test.py</code></pre> under folder </pre><code>exact_skellam</code></pre> benchmarks the running time for generating **Poisson** variates in python and evaluates the distance from the empirical distribution to the underlying Poisson distribution under KL divergence (see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). An example script, which generates of $100$ samples from $Poisson(\frac{4}{3})$, is as follows.

</pre><code>python3 poisson_test.py --mx=4 --my=3 --n=100</code></pre>

