# Skellam Mixture Mechanism: a Novel Approach to Federated Learning with Differential Privacy

This repository provides the python3 implementation for Skellam Mixture Mechanism.

## Dependencies
scipy,
numpy,
tensorflow,
and log_maths.py for logarithm mathematics as used in tf-privacy.


## SMM for distributed sum estimation

An example python script for computing the privacy guarantee of SMM for distributed sum estimation task is as follows.

</pre><code>python3 smm_dse_analysis.py --gamma=64 --mu=5.95</code></pre>

By default, we estimate the privacy parameter $\epsilon$ with target \$delta\$ set to $10^{-5}$, as we adopt the classic framework (epsilon,delta)-DP. In addition, the data point is sampled i.i.d. from a unit sphere $r=1$. Finally, each client applies a scale parameter of \$gamma=64\$ and injects Skellam noise sampled from $Sk(mu,mu)$ with $mu=5.95$ to the data. The script also returns the l-infinity clipping bound for the data points to achieve the designated privacy guarantee. 

By setting the l-infinity clipping bound to $5.94$, we compute the error of SMM under \$32\$ bitwidth per dimension as follows.

</pre><code>python3 smm_dse.py --gamma=64 --mu=5.95 --l_inf=5.94 --bits=32</code></pre>

## SMM for MNIST

An example python script for computing the privacy guarantee of SMM for MNIST is as follows. 

</pre><code>python3 smm_mnist.py --rounds=1000 --n=240  --c=4096 --gamma=64  --mu=5.95  --bits=8 --l_inf=4.73</code></pre>

Here we run SMM for \$1000\$ rounds, where each round we sample \$240\$ training images (i.e., \$4\$ epochs). The clipping bound for SMM is set to \$4096\$ with l-infinity clipping bound set to \$4.73\$. Here the scale parameter is \$64\$ and the noise parameter is \$5.95\$. The per-parameter communication bitwidth is \$8\$ bits. The privacy guarantee for the overall training process with target \$delta\$ set to $10^{-5} can be obtained from the following script.

</pre><code>python smm_analysis.py --epochs=4  --c=4096  --n=240 --mu=5.95</code></pre>

### SMM using fast Walsh-Hadamard transform

We provide an alternative implementation for SMM with Fast Walsh-Hadamard transform (FWHT, see https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform), instead of direct matrix-vector multiplication. SMM with FWHT saves memory consumption. Our implementation utilizes the FWHT library (see https://github.com/FALCONN-LIB/FFHT). An example script is as follows.

</pre><code>python smm_mnist_ffht.py --rounds=1000 --n=240  --c=4096 --gamma=64  --mu=5.95  --bits=8 --l_inf=4.73</code></pre>


## Exact Skellam Sampler

Script </pre><code>skellam.py</code></pre> under folder </pre><code>exact_skellam</code></pre> benchmarks the running time for generating **symmetric Skellam** variates in python. An example script, which repeatedly generates of $100$ samples from $Sk(\frac{8}{3},\frac{8}{3})$ for $10$ times, is as follows.

</pre><code>python3 skellam.py --mx=8 --my=3 --size=100 --m=10</code></pre>

As a **symmetric Skellam** variate is the difference between two independent **Poisson** variates of the same variance, the task of generating exact Skellam variate reduces to generating exact **Poisson** variates, which is based on *Duchon and Duvignau, 2016* (see Algorithm 1 in https://www.combinatorics.org/ojs/index.php/eljc/article/view/v23i4p22/pdf), *Devorye, 1986* (see page 487 in http://www.eirene.de/Devroye.pdf), and the pseudo code by Peter Occil (see https://github.com/peteroupc/peteroupc.github.io/blob/master/randomfunc.md#Poisson_Distribution). In our implementation for the exact Poisson sampler, we adopt the convention that </pre><code>randrange()</code></pre> (see https://docs.python.org/3/library/random.html) is the only accessible randomness. 

Script </pre><code>poisson_test.py</code></pre> under folder </pre><code>exact_skellam</code></pre> benchmarks the running time for generating **Poisson** variates in python and evaluates the distance from the empirical distribution to the underlying Poisson distribution under KL divergence (see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). An example script, which generates of $100$ samples from $Poisson(\frac{4}{3})$, is as follows.

</pre><code>python3 poisson_test.py --mx=4 --my=3 --n=100</code></pre>

