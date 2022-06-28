# Skellam Mixture Mechanism: a Novel Approach to Federated Learning with Differential Privacy

This repository provides Skellam Mixture Mechanism.

## SMM

## Exact Skellam Sampler

Script </pre><code>skellam.py</code></pre> under folder </pre><code>exact_skellam</code></pre> benchmarks the running time for generating **symmetric Skellam** variates in python. 

An example script, which simulates the generation of $100$ samples is as follows.

</pre><code>python skellam.py --mx=8 --my=3 --size=100 --m=10</code></pre>

As a **symmetric Skellam** variate is the difference between two independent **Poisson** variates of the same variance, the task of generating exact Skellam variate reduces to generating exact **Poisson** variates, which is based on *Duchon and Duvignau, 2016* (see Algorithm 1 in https://www.combinatorics.org/ojs/index.php/eljc/article/view/v23i4p22/pdf), *Devorye, 1986* (see page 487 in http://www.eirene.de/Devroye.pdf), and the pseudo code by Peter Occil (see https://github.com/peteroupc/peteroupc.github.io/blob/master/randomfunc.md#Poisson_Distribution).

In our implementation, we adopt the convention that </pre><code>randrange()</code></pre> (see https://docs.python.org/3/library/random.html) is the only accessible randomness.

