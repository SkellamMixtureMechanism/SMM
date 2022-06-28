# exact_skellam

An exact sampler for generating **symmetric Skellam** variates in python. 

A **symmetric Skellam** variate is the difference between two independent **Poisson** variates of the same variance.

The algorithm for generating exact **Poisson** variates is based on *Duchon and Duvignau, 2016* (see Algorithm 1 in https://www.combinatorics.org/ojs/index.php/eljc/article/view/v23i4p22/pdf), *Devorye, 1986* (see page 487 in http://www.eirene.de/Devroye.pdf), and the pseudo code by Peter Occil (see https://github.com/peteroupc/peteroupc.github.io/blob/master/randomfunc.md#Poisson_Distribution).

We adopt the convention that </pre><code>randrange()</code></pre> (see https://docs.python.org/3/library/random.html) is the only accessible randomness.
