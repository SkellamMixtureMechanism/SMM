import time
import math
from scipy import stats

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('mx', 1, 'mx for poissonint')
flags.DEFINE_integer('my', 1, 'my for poisson int')
flags.DEFINE_integer('n', 1, 'number of samples generated')

import poisson

# test the poisson distribution
# see if it is indeed a poisson......
# code adapted from https://github.com/IBM/discrete-gaussian-differential-privacy
def main(argv):
    del argv  # argv is not used.
    assert FLAGS.mx is not None, 'Flag mx is missing.'
    assert FLAGS.my is not None, 'Flag my is missing.'
    assert FLAGS.n is not None, 'Flag n is missing.'
    mx = FLAGS.mx
    my = FLAGS.my
    n = FLAGS.n

    test_index = 0
    overall_time = 0
    print('benchmarking time for generating Poisson..... ')
    start = time.time()

    samples = [poisson.PoissonInt(mx,my) for i in range(n)]

    end = time.time()
    elapsed_time = end - start
    print('generated ', n, ' samples in ', elapsed_time, ' seconds.' )

    # process the samples
    samples.sort()
    values = []
    counts = []
    counter = None
    prev = None
    for sample in samples:
        if prev is None: #initializing
            prev=sample
            counter=1
        elif sample==prev: #still same element
            counter=counter+1
        else:
            # add prev to histogram
            values.append(prev)
            counts.append(counter)
            # start counting
            prev=sample
            counter=1
    # add final value
    values.append(prev)
    counts.append(counter)

    # print & sum
    sum = 0
    sumsquared = 0
    kl = 0
    for i in range(len(values)):
        if len(values)<=100: #don't print too much
            print(str(values[i])+":\t"+str(counts[i]))
        sum = sum + values[i]*counts[i]
        sumsquared = sumsquared + values[i]*values[i]*counts[i]
        kl = kl + counts[i]*(math.log(counts[i]/n)-stats.poisson.logpmf(values[i], mx/my))
    mean = sum/n
    var = (sumsquared-sum*sum/n)/(n-1)
    kl = kl/n
    true_mean = mx/my
    true_var = mx/my
    print("mean="+str(float(mean))+" (true="+str(true_mean)+")")
    print("variance="+str(float(var))+" (true="+str(true_var)+")")
    print("KL(empirical||true)="+str(kl))


if __name__ == '__main__':
    app.run(main)
