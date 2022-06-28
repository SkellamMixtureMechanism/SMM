# logarithmic calculations from tf-privacy

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import sys

import numpy as np
from scipy import special

def _log_add(logx, logy):
    # Add two numbers in the log space
    # log(x+y)
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
    # Subtract two numbers in the log space. Answer must be non-negative.
    # log(x-y)
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.
    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_sub_sign(logx, logy):
    # Returns log(exp(logx)-exp(logy)) and its sign.
    if logx > logy:
        s = True
        mag = logx + np.log(1 - np.exp(logy - logx))
    elif logx < logy:
        s = False
        mag = logy + np.log(1 - np.exp(logx - logy))
    else:
        s = True
        mag = -np.inf
    return s, mag


def _log_print(logx):
    # pretty print x, i.e., exp(logx)
    if logx < math.log(sys.float_info.max):
        return "{}".format(math.exp(logx))
    else:
        return "exp({})".format(logx)


def _log_comb(n, k):
    # the logarithm (base e) of n choose k
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1))

