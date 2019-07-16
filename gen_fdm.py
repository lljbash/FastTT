#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

alpha = np.random.rand(7)
alpha /= np.linalg.norm(alpha, 1)
n = 40

def index_to_position(index):
    p = 0
    a, b, c, d, e, f = index
    index = [a, d, b, e, c, f]
    for i in index:
        p = p * n + i
    return p

if __name__ == "__main__":
    with open("fdm.tsv", "w") as f:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    alpha = np.random.rand(7)
                    alpha /= np.linalg.norm(alpha, 1)
                    p = index_to_position([i, j, k, i, j, k])
                    print("{}\t{}".format(p, alpha[0]), file=f)
                    if i - 1 >= 0:
                        p = index_to_position([i, j, k, i - 1, j, k])
                        print("{}\t{}".format(p, alpha[1]), file=f)
                    if i + 1 < n:
                        p = index_to_position([i, j, k, i + 1, j, k])
                        print("{}\t{}".format(p, alpha[2]), file=f)
                    if j - 1 >= 0:
                        p = index_to_position([i, j, k, i, j - 1, k])
                        print("{}\t{}".format(p, alpha[3]), file=f)
                    if j + 1 < n:
                        p = index_to_position([i, j, k, i, j + 1, k])
                        print("{}\t{}".format(p, alpha[4]), file=f)
                    if k - 1 >= 0:
                        p = index_to_position([i, j, k, i, j, k - 1])
                        print("{}\t{}".format(p, alpha[5]), file=f)
                    if k + 1 < n:
                        p = index_to_position([i, j, k, i, j, k + 1])
                        print("{}\t{}".format(p, alpha[6]), file=f)

