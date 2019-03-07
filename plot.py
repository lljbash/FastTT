#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def exp11():
    t_sample_fasttt = [
        116, 113, 115, 107, 109,
        212, 205, 198, 168, 186,
        402, 276, 378, 301, 274,
        415, 377, 393, 448, 394,
        691, 594, 623, 597, 626,
        918, 858, 862, 812, 791,
        1242, 1200, 1198, 1144, 1238,
        1535, 1485, 1421, 1452, 1523,
        2076, 1860, 1905, 2046, 1973,
        2558, 2279, 2248, 2364, 2371,
    ]
    d_sample_fasttt = [
        15, 15, 15, 15, 15,
        16, 16, 16, 16, 16,
        17, 17, 17, 17, 17,
        18, 18, 18, 18, 18,
        19, 19, 19, 19, 19,
        20, 20, 20, 20, 20,
        21, 21, 21, 21, 21,
        22, 22, 22, 22, 22,
        23, 23, 23, 23, 23,
        24, 24, 24, 24, 24,
    ]
    t_average_fasttt = [
	112, 193.8, 326.2, 405.4, 626.2, 848.2, 1204.4, 1483.2, 1972, 2364, 
    ]
    d_average_fasttt = [
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ]
    t_sample_ttsvd = [
        87, 90, 89, 91, 90,
        170, 157, 197, 158, 214,
        325, 293, 295, 293, 286,
        528, 582, 544, 538, 542,
        1124, 1059, 1054, 1027, 1045,
        1832, 1952, 1875, 1890, 1928,
        3688, 3722, 3763, 3609, 3669,
        7241, 7272, 7066, 7224, 7314,
    ]
    d_sample_ttsvd = [
        15, 15, 15, 15, 15,
        16, 16, 16, 16, 16,
        17, 17, 17, 17, 17,
        18, 18, 18, 18, 18,
        19, 19, 19, 19, 19,
        20, 20, 20, 20, 20,
        21, 21, 21, 21, 21,
        22, 22, 22, 22, 22,
    ]
    t_average_ttsvd = [
        89.4, 179.2, 298.4, 546.8, 1061.8, 1895.4, 3690.2, 7223.4,
    ]
    d_average_ttsvd = [
        15, 16, 17, 18, 19, 20, 21, 22
    ]
    plt.scatter(d_sample_ttsvd, t_sample_ttsvd, c='r', alpha=0.3, label='TT-SVD Samples')
    plt.scatter(d_sample_fasttt, t_sample_fasttt, c='b', alpha=0.3, label='FastTT Samples')
    plt.plot(d_average_ttsvd, t_average_ttsvd, c='r', label='TT-SVD Average')
    plt.plot(d_average_fasttt, t_average_fasttt, c='b', label='FastTT Average')
    plt.legend()
    plt.xlabel('d')
    plt.ylabel('Runtime / s')
    plt.show()


def exp12():
    t_sample_fasttt = [
        918, 858, 862, 812, 791,
        1543, 1550, 1442, 1574, 1697,
        2014, 1933, 1994, 1945, 1885,
        2278, 2217, 2228, 2225, 2164,
        2496, 2431, 2407, 2493, 2513,
        2660, 2576, 2549, 2616, 2530,
        2567, 2695, 2587, 2609, 2598,
        2653, 2707, 2652, 2694, 2756,
        2772, 2709, 2706, 2756, 2692,
        2763, 2734, 2677, 2810, 2758,
    ]
    t_average_fasttt = [
        848.2, 1561.2, 1954.2, 2222.4, 2468, 2586.2, 2611.2, 2692.4, 2727, 2748.4,
    ]
    t_sample_ttsvd = [
        1832, 1952, 1875, 1890, 1928,
        2471, 2407, 2508, 2490, 2490,
        2829, 2758, 2718, 2720, 2694,
        2837, 2907, 2889, 2897, 2909,
        2949, 2933, 3003, 2889, 2905,
        2926, 2897, 2967, 2930, 2882,
        2928, 2976, 2907, 2876, 2990,
        2899, 2980, 2933, 3008, 2926,
        2904, 2993, 3030, 2972, 3053,
        2990, 2996, 3018, 3005, 2928,
    ]
    t_average_ttsvd = [
        1895.4, 2473.2, 2743.8, 2887.8, 2935.8, 2920.4, 2935.4, 2949.2, 2990.4, 2987.4,
    ]
    N_sample = [
        500, 500, 500, 500, 500,
        1000, 1000, 1000, 1000, 1000,
        1500, 1500, 1500, 1500, 1500,
        2000, 2000, 2000, 2000, 2000,
        2500, 2500, 2500, 2500, 2500,
        3000, 3000, 3000, 3000, 3000,
        3500, 3500, 3500, 3500, 3500,
        4000, 4000, 4000, 4000, 4000,
        4500, 4500, 4500, 4500, 4500,
        5000, 5000, 5000, 5000, 5000,
    ]
    N_average = [
        500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
    ]
    plt.scatter(N_sample, t_sample_ttsvd, c='r', alpha=0.3, label='TT-SVD Samples')
    plt.scatter(N_sample, t_sample_fasttt, c='b', alpha=0.3, label='FastTT Samples')
    plt.plot(N_average, t_average_ttsvd, c='r', label='TT-SVD Average')
    plt.plot(N_average, t_average_fasttt, c='b', label='FastTT Average')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Runtime / s')
    plt.show()


def exp4():
    p = list(range(7))
    e = [37.300, 37.297, 141.587, 36.487, 13.022, 12.508, 12.506]
    t = [30.116, 30.407, 100.480, 17.392, 9.808, 9.184, 8.889]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(p, e, c='r', label='Estimated FLOPs')
    ax1.set_ylabel(r"GFLOP / $\sigma$")
    ax1.set_ylim([0, 150])
    ax2 = ax1.twinx()
    lns2 = ax2.plot(p, t, c='b', label='Exact Runtime')
    ax2.set_ylabel("Runtime / s")
    ax2.set_ylim([0, 150])
    ax2.set_xlabel("p")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.show()


if __name__ == "__main__":
    # exp11()
    # exp12()
    exp4()

