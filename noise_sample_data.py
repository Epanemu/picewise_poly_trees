# original implementation taken from the repositiory of the authors
# https://github.com/hernanmp/Partition_recovery/blob/main/utils.R
# datagenerator rewritten from R

import numpy as np


def generate_data(
    n: int, scenario: int, sigma: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # theta0 = matrix(0,n,n) To my understanding, theta0 is the ground truth
    ground_truth = np.zeros((n, n))

    if scenario == 1:
        # I disregard minor shifts caused by use of </<=... and changes in indexing
        ground_truth[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1

    elif scenario == 2:
        for i in range(n):
            for j in range(n):
                if (i - n / 4) ** 2 + (j - n / 4) ** 2 < (n / 5) ** 2:
                    ground_truth[i, j] = 1
                if (i - 3 * n / 4) ** 2 + (j - 3 * n / 4) ** 2 < (n / 5) ** 2:
                    ground_truth[i, j] = -1

    elif scenario == 3:
        ground_truth[n // 4 : 3 * n // 4, n // 4 : (n // 4 + n // 8)] = 1
        ground_truth[(n // 2 + n // 8) : 3 * n // 4, (n // 4 + n // 8) : 3 * n // 4] = 1
        ground_truth[6 * n // 8 :, 6 * n // 8 :] = -1

    elif scenario == 4:
        ground_truth[: n // 5, : n // 5] = 1
        ground_truth[: n // 5, 4 * n // 5 :] = 2
        ground_truth[4 * n // 5 :, : n // 5] = 3
        ground_truth[4 * n // 5 :, 4 * n // 5 :] = 4
        ground_truth[
            n // 2 - n // 8 : n // 2 + n // 8, n // 2 - n // 8 : n // 2 + n // 8
        ] = 5

    elif scenario == 5:  # not presented in the original paper
        # ground_truth[:n//5, :n//5] = 1 # is commented out there as well.
        ground_truth[: n // 5, 2 * n // 5 :] = 2
        ground_truth[4 * n // 5 :, : 3 * n // 5] = 3
        r = round(n / 4.5)
        ground_truth[n // 2 - r : n // 2 + r, n // 2 - r : n // 2 + r] = 4

    vals = np.unique(ground_truth)
    patches = {}
    for v in vals:
        patches[v] = ground_truth == v

    # y = ground_truth + sigma*matrix(rnorm(n*n),n,n)
    noisy_data = ground_truth + sigma * np.random.normal(0, 1, (n, n))

    return ground_truth, noisy_data, patches
