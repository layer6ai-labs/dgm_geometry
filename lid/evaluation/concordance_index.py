"""
The implementation of a concordance index metric for LID estimation
This evaluates how good our LID estimators rank inputs based on their LID values,
for example, in cases where LID is used as a proxy for "complexity" this can be used
to validate estimators.
Other applications include LID for OOD detection, or LID.
"""

import numpy as np

# _fenwick tree code taken from https://gist.github.com/rajatdiptabiswas/79fc1ce5cf410df4139b291f75bf0794


def _fenwick_update(index, value, array, bi_tree):
    """
    Updates the binary indexed tree with the given value
    :param index: index at which the update is to be made
    :param value: the new element at the index
    :param array: the input array
    :param bi_tree: the array representation of the binary indexed tree
    :return: void
    """
    while index < len(array):
        bi_tree[index] += value
        index += index & -index


def _fenwick_get_sum(index, bi_tree):
    """
    Calculates the sum of the elements from the beginning to the index
    :param index: index till which the sum is to be calculated
    :param bi_tree: the array representation of the binary indexed tree
    :return: (integer) sum of the elements from beginning till index
    """
    ans = 0

    while index > 0:
        ans += bi_tree[index]
        index -= index & -index

    return ans


def _fenwick_get_range_sum(left, right, bi_tree):
    """
    Calculates the sum from the given range

    :param bi_tree: the array representation of the binary indexed tree
    :param left: left index of the range (1-indexed)
    :param right: right index of the range (1-indexed)
    :return: (integer) sum of the elements in the range
    """
    ans = _fenwick_get_sum(right, bi_tree) - _fenwick_get_sum(left - 1, bi_tree)

    return ans


def _process_equals(a_list: np.array, b_list: np.array):
    """
    Count the number of pairs that are equal in a_list but
    unequal in b_list in O(n) time
    """
    pairs = list(zip(a_list, b_list))
    pairs.sort()
    invalid_pairs = 0
    current_dict = {}
    for i in range(len(pairs) + 1):
        if i == len(pairs) or (i > 0 and pairs[i][0] != pairs[i - 1][0]):
            sm = 0
            for key in current_dict:
                sm += current_dict[key]
                invalid_pairs -= current_dict[key] * (current_dict[key] - 1) // 2
            invalid_pairs += sm * (sm - 1) // 2
            current_dict = {}
        if i == len(pairs):
            break
        pred, gt = pairs[i]
        if gt in current_dict:
            current_dict[gt] += 1
        else:
            current_dict[gt] = 1
    return invalid_pairs


def concordance_index(gt_lid: np.array, pred_lid: np.array, with_equal: bool = False) -> float:
    """
    Compute the number of pairs i, j such that i < j and gt_lid[i] > gt_lid[j] and pred_lid[i] < pred_lid[j]
    Runs in O((n + d) log d) using _fenwick tree where n is the number of elements in pred_lid and gt_lid
    and d is the maximum value of gt_lid or the ambient dimension

    Now compute the number of such pairs and divide it by the total number of pairs, n(n-1)/2
    finally, 1 minus the ratio is the concordance index
    """
    gt_lid = gt_lid.astype(np.int32)
    gt_lid = gt_lid - np.min(
        gt_lid
    )  # shifting the ground truth will not change the concordance index
    if with_equal:
        pred_lid = np.round(pred_lid).astype(np.int32)
    # create a list (pred_lid, gt_lid) and sort by pred_lid
    pairs = list(zip(pred_lid, gt_lid))
    pairs.sort()

    mx_gt_lid = int(np.max(gt_lid))
    arr = [0 for _ in range(mx_gt_lid + 1)]
    arr.insert(0, 0)
    bit = [0 for _ in range(mx_gt_lid + 2)]
    invalid_pairs_cnt = 0
    for _, gt in pairs:
        _fenwick_update(gt + 1, 1, arr, bit)
        invalid_pairs_cnt += _fenwick_get_range_sum(gt + 2, mx_gt_lid + 1, bit)
    # now take all the equal values into account
    if with_equal:
        invalid_pairs_cnt += _process_equals((pred_lid * 1e6).astype(np.int32), gt_lid)
        invalid_pairs_cnt += _process_equals(gt_lid, pred_lid)
    return 1 - float(invalid_pairs_cnt) / (len(pairs) * (len(pairs) - 1) / 2)
