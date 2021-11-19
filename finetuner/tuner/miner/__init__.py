from itertools import combinations, groupby, permutations
from typing import List, Tuple


def get_session_pairs(
    sessions: List[int], match_types: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Generate all possible pairs for each session.

    :param sessions: A list of integers denoting session label
    :param match_types: A list of integers denoting match type - 0 for anchor, 1 for
        postive match and -1 for negative match)

    :return: three list of integers, first one holding integers of first element of
        pair, second of the second element of pair, and third one the label (0 or
        1) for the pair for each pair
    """
    ind_one, ind_two, labels_ret = [], [], []

    labels_index = [
        (sess, match_type, index)
        for index, (sess, match_type) in enumerate(zip(sessions, match_types))
    ]
    sorted_labels_with_index = sorted(labels_index, key=lambda x: x[0])

    for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
        for left, right in combinations(group, 2):  # (session_id, label, ind)
            if left[1] != -1 or right[1] != -1:
                ind_one.append(left[2])
                ind_two.append(right[2])
                labels_ret.append(0 if min(left[1], right[1]) == -1 else 1)

    return ind_one, ind_two, labels_ret


def get_session_triplets(
    sessions: List[int], match_types: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Generate all possible triplets for each session.

    :param sessions: A list of integers denoting session label
    :param match_types: A list of integers denoting match type - 0 for anchor, 1 for
        postive match and -1 for negative match)

    :return: three list of integers, holding the anchor index, positive index and
        negative index of each triplet, respectively
    """
    anchor_ind, pos_ind, neg_ind = [], [], []

    labels_index = [
        (sess, match_type, index)
        for index, (sess, match_type) in enumerate(zip(sessions, match_types))
    ]
    sorted_labels_with_index = sorted(labels_index, key=lambda x: x[0])

    for _, group in groupby(sorted_labels_with_index, lambda x: x[0]):
        anchor_pos_session_indices = []
        neg_session_indices = []

        for _, session_label, session_index in group:
            if session_label >= 0:
                anchor_pos_session_indices.append(session_index)
            else:
                neg_session_indices.append(session_index)

        for anchor, pos in permutations(anchor_pos_session_indices, 2):
            anchor_ind += [anchor] * len(neg_session_indices)
            pos_ind += [pos] * len(neg_session_indices)
            neg_ind += neg_session_indices

    return anchor_ind, pos_ind, neg_ind
