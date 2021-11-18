from typing import Tuple

import tensorflow as tf

from ..miner import get_session_pairs, get_session_triplets
from ..miner.base import BaseClassMiner, BaseSessionMiner


class SiameseMiner(BaseClassMiner[tf.Tensor]):
    def mine(
        self, labels: tf.Tensor, distances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate all possible pairs.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels)

        l1, l2 = tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)
        matches = tf.cast(l1 == l2, tf.uint8)
        diffs = 1 - matches
        matches = tf.experimental.numpy.triu(matches, 1)
        diffs = tf.experimental.numpy.triu(diffs)

        ind1_pos, ind2_pos = tf.unstack(tf.where(matches), axis=1)
        ind1_neg, ind2_neg = tf.unstack(tf.where(diffs), axis=1)

        ind1 = tf.concat([ind1_pos, ind1_neg], axis=0)
        ind2 = tf.concat([ind2_pos, ind2_neg], axis=0)

        target = tf.concat([tf.ones_like(ind1_pos), tf.zeros_like(ind1_neg)], axis=0)
        return ind1, ind2, target


class TripletMiner(BaseClassMiner[tf.Tensor]):
    def mine(
        self, labels: tf.Tensor, distances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate all possible triplets.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three 1D tensors, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """
        assert len(distances) == len(labels)

        l1, l2 = tf.expand_dims(labels, 1), tf.expand_dims(labels, 0)
        matches = tf.cast(l1 == l2, tf.uint8)
        diffs = matches ^ 1

        matches = tf.linalg.set_diag(matches, diagonal=tf.zeros_like(labels, tf.uint8))
        triplets = tf.expand_dims(matches, 2) * tf.expand_dims(diffs, 1)

        return tf.transpose(tf.where(triplets))


class SiameseSessionMiner(BaseSessionMiner[tf.Tensor]):
    def mine(
        self, labels: Tuple[tf.Tensor, tf.Tensor], distances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate all possible pairs for each session.

        :param labels: A tuple of 1D tensors, denotind the items' session and match
            type (0 for anchor, 1 for postive match and -1 for negative match),
            respectively
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three numpy arrays, first one holding integers of first element of
            pair, second of the second element of pair, and third one the label (0 or
            1) for the pair for each pair
        """
        assert len(distances) == len(labels[0]) == len(labels[1])

        sessions, match_types = [x.numpy().tolist() for x in labels]
        ind_one, ind_two, labels_ret = get_session_pairs(sessions, match_types)

        return (
            tf.constant(ind_one),
            tf.constant(ind_two),
            tf.constant(labels_ret),
        )


class TripletSessionMiner(BaseSessionMiner[tf.Tensor]):
    def mine(
        self, labels: Tuple[tf.Tensor, tf.Tensor], distances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate all possible triplets for each session.

        :param labels: A tuple of 1D tensors, denotind the items' session and match
            type (0 for anchor, 1 for postive match and -1 for negative match),
            respectively
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: three numpy arrays, holding the anchor index, positive index and
            negative index of each triplet, respectively
        """

        assert len(distances) == len(labels[0]) == len(labels[1])

        sessions, match_types = [x.numpy().tolist() for x in labels]
        anchor_ind, pos_ind, neg_ind = get_session_triplets(sessions, match_types)

        return (
            tf.constant(anchor_ind),
            tf.constant(pos_ind),
            tf.constant(neg_ind),
        )
