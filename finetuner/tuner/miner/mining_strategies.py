import numpy as np
import torch

from typing import Callable, Optional, Tuple, Union


class TorchStrategicMiningHelper:
    def __init__(self, pos_strategy: str, neg_strategy: str) -> None:
        """
        This helper implements easy-hard mining for tuples or triplets in
        siamese and triplet respectively. The following strategies are
        available.

        Pos. Strategy:
        - 'hard': Returns hardest positive (furthest) sample per anchor
        - 'semihard': Returns the hardest positive sample per anchor, such
          that it is closer than the selected negative
        - 'easy': Returns the easiest positive sample per anchor
        - 'all': Returns all positive samples

        Neg. Strategy:
        - 'hard': Returns hardest negative (closest) sample per anchor
        - 'semihard': Returns the hardest negative sample per anchor, such
          that it is further than the selected negative
        - 'easy': Returns the easiest negative sample per anchor
        - 'all': Returns all negative samples

        Not allowed:
        - pos. and neg. strategy cannot be set to 'semihard' simultaneously
        - When pos. or neg. strategy is set to 'semihard' the other cannot be
          set to 'all'

        :param pos_strategy: Strategy for selecting positive samples
        :param neg_strategy: Strategy for selecting negative samples
        """

        allowed_strategies = ['easy', 'semihard', 'hard', 'all']
        if (
            pos_strategy not in allowed_strategies
            or neg_strategy not in allowed_strategies
        ):
            raise ValueError(
                f'The strategy has to be one of all, easy, semihard, and hard, but '
                'was: {strategy}'
            )
        elif pos_strategy == 'semihard' and neg_strategy == 'semihard':
            raise ValueError(
                'Positive and negative strategy cannot both be set to semihard.'
            )
        elif (pos_strategy == 'all' and neg_strategy == 'semihard') or (
            pos_strategy == 'semihard' and neg_strategy == 'all'
        ):
            raise ValueError(
                'When one strategy is set to semihard, the other cannot be set to hard.'
            )
        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_per_row_min(
        self, dist_mat: torch.Tensor, semihard_tsh: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        """Given a matrix, this function gets the min value of each valid row and
        their respective column indices

        :param dist_mat: Symmetric tensor with pair-wise embedding distances
        :param semihard_tsh: Maximum upper bound on the distance that is selected
          as minumum. This is needed for semihard mining.

        :return: Tuple with two tensors of per-row min values and respective indices
        :return non_inf_rows: Rows where the extracted max is larger than zero
          and has not been masked through thresholding
        """

        zero_element_mask = torch.logical_not(dist_mat > 0)
        if len(dist_mat) == 0:
            return (torch.empty(()), torch.empty((), dtype=torch.bool)), torch.empty(
                (), dtype=torch.bool
            )

        # Set zeros to max value, so they are not extracted as row minimum
        row_max = torch.max(dist_mat, dim=1, keepdim=True)[0]
        dist_mat += (row_max + 1) * zero_element_mask

        if semihard_tsh is not None:
            dist_mat[dist_mat <= semihard_tsh] = float('inf')

        # Get row mask for rows where thresholding caused min to be infinity
        non_inf_rows = torch.all(dist_mat == float('inf'), dim=1)
        return torch.min(dist_mat, dim=1, keepdim=True), non_inf_rows

    def _get_per_row_max(
        self, dist_mat: torch.Tensor, semihard_tsh: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Given a matrix, this function gets the max value of each non-zero row and
        their respective column indices

        :param dist_mat: Symmetric tensor with pair-wise embedding distances
        :param semihard_tsh: Minimum lower bound on the distance that is selected
          as maximum. This is needed for semihard mining.

        :return: Tuple with two tensors of per-row max values and respective indices
        :return non_zero_rows: Rows where the extracted max is larger than zero
          and has not been masked through thresholding
        """

        if len(dist_mat) == 0:
            return (torch.empty(()), torch.empty((), dtype=torch.bool)), torch.empty(
                (), dtype=torch.bool
            )
        # Mask for semihard case
        if semihard_tsh is not None:
            dist_mat[dist_mat >= semihard_tsh] = 0

        # Get row mask for rows where thresholding caused max to be zero
        all_zero_rows = torch.all(dist_mat == 0, dim=1)
        return torch.max(dist_mat, dim=1, keepdim=True), all_zero_rows

    def _update_dist_mat(
        self, dist_mat: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Given a distance matrix and indices for row-wise min or max values,
         this func removes all but the extreme values from the matrix

        :param dist_mat: Pair-wise distance matrix
        :param indices: Row-wise indices of min or max values

        :return: The distance matrix, where only the min or max values
          remain in each row
        """
        keep_mask = torch.zeros_like(dist_mat)
        keep_mask[range(keep_mask.shape[0]), indices.squeeze()] = 1
        # Mask and return the distance matrix
        return dist_mat * keep_mask

    def _get_mine_func(self, strategy: str) -> Callable:
        """Given a strategy, this function gets the correct extractor for
        min or max distance values

        :param strategy: Name of the mining strategy

        :return: Function that either gets the row-wise min or max
        """
        if strategy in ['hard', 'semihard']:
            return self._get_per_row_max
        else:
            return self._get_per_row_min

    def _update_pos_mat(self, match_mat, dist_mat, pos_strategy, semihard_tsh=None):
        """Function that wraps the update of the postitive pair distancs to realize
        mining strategy.

        :param match_mat: Matrix that indicates the values in distance matrix, that
          belong to positive pairs
        :param dist_mat: Matrix with pair-wise encoding distances
        :param strategy: Mining strategy for positive samples
        :param semihard_tsh: Row-wise threshold values incorporated during semihard
          mining

        :return: Updated distance matrix so it can be used to realize mining strategy,
          and the row-wise min or max distances, depending on mining strategy
        """
        # Get all positive distances d(a, p)
        d_a_p = match_mat * dist_mat
        mine_func = self._get_mine_func(pos_strategy)
        (pos_dists, min_max_indices), invalid_row_mask = mine_func(d_a_p, semihard_tsh)
        # Remove rows where semihard thresholding has created unusable values
        match_mat[invalid_row_mask] = 0
        return self._update_dist_mat(match_mat, min_max_indices), pos_dists

    def _update_neg_mat(self, diff_mat, dist_mat, neg_strategy, semihard_tsh=None):
        # Get all negative distances d(a, n)
        d_a_n = diff_mat * dist_mat

        # Neg. needs to be handled in opposite fashion than pos. strategy
        neg_strategy = 'easy' if neg_strategy in ('hard', 'semihard') else 'hard'
        mine_func = self._get_mine_func(neg_strategy)
        (neg_dists, min_max_indices), invalid_row_mask = mine_func(d_a_n, semihard_tsh)
        # Remove rows where semihard thresholding has created unusable values
        diff_mat[invalid_row_mask] = 0
        return self._update_dist_mat(diff_mat, min_max_indices), neg_dists

    def _handle_input_tensors(self, tensor: Union[torch.Tensor, np.ndarray]):
        """Helper function to parse input tensors for strategy application"""
        if not isinstance(tensor, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f'Application of mining strategies only works '
                f'on ndarrays or pytorch tensors, but the passed tensor was of type: {type(tensor)}'
            )
        if isinstance(tensor, np.ndarray):
            tensor = torch.Tensor(tensor)
        return tensor.to(self.device)

    def apply_strategy(
        self,
        match_mat: Union[torch.Tensor, np.ndarray],
        diff_mat: Union[torch.Tensor, np.ndarray],
        dist_mat: Union[torch.Tensor, np.ndarray],
        to_numpy: bool = False,
    ):
        """Wraps the application of mining strategies to update the matrices
        with positive and negative matches depending, using the distance
        matrix for filtering

        :param match_mat: Matrix indicating matches between positive embeddings
        :param diff_mat: Matrix indicating matches between negative embeddings
        :param dist_mat: Matrix with pair-wise embedding distances
        :param to_numpy: Boolean switch determining whether to return numpy or
          torch tensors

        :return match_mat: Updated matrix of positve matches after applying
          strategy
        :return diff_mat: Updated matrix of negative matches after applying
          strategy
        """
        match_mat = self._handle_input_tensors(match_mat)
        diff_mat = self._handle_input_tensors(diff_mat)
        dist_mat = self._handle_input_tensors(dist_mat)

        if self.pos_strategy == 'semihard' and self.neg_strategy != 'all':

            diff_mat, neg_dists = self._update_neg_mat(
                diff_mat, dist_mat, self.neg_strategy
            )
            match_mat, _ = self._update_pos_mat(
                match_mat, dist_mat, self.pos_strategy, neg_dists
            )
        elif self.pos_strategy != 'all' and self.neg_strategy == 'semihard':

            match_mat, pos_dists = self._update_pos_mat(
                match_mat, dist_mat, self.pos_strategy
            )
            diff_mat, _ = self._update_neg_mat(
                diff_mat, dist_mat, self.neg_strategy, pos_dists
            )
        else:
            if self.pos_strategy != 'all':
                match_mat, _ = self._update_pos_mat(
                    match_mat, dist_mat, self.pos_strategy
                )
            if self.neg_strategy != 'all':
                diff_mat, _ = self._update_neg_mat(
                    diff_mat, dist_mat, self.neg_strategy
                )
        if to_numpy:
            return (
                # This has to int32 for paddle to
                match_mat.cpu().numpy().astype('int32'),
                diff_mat.cpu().detach().numpy().astype('int32'),
            )
        else:
            return match_mat.to(torch.int32), diff_mat.to(torch.int32)
