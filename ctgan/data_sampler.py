"""DataSampler module."""

import numpy as np


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency, protected_columns=None, candidates=5):
        self._data_length = len(data)
        # New attribute to store protected columns names (DS)
        self.protected_columns = protected_columns
        # New attribute to store the number of candidate solutions to generate for fair sampling (DS)
        # Default of 5 is the same as in the fairdo package (DS)
        self.candidates = candidates

        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'

        n_discrete_columns = sum([
            1 for column_info in output_info if is_discrete_column(column_info)
        ])

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim for column_info in output_info if is_discrete_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    # Uniform selection to Fair selection (DS)
    def _random_choice_prob_index(self, discrete_column_id):
        # If discrete_column_id is a 1D numpy array and all values are the same, convert it to a single integer. 
        if isinstance(discrete_column_id, np.ndarray) and discrete_column_id.ndim == 1:
            if not np.all(discrete_column_id == discrete_column_id[0]):
                # Compute the mode: the unique value that appears most frequently
                unique_vals, counts = np.unique(discrete_column_id, return_counts=True)
                discrete_column_id = int(unique_vals[np.argmax(counts)])
            else:
                discrete_column_id = int(discrete_column_id[0])
        
        probs = self._discrete_column_category_prob[discrete_column_id]
        # Get the number of samples (batch_size) and the number of possible categories (num_categories).
        batch_size, num_categories = probs.shape
        
        # If this column is not marked as protected, use the standard sampling method
        if (self.protected_columns is None) or (discrete_column_id not in self.protected_columns):
            r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
            return (probs.cumsum(axis=1) > r).argmax(axis=1)
        # If the column is protected, we want to find fair candidates
        else: 
            candidate_list = []
            fairness_values = []
            # Loop to generate a set number of candidate solutions
            for i in range(self.candidates):
                # Generate a candidate selection for the entire batch using the cumulative-sum sampling method
                r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
                candidate = (probs.cumsum(axis=1) > r).argmax(axis=1)
                candidate_list.append(candidate)

                # Computes the observed frequency of each category for this candidate
                observed_treatment = np.bincount(candidate, minlength=num_categories)
                # Create the ideal (fair) frequency distribution
                ideal_treatment = np.full(num_categories, batch_size / num_categories)
                # Calculate the disparity (fairness score) as the sum of absolute differences between observed and ideal frequencies
                # A lower disparity means the candidate is closer to a fair, balanced distribution
                disparity = np.sum(np.abs(observed_treatment - ideal_treatment))
                fairness_values.append(disparity)
                
            # Convert the fairness scores to a numpy array for easier processing.
            fairness_values = np.array(fairness_values)
            # Find the candidate(s) with the minimum disparity.
            candidate_indices = np.where(fairness_values == np.min(fairness_values))[0]
            # Choose the first candidate among those with the minimum disparity.
            best_candidate_index = candidate_indices[0]
            # Return the candidate with the lowest disparity 
            return candidate_list[best_candidate_index]

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        cond[np.arange(batch), col_idxs] = 1

        return cond

    def sample_data(self, data, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Args:
            data:
                The training data.

        Returns:
            n:
                n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
