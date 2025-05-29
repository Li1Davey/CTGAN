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
        # Add counter attributes for tracking uniform and fairness branch visits (DS)
        self.uniform_branch_counter = 0
        self.fair_branch_counter = 0

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
                
    def _random_choice_prob_index(self, discrete_column_id):
        self.uniform_branch_counter += 1
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    # Uniform selection to Fair selection (DS)
    def _fair_choice_prob_index(self, discrete_column_id, batch_size):
        # If discrete_column_id is a 1D numpy array, it is converted to a scalar by taking the first element.
        if isinstance(discrete_column_id, np.ndarray) and discrete_column_id.ndim == 1:
            discrete_column_id = int(discrete_column_id[0])
        
        # Retrieve the probability distribution for the chosen discrete column.
        probs = self._discrete_column_category_prob[discrete_column_id]
        
        # If probs is one-dimensional (only one row), replicate it to match the current batch size.
        if probs.ndim == 1:
            # np.tile replicates the array 'batch_size' times along the first dimension
            probs = np.tile(probs, (batch_size, 1))
        
        # Now get the actual shape of the probability array.
        batch_size_actual, num_categories = probs.shape

        # If the discrete column is not marked as protected, use the standard cumulative-sum sampling.
        if (self.protected_columns is None) or (discrete_column_id not in self.protected_columns):
            self.uniform_branch_counter += 1
            # For each sample, generate a random number and select the category where the cumulative probability exceeds it.
            r = np.expand_dims(np.random.rand(batch_size_actual), axis=1)
            return (probs.cumsum(axis=1) > r).argmax(axis=1)
        else:
            
            # Icrement the fairness counter to see how many times fairness is incremented
            self.fair_branch_counter += 1
            
            # For protected attributes, we generate several candidate selections and choose the fairest one.
            candidate_list = []
            fairness_values = []

            # Loop to generate a number of candidate solutions (the number is set in self.candidates).
            for i in range(self.candidates):
                # Generate a candidate: for each sample, sample a category using the cumulative-sum method.
                r = np.expand_dims(np.random.rand(batch_size_actual), axis=1)
                candidate = (probs.cumsum(axis=1) > r).argmax(axis=1)
                candidate_list.append(candidate)

                # Observed treatment: Count how many times each category is chosen in this candidate.
                observed_treatment = np.bincount(candidate, minlength=num_categories)
                # Ideal treatment: Define the ideal (uniform) distribution: each category should appear equally.
                ideal_treatment = np.full(num_categories, batch_size_actual / num_categories)
                # Compute the disparity as the sum of absolute differences between observed and ideal counts.
                disparity = np.sum(np.abs(observed_treatment - ideal_treatment))
                fairness_values.append(disparity)


                # Print candidate and its disparity
                print(f"Candidate {i + 1}:")
                print(f"Observed Treatment: {observed_treatment}")
                print(f"Disparity Score: {disparity:.2f}")
                print("-" * 40)
            # Convert the list of fairness values into a numpy array.
            fairness_values = np.array(fairness_values)
            # Find the indices of candidate(s) that have the smallest disparity (i.e., the fairest candidates).
            candidate_indices = np.where(fairness_values == fairness_values.min())[0]
            # If multiple candidates tie, choose the first one.
            best_candidate_index = candidate_indices[0]
            # Return the candidate selection corresponding to the best candidate.
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
        
        # Force the discrete column ID to be uniform
        if not np.all(discrete_column_id == discrete_column_id[0]):
            discrete_column_id = np.full(batch, discrete_column_id[0])

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        
        # Decide which sampling method to use (DS)
        if (self.protected_columns is None) or (len(self.protected_columns) == 0):
            # Standard uniform sampling (as defined in _random_choice_prob_index).
            if not hasattr(self, "_printed_sampling_type"):
                print("Uniform Sampling")
                self._printed_sampling_type = True
            category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        else:
            # Fairness-aware sampling for protected attributes.
            if not hasattr(self, "_printed_sampling_type"):
                print("Fairness-aware Sampling, # of candidates: ", self.candidates)
                self._printed_sampling_type = True
            category_id_in_col = self._fair_choice_prob_index(discrete_column_id, batch)
        
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
