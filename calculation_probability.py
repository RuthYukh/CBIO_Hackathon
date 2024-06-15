import numpy as np
import pandas as pd


class Calculation_probability:
    def __init__(self, input_df) -> None:
        self.df = input_df

    def count_aa_positions(self):
        # Define amino acids and positions
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        num_positions = 15  # Number of positions in each sequence

        # Mapping of amino acids to row indexes
        aa_to_index = {aa: index for index, aa in enumerate(amino_acids)}

        # Initialize the matrix with zeros
        count_matrix = np.zeros((len(amino_acids), num_positions))

        # Fill the matrix with counts
        for sequence in self.df['SEQUENCES']:
            for position, aa in enumerate(sequence):
                if aa in aa_to_index:  # Check if the amino acid is recognized
                    row_index = aa_to_index[aa]
                    count_matrix[row_index, position] += 1

        # Convert matrix to a DataFrame for easier visualization
        count_df = pd.DataFrame(count_matrix, index=list(amino_acids),
                                columns=[i for i in range(1, num_positions+1)])
        return count_df

    def calculate_probabilities(self):
        count_df = self.count_aa_positions()  # count without _
        prob_df = count_df
        for i in range(1, 16):  # go over cols
            len_col = count_df[i].sum()  # only real amino acids
            prob_df[i] = count_df[i] / len_col  # normalization
        return prob_df
