from calculation_probability import Calculation_probability
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logomaker
import matplotlib.pyplot as plt


##### BG + PSSM #####
def calculate_lh_ratio(bg_df, motif_df):
    # Calculate the likelihood ratio
    motif_df += 1e-5
    log_final = np.log2(motif_df / bg_df)
    return log_final

##### POSITIVE DATA FUNCTIONS  #####
def open_dataset():
    """
    open the dataset + mini preprocess
    :return: the df
    """
    tsv_file_path = './Phosphorylation_site_dataset.tsv'
    df = pd.read_csv(tsv_file_path, sep='\t')
    df = df[df['ORGANISM'] == 'human']
    df.rename(columns={'SITE_+/-7_AA': 'SEQUENCES'}, inplace=True)
    df['SEQUENCES'] = df['SEQUENCES'].str.upper()
    df['LABEL'] = 1  # because all the data has phosphorylation sites
    return df


def split_df_by_site(df):
    """
    Split the df to 3 dfs - only with T, only with Y, and S
    :param df:
    :return: 3 DFs split by T/ Y/ S
    """
    df_T = df[df['SEQUENCES'].str[7] == 'T']
    df_S = df[df['SEQUENCES'].str[7] == 'S']
    df_Y = df[df['SEQUENCES'].str[7] == 'Y']
    return df_T, df_S, df_Y


def split_data_train_test(df):
    lables = df['LABEL']
    filterd_df = df.drop(columns=['LABEL'])  # remove the lables
    X_train, X_test, Y_train, Y_test = train_test_split(filterd_df, lables,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, Y_train, Y_test


def calculate_prob_matrix(df):
    """
    Split DF + create probabilities matrix using X_train
    :return: X_train, X_test, Y_train, Y_test, probability_df_all
    """
    X_train, X_test, Y_train, Y_test = split_data_train_test(df)
    calculation_all = Calculation_probability(X_train)
    probability_df_all = calculation_all.calculate_probabilities()
    return X_train, X_test, Y_train, Y_test, probability_df_all


def create_logos(X_train_all, X_train_T, X_train_S, X_train_Y):
    """
    Save relevant data for sequence logos
    :return:
    """
    X_train_all['SEQUENCES'].to_csv("./seq_for_logo_all.txt", index=False,
                                    header=False, sep='\n')
    X_train_T['SEQUENCES'].to_csv("./seq_for_logo_T.txt", index=False,
                                  header=False, sep='\n')
    X_train_S['SEQUENCES'].to_csv("./seq_for_logo_S.txt", index=False,
                                  header=False, sep='\n')
    X_train_Y['SEQUENCES'].to_csv("./seq_for_logo_Y.txt", index=False,
                                  header=False, sep='\n')

##### NEGATIVE DATA #####

def extract_subsequences(sequence, protein_name):
    """
    Function to extract 15-mer sequences with S, T, or Y at
    the 8th position and keep the protein name
    """
    subsequences = []
    for i in range(len(sequence) - 14):
        subseq = sequence[i:i + 15]
        if subseq[7] in 'STY':
            subsequences.append((protein_name, subseq))
    return subsequences

def heatmap(lh_ratio, amino_acids, pos_names):
    plt.imshow(lh_ratio, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.yticks(ticks=np.arange(20), labels=amino_acids)
    plt.xticks(ticks=np.arange(14), labels=pos_names, rotation=45)
    plt.show()

def create_negative_df():
    """
    Create a dataset of negative sequences - that do not have phosphorylation sites
    :return: df of negative data
    """
    negative_df = pd.read_csv('uniprotkb_AND_model_organism_9606_2024_03_10.tsv', sep='\t', nrows=100)
    negative_df = negative_df[negative_df['Post-translational modification'].isna()]

    # Apply the function to each sequence and collect the results
    new_sequences = []
    for _, row in negative_df.iterrows():
        new_sequences.extend(
            extract_subsequences(row['Sequence'], row['Protein names']))

    # Create a new DataFrame from the collected subsequences
    new_df = pd.DataFrame(new_sequences,
                          columns=['protein_name', 'SEQUENCES'])
    new_df['LABEL'] = 0
    return new_df

if __name__ == '__main__':
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    pos_names = list("pos" + str(i) for i in range(1, 15))

    ##### POSITIVE DATA #####
    df = open_dataset()
    # TODO: filter only cols?
    # Option 1 - use DF as is
    X_train_all, X_test_all, Y_train_all, Y_test_all, probability_df_all = calculate_prob_matrix(df)

    # Option 2 - split DF to 3 DFs for T/S/Y
    df_T, df_S, df_Y = split_df_by_site(df)
    X_train_T, X_test_T, Y_train_T, Y_test_T, probability_df_T = calculate_prob_matrix(df_T)
    X_train_S, X_test_S, Y_train_S, Y_test_S, probability_df_S = calculate_prob_matrix(df_S)
    X_train_Y, X_test_Y, Y_train_Y, Y_test_Y, probability_df_Y = calculate_prob_matrix(df_Y)

    ##### VISUALISATION #####
    # TODO
    create_logos(X_train_all, X_train_T, X_train_S, X_train_Y)

    ##### NEGATIVE DATA #####
    negative_df = create_negative_df()
    # combine negative data and X test data, to create full test data
    # TODO

    ##### BG + PSSM #####
    bg_path = "./bg_freq.txt"
    bg_df = pd.read_csv(bg_path, index_col=0, dtype={'freq': np.float64})
    bg_np = bg_df.to_numpy()

    # Option 1
    motif_np = probability_df_all.to_numpy()
    lh_ratio_all = calculate_lh_ratio(bg_np, motif_np)

    # Option 2 - separated T/S/Y
    motifT_np = probability_df_T.to_numpy()
    motifS_np = probability_df_S.to_numpy()
    motifY_np = probability_df_Y.to_numpy()
    lh_ratio_T = calculate_lh_ratio(bg_np, motifT_np)
    lh_ratio_S = calculate_lh_ratio(bg_np, motifS_np)
    lh_ratio_Y = calculate_lh_ratio(bg_np, motifY_np)

    ##### TEST ALGO #####
    # TODO