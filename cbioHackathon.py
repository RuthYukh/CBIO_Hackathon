from calculation_probability import Calculation_probability
from sklearn.model_selection import train_test_split
from logo import Logo_class
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.graph_objects as go


#################### PSSM FUNCTIONS ####################
def calculate_lh_ratio(neg_df, motif_df):
    # Calculate the likelihood ratio
    motif_df += 1e-5
    neg_df += 1e-5
    log_final = np.log2(motif_df / neg_df)
    return log_final


def convert_pssm_to_df(lh_ratio_all, lh_ratio_T, lh_ratio_S, lh_ratio_Y):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                   'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pssm_all = pd.DataFrame(lh_ratio_all, index=amino_acids,
                            columns=[f'{i}' for i in range(15)])
    pssm_T = pd.DataFrame(lh_ratio_T, index=amino_acids,
                          columns=[f'{i}' for i in range(15)])
    pssm_S = pd.DataFrame(lh_ratio_S, index=amino_acids,
                          columns=[f'{i}' for i in range(15)])
    pssm_Y = pd.DataFrame(lh_ratio_Y, index=amino_acids,
                          columns=[f'{i}' for i in range(15)])
    return pssm_all, pssm_T, pssm_S, pssm_Y


def option2_calculate_lh_ratio(probability_df_S_neg, probability_df_T_neg,
                               probability_df_Y_neg, probability_df_S_pos,
                               probability_df_T_pos, probability_df_Y_pos):
    motifT_np = probability_df_T_pos.to_numpy()
    motifS_np = probability_df_S_pos.to_numpy()
    motifY_np = probability_df_Y_pos.to_numpy()

    negT_np = probability_df_T_neg.to_numpy()
    negS_np = probability_df_S_neg.to_numpy()
    negY_np = probability_df_Y_neg.to_numpy()

    lh_ratio_T = calculate_lh_ratio(negT_np, motifT_np)
    lh_ratio_S = calculate_lh_ratio(negS_np, motifS_np)
    lh_ratio_Y = calculate_lh_ratio(negY_np, motifY_np)
    return lh_ratio_T, lh_ratio_S, lh_ratio_Y

#################### POSITIVE DATA FUNCTIONS ####################
def open_dataset():
    """
    open the dataset + mini preprocess
    filter only T/S/Y, humans
    :return: the df
    """
    tsv_file_path = './Phosphorylation_site_dataset.tsv'
    df = pd.read_csv(tsv_file_path, sep='\t')
    df = df[df['ORGANISM'] == 'human']
    df.rename(columns={'SITE_+/-7_AA': 'SEQUENCES', 'PROTEIN': 'PROTEIN_NAME'}, inplace=True)
    df['SEQUENCES'] = df['SEQUENCES'].str.upper()
    df['LABEL'] = 1  # because all the data has phosphorylation sites
    selected_cols = ["PROTEIN_NAME", "SEQUENCES", "LABEL"]
    df = df[selected_cols]
    df_T, df_S, df_Y = split_df_by_site(df)
    combined_df = pd.concat([df_T, df_S, df_Y], ignore_index=True)
    return combined_df


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


def calculate_prob_matrix(df):
    """
    Split DF to train, validation and test data
    Create probabilities matrix using X_train
    :return: X_train, X_test, Y_train, Y_test, probability_df_all
    """
    # split data to train, validation, test
    lables = df['LABEL']
    filterd_df = df.drop(columns=['LABEL'])  # remove the lables
    X_train, X_temp, Y_train, Y_temp = train_test_split(filterd_df, lables, test_size=0.3,
                                                        random_state=42)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp,
                                                        test_size=0.5,
                                                        random_state=42)
    calculation_all = Calculation_probability(X_train)
    probability_df_all = calculation_all.calculate_probabilities()
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, probability_df_all


#################### NEGATIVE DATA FUNCTIONS ####################

def extract_subsequences(sequence, protein_name):
    """
    Helper for create_negative_df
    Function to extract 15-mer sequences with S, T, or Y at
    the 8th position and keep the protein name
    """
    subsequences = []
    for i in range(len(sequence) - 14):
        subseq = sequence[i:i + 15]
        if subseq[7] in 'STY':
            subsequences.append((protein_name, subseq))
    return subsequences


def create_negative_df():
    """
    Create a dataset of negative sequences - that do not have phosphorylation sites
    :return: df of negative data
    """
    negative_df = pd.read_csv('uniprotkb_AND_model_organism_9606_2024_03_10.tsv', sep='\t', nrows=3000)
    negative_df = negative_df[negative_df['Post-translational modification'].isna()]

    # Apply the function to each sequence and collect the results
    new_sequences = []
    for _, row in negative_df.iterrows():
        new_sequences.extend(
            extract_subsequences(row['Sequence'], row['Protein names']))

    # Create a new DataFrame from the collected subsequences
    new_df = pd.DataFrame(new_sequences,
                          columns=['PROTEIN_NAME', 'SEQUENCES'])
    new_df['LABEL'] = 0
    return new_df


#################### VALIDATION FUNCTIONS ####################
def prepare_data_for_validation(X_all_pos, X_all_neg, Y_all_pos, Y_all_neg):
    # Concatenate positive and negative data and labels
    X_valid_combined = pd.concat([X_all_pos, X_all_neg],
                                 ignore_index=True)
    Y_valid_combined = pd.concat([Y_all_pos, Y_all_neg],
                                 ignore_index=True)
    validation_df = pd.concat([X_valid_combined, Y_valid_combined], axis=1)
    # Shuffle the data frame - neg and pos
    validation_df = shuffle(validation_df, random_state=42).reset_index(drop=True)
    return validation_df


def option2_prepare_data_for_validation(X_T_pos, X_T_neg, Y_T_pos, Y_T_neg,
                                        X_S_pos, X_S_neg, Y_S_pos, Y_S_neg,
                                        X_Y_pos, X_Y_neg, Y_Y_pos, Y_Y_neg):
    """
    Runs the prepare_data_for_validation for all 3 options T/S/Y
    """
    validation_df_t = prepare_data_for_validation(X_T_pos, X_T_neg, Y_T_pos, Y_T_neg)
    validation_df_s = prepare_data_for_validation(X_S_pos, X_S_neg, Y_S_pos, Y_S_neg)
    validation_df_y = prepare_data_for_validation(X_Y_pos, X_Y_neg, Y_Y_pos, Y_Y_neg)
    return validation_df_t, validation_df_s, validation_df_y


def get_scores(df, pssm):
    """
    For each sequences in the data frame it calculates the sequence score
    (according to the PSSM) and fill it in the score column
    """
    scores = []  # Initialize a list to store the scores

    # Iterate through each sequence in the DataFrame
    for seq in df['SEQUENCES']:
        score = 0
        # Calculate the score for the sequence
        for pos, aa in enumerate(seq):
            # Ensure 'aa' and 'adjusted_pos' are valid indices in PSSM
            if aa in pssm.index and str(pos) in pssm.columns:
                score += pssm.at[aa, str(pos)]
            # Alternatively, use 'pssm.loc[aa, str(adjusted_pos)]' if needed
        scores.append(score)

    # Add the scores to the DataFrame
    df['SCORE'] = scores
    return df


def plot_roc_curve_validation(df, title):
    """
    Plot the ROC curve for a binary classifier with a customizable title.
    Parameters:
    - y_true: True binary labels, gets the column['label']
    - y_scores: The scores we predicted, gets the column['score']
    - title: str, the amino acid we check for
    """
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(df['LABEL'], df['SCORE'])
    roc_auc = auc(fpr, tpr)

    # Identify the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
             label=f'Optimal Threshold: {np.round(optimal_threshold, 3)}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.savefig(f'{title}.png')

    return optimal_threshold


def create_confusion_matrix(df, title):
    y_true = df["LABEL"]
    y_pred = df["PREDICTION"]
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate additional metrics
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    P = TP + FN  # Number of actual positives
    N = TN + FP  # Number of actual negatives

    accuracy = (TP + TN) / (P + N)
    FPR = FP / N
    TPR = TP / P

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {title}')

    # Annotate the plot with additional metrics
    #plt.figtext(0.5, -0.1, f'Accuracy: {accuracy:.2f}\nFPR: {FPR:.2f}\nTPR: {TPR:.2f}',
    #        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    print(f'Accuracy for {title}: {accuracy:.2f}\nFPR: {FPR:.2f}\nTPR: {TPR:.2f}')
    plt.savefig(f'Confusion_matrix_{title}.png')


def feature_selection(lh_ratio, pos_names, amino_acids, title):
    ratio_all_colapsed = lh_ratio.flatten()
    col_names = list()
    for aa in amino_acids:
        for pos in pos_names:
            col_names.append(aa + '_' + pos)
    df_flatten_all = pd.DataFrame([ratio_all_colapsed], columns=col_names)
    cols_to_drop = [col for col in df_flatten_all.columns if "pos7" in col]
    df_flatten_all.drop(cols_to_drop, axis=1, inplace=True)
    df_flatten_all = df_flatten_all.T
    df_flatten_all_sort = df_flatten_all.sort_values(by=0)

    df_flatten_all_sort.reset_index(inplace=True)
    df_flatten_all_sort.columns = ['Column', 'Value']

    fig = go.Figure(
        data=go.Scatter(x=df_flatten_all_sort.iloc[:, 0], y=df_flatten_all_sort.iloc[:, 1], mode='lines+markers'))

    fig.update_layout(title=f'Log Ratio by Feature for {title}',
                      xaxis_title='Features',
                      yaxis_title='Log Ratio')
    fig.show()


#################### FEATURE SELECTION FUNCTIONS ####################
def pssm_feature(pssm, pos_val, neg_val):
    # Old option - make zeros middle values
    # pssm[(pssm <= pos_val) & (pssm >= neg_val)] = 0

    # New option - power the matrix
    pssm[pssm > 0] = np.power(pssm, 4)
    pssm[pssm < 0] = -np.power(pssm, 4)
    return pssm

#################### TEST FUNCTIONS ####################
def prepare_data_for_test(X_all_pos, X_all_neg, Y_all_pos, Y_all_neg):
    # Concatenate positive and negative data and labels
    X_valid_combined = pd.concat([X_all_pos, X_all_neg],
                                 ignore_index=True)
    Y_valid_combined = pd.concat([Y_all_pos, Y_all_neg],
                                 ignore_index=True)
    validation_df = pd.concat([X_valid_combined, Y_valid_combined], axis=1)
    validation_df['SCORE'] = None  # add empty col for scoring
    # Shuffle the data frame - neg and pos
    validation_df = shuffle(validation_df, random_state=42).reset_index(
        drop=True)
    return validation_df


def option2_prepare_data_for_test(X_T_pos, X_T_neg, Y_T_pos, Y_T_neg,
                                        X_S_pos, X_S_neg, Y_S_pos, Y_S_neg,
                                        X_Y_pos, X_Y_neg, Y_Y_pos, Y_Y_neg):
    """
    Runs the prepare_data_for_test for all 3 options T/S/Y
    """
    test_df_t = prepare_data_for_test(X_T_pos, X_T_neg, Y_T_pos, Y_T_neg)
    test_df_s = prepare_data_for_test(X_S_pos, X_S_neg, Y_S_pos, Y_S_neg)
    test_df_y = prepare_data_for_test(X_Y_pos, X_Y_neg, Y_Y_pos, Y_Y_neg)
    return test_df_t, test_df_s, test_df_y


def evaluate_phosphorylation_status_binary_df(df, pssm, threshold):
    """
    Function that calculates the score for all the sequences
    (using the get_scores function) and then evaluate phosphorylation status
     with binary predictions
    """
    # Calculate scores using the PSSM
    df_with_scores = get_scores(df, pssm)

    # Compare the scores against the threshold for binary predictions
    df_with_scores['PREDICTION'] = df_with_scores[
        'SCORE'].apply(lambda x: 1 if x > threshold else 0)

    return df_with_scores  # includes the predictions


#################### VISUALISATION FUNCTIONS ####################
def create_seq_logos(probability_df_all, probability_df_T, probability_df_S, probability_df_Y):
    logo_all = Logo_class(probability_df_all)
    logo_all.get_logo()  # saves the logo

    logo_T = Logo_class(probability_df_T)
    logo_T.get_logo()  # saves the logo

    logo_S = Logo_class(probability_df_S)
    logo_S.get_logo()  # saves the logo

    logo_Y = Logo_class(probability_df_Y)
    logo_Y.get_logo()  # saves the logo


def heatmap(lh_ratio, amino_acids, pos_names, title_name):
    pos_names = np.delete(pos_names, 7)
    lh_ratio = np.delete(lh_ratio, 7, axis=1)
    fig = go.Figure(data=go.Heatmap(
        z=lh_ratio,
        x=pos_names,
        y=amino_acids,
        colorscale='RdBu',
        zmid=0))
    fig.update_layout(title=title_name, xaxis_title='Position', yaxis_title='Amino Acid', width=450, height=600)

    fig.show()


#################### MAIN ####################
def main():
    ##### POSITIVE DATA #####
    pos_df = open_dataset()
    # Option 1 - use DF as is
    X_train_all_pos, X_valid_all_pos, X_test_all_pos, Y_train_all_pos, \
        Y_valid_all_pos, Y_test_all_pos, probability_df_all_pos = \
        calculate_prob_matrix(pos_df)

    # Option 2 - split DF to 3 DFs for T/S/Y
    df_T_pos, df_S_pos, df_Y_pos = split_df_by_site(pos_df)
    X_train_T_pos, X_valid_T_pos, X_test_T_pos, Y_train_T_pos, Y_valid_T_pos, \
        Y_test_T_pos, probability_df_T_pos = calculate_prob_matrix(df_T_pos)
    X_train_S_pos, X_valid_S_pos, X_test_S_pos, Y_train_S_pos, Y_valid_S_pos, \
        Y_test_S_pos, probability_df_S_pos = calculate_prob_matrix(df_S_pos)
    X_train_Y_pos, X_valid_Y_pos, X_test_Y_pos, Y_train_Y_pos, Y_valid_Y_pos, \
        Y_test_Y_pos, probability_df_Y_pos = calculate_prob_matrix(df_Y_pos)

    ##### NEGATIVE DATA #####
    neg_df = create_negative_df()
    # Option 1 - use DF as is
    X_train_all_neg, X_valid_all_neg, X_test_all_neg, Y_train_all_neg, \
        Y_valid_all_neg, Y_test_all_neg, probability_df_all_neg = calculate_prob_matrix(neg_df)

    # Option 2 - split DF to 3 DFs for T/S/Y
    df_T_neg, df_S_neg, df_Y_neg = split_df_by_site(neg_df)
    X_train_T_neg, X_valid_T_neg, X_test_T_neg, Y_train_T_neg, Y_valid_T_neg, \
        Y_test_T_neg, probability_df_T_neg = calculate_prob_matrix(df_T_neg)
    X_train_S_neg, X_valid_S_neg, X_test_S_neg, Y_train_S_neg, Y_valid_S_neg, \
        Y_test_S_neg, probability_df_S_neg = calculate_prob_matrix(df_S_neg)
    X_train_Y_neg, X_valid_Y_neg, X_test_Y_neg, Y_train_Y_neg, Y_valid_Y_neg, \
        Y_test_Y_neg, probability_df_Y_neg = calculate_prob_matrix(df_Y_neg)

    #################### TRAIN ####################
    ##### PSSM ON POSITIVE DATA#####
    # bg_path = "./bg_freq.txt" # TODO return?
    # bg_df = pd.read_csv(bg_path, index_col=0, dtype={'freq': np.float64})
    # bg_np = bg_df.to_numpy()

    # Calculate lh_ratio (the PSSM) + convert to DF
    # Option 1 - use all data together
    motif_np = probability_df_all_pos.to_numpy()
    neg_np = probability_df_all_neg.to_numpy()
    lh_ratio_all = calculate_lh_ratio(neg_np, motif_np)

    # Option 2 - separated T/S/Y
    lh_ratio_T, lh_ratio_S, lh_ratio_Y = option2_calculate_lh_ratio(probability_df_S_neg,
                                                                    probability_df_T_neg,
                                                                    probability_df_Y_neg,
                                                                    probability_df_S_pos,
                                                                    probability_df_T_pos,
                                                                    probability_df_Y_pos)

    # Convert the PSSM matrices from np.array to DF format for validation + test
    pssm_all, pssm_T, pssm_S, pssm_Y = convert_pssm_to_df(lh_ratio_all,
                                                          lh_ratio_T,
                                                          lh_ratio_S,
                                                          lh_ratio_Y)


    #################### VALIDATION ####################
    # Combine positive and negative data + shuffle + fill scores
    # Option 1 - use all data together
    validation_df_all = prepare_data_for_validation(X_valid_all_pos,
                                                    X_valid_all_neg,
                                                    Y_valid_all_pos,
                                                    Y_valid_all_neg)
    # Fill scores in matrix
    validation_df_all = get_scores(validation_df_all, pssm_all)

    # Option 2 - separated T/S/Y
    validation_df_T, validation_df_S, validation_df_Y = \
        option2_prepare_data_for_validation(X_valid_T_pos, X_valid_T_neg,
                                            Y_valid_T_pos, Y_valid_T_neg,
                                            X_valid_S_pos, X_valid_S_neg,
                                            Y_valid_S_pos, Y_valid_S_neg,
                                            X_valid_Y_pos, X_valid_Y_neg,
                                            Y_valid_Y_pos, Y_valid_Y_neg)
    # Fill scores in matrices
    validation_df_T = get_scores(validation_df_T, pssm_T)
    validation_df_S = get_scores(validation_df_S, pssm_S)
    validation_df_Y = get_scores(validation_df_Y, pssm_Y)

    # Find optimal threshold + ROC curve + table
    opt_thr_all = plot_roc_curve_validation(validation_df_all, "All")
    opt_thr_T = plot_roc_curve_validation(validation_df_T, "T")
    opt_thr_S = plot_roc_curve_validation(validation_df_S, "S")
    opt_thr_Y = plot_roc_curve_validation(validation_df_Y, "Y")

    # Feature selection + find new optimal thresholds
    positive_limit = 0.48
    negative_limit = -0.9
    pssm_feature_all = pssm_feature(pssm_all, positive_limit, negative_limit)
    pssm_feature_T = pssm_feature(pssm_T, positive_limit, negative_limit)
    pssm_feature_S = pssm_feature(pssm_S, positive_limit, negative_limit)
    pssm_feature_Y = pssm_feature(pssm_Y, positive_limit, negative_limit)

    # After feature selection, score and create ROC AGAIN!
    validation_df_all_fs = get_scores(validation_df_all, pssm_feature_all)
    validation_df_T_fs = get_scores(validation_df_T, pssm_feature_T)
    validation_df_S_fs = get_scores(validation_df_S, pssm_feature_S)
    validation_df_Y_fs = get_scores(validation_df_Y, pssm_feature_Y)
    new_opt_thr_all = plot_roc_curve_validation(validation_df_all_fs, "all - after feature selection")
    new_opt_thr_T = plot_roc_curve_validation(validation_df_T_fs, "T - after feature selection")
    new_opt_thr_S = plot_roc_curve_validation(validation_df_S_fs, "S - after feature selection")
    new_opt_thr_Y = plot_roc_curve_validation(validation_df_Y_fs, "Y - after feature selection")

    #################### TEST ####################
    # Option 1 - use all data together
    test_df_all = prepare_data_for_test(X_test_all_pos,
                                        X_test_all_neg,
                                        Y_test_all_pos,
                                        Y_test_all_neg)
    # Add scores of test
    test_df_all = evaluate_phosphorylation_status_binary_df(test_df_all,
                                                            pssm_all,
                                                            opt_thr_all)
    # Option 2 - separated T/S/Y
    test_df_T, test_df_S, test_df_Y = \
        option2_prepare_data_for_test(X_test_T_pos, X_test_T_neg,
                                      Y_test_T_pos, Y_test_T_neg,
                                      X_test_S_pos, X_test_S_neg,
                                      Y_test_S_pos, Y_test_S_neg,
                                      X_test_Y_pos, X_test_Y_neg,
                                      Y_test_Y_pos, Y_test_Y_neg)
    test_df_T = evaluate_phosphorylation_status_binary_df(test_df_T,
                                                          pssm_T,
                                                          opt_thr_T)
    test_df_S = evaluate_phosphorylation_status_binary_df(test_df_S,
                                                          pssm_S,
                                                          opt_thr_S)
    test_df_Y = evaluate_phosphorylation_status_binary_df(test_df_Y,
                                                          pssm_Y,
                                                          opt_thr_Y)
    # calculate confusion matrices for test data
    create_confusion_matrix(test_df_all, "all")
    create_confusion_matrix(test_df_T, "T")
    create_confusion_matrix(test_df_S, "S")
    create_confusion_matrix(test_df_Y, "Y")

    ##### VISUALISATION #####
    # Creates 4 sequence logos based on the positive train data
    create_seq_logos(probability_df_all_pos, probability_df_T_pos,
                     probability_df_S_pos, probability_df_Y_pos)

    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    pos_names = list("pos" + str(i) for i in range(0, 15))

    # Feature Selection
    feature_selection(lh_ratio_all, pos_names, amino_acids, "ALL")
    feature_selection(lh_ratio_T, pos_names, amino_acids, "T")
    feature_selection(lh_ratio_S, pos_names, amino_acids, "S")
    feature_selection(lh_ratio_Y, pos_names, amino_acids, "Y")

    heatmap(lh_ratio_all, amino_acids, pos_names, "All positive")
    heatmap(lh_ratio_T, amino_acids, pos_names, "T")
    heatmap(lh_ratio_S, amino_acids, pos_names, "S")
    heatmap(lh_ratio_Y, amino_acids, pos_names, "Y")


if __name__ == '__main__':
    main()