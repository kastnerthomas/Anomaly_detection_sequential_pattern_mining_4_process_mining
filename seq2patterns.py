import pandas as pd
import numpy as np
import ast
import operator # pour compter le nombre d'observations uniques distinctes

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from pyod.utils.utility import precision_n_scores
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target

from collections import defaultdict
from itertools import combinations
from itertools import product

import random
random.seed(1)

from prefixspan import PrefixSpan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

import warnings

class Seq2patterns(): 
    def __init__(self, nb_of_frequent_patterns, min_len_of_frequent_pattern, kmeans_is_closed, n_clust, algo_clustering, aggreg_method):
        """
        Initialize the Seq2patterns object with specified parameters.

        Parameters:
        - nb_of_frequent_patterns: Number of top frequent patterns to consider
        - min_len_of_frequent_pattern: Minimum length of frequent patterns to be considered
        - kmeans_is_closed: Boolean flag indicating whether PrefixSpan uses closed itemsets
        - n_clust: Number of clusters for KMeans clustering
        """
        self.nb_of_frequent_patterns = nb_of_frequent_patterns
        self.min_len_of_frequent_pattern = min_len_of_frequent_pattern
        self.kmeans_is_closed = kmeans_is_closed
        self.n_clust = n_clust
        self.algo_clustering=algo_clustering
        self.aggreg_method=aggreg_method
        
        self.freq_patterns = None
        self.scaler_kmeans_dict = None

    def get_params(self, deep=True): 
        
        return {
        "nb_of_frequent_patterns":self.nb_of_frequent_patterns,
        "min_len_of_frequent_pattern":self.min_len_of_frequent_pattern,
        "kmeans_is_closed":self.kmeans_is_closed,
        "n_clust":self.n_clust,
        "algo_clustering":self.algo_clustering,
        "aggreg_method":self.aggreg_method,
        "freq_patterns":self.freq_patterns,
        "scaler_kmeans_dict":self.scaler_kmeans_dict
        }

    def set_params(self, **parameters): 
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 
    

    def __top_k_frequent_patterns(self, data, k, min_pattern_length,is_closed):
        """
        Finds the top-k frequent patterns in the input data using PrefixSpan algorithm.

        Parameters:
        - data: Input data for pattern extraction
        - k: Number of top frequent patterns to retrieve
        - min_pattern_length: Minimum length of patterns to be considered
        - is_closed: Boolean flag indicating whether closed itemsets are considered
        
        Returns:
        - List of selected top-k frequent patterns
        """
        
        # Create PrefixSpan object
        
        ps = PrefixSpan(data['Obs_seq'].tolist())
        ps.minlen = min_pattern_length
        # Find top-k frequent patterns
        distinct_values = len(set(data['Obs_seq'].sum()))  # Calculate the number of distinct values
        len_data=len(data)
        
        if k<1:
            top_k_patterns=ps.frequent(int(k*len_data),closed=is_closed)
        else:
            top_k_patterns = ps.topk(k,closed=is_closed)

    #         # closed itemsets provide a non-redundant representation of all frequent patterns
    #         # Filter patterns with at least min_pattern_length items and select the top-k
    #         selected_patterns = [pattern for pattern in top_k_patterns if len(pattern[1]) >= min_pattern_length][:k]

    #         if len(selected_patterns)<k:
    #             warnings.warn("LONG RUNNING TIME BECAUSE CALCULATE ALL FREQUENT PATTERNS", RuntimeWarning)
    #             top_k_patterns = ps.topk(distinct_values**distinct_values,closed=is_closed)  # Adjust the parameter to potentially optimize

    #             selected_patterns = [pattern for pattern in top_k_patterns if len(pattern[1]) >= min_pattern_length][:k]   

    #         if len(selected_patterns)<k:
    #             raise ValueError("Pas assez de patterns trouvés--> Number of frequent patterns found: {} and the value of k is '{}'.".format(len(selected_patterns), k))

        return top_k_patterns#selected_patterns


    # Functions for finding combination indices, checking for strictly increasing lists, and matching indexes to intervals
    def __find_combination_indices(self, sequence, target_combination):
        """
        Finds indices of target_combination within the sequence.

        Parameters:
        - sequence: Input sequence to search for combinations
        - target_combination: Target combination to search for within the sequence
        
        Returns:
        - List of indices representing occurrences of the target_combination in the sequence
        """
        
        indices_dict = defaultdict(list)
        for index, value in enumerate(sequence):
            if value in target_combination:
                indices_dict[value].append(index)

        possible_indices = list(product(*(indices_dict[key] for key in target_combination)))
        combinations_indices = [indices for indices in possible_indices if list(sequence[i] for i in indices) == target_combination]
        
        return combinations_indices   

    def __is_strictly_increasing(self, lst):
        """
        Checks if a list is strictly increasing.

        Parameters:
        - lst: Input list to check for strict increasing order

        Returns:
        - Boolean value indicating if the list elements are in strictly increasing order
        """
        
        return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))

    def __find_subsequences(self, sequence, pattern):
        """
        Finds subsequences in the input sequence that match the given pattern.

        Parameters:
        - sequence: Input sequence to search for subsequences
        - pattern: Target pattern to search for within the sequence

        Returns:
        - DataFrame containing subsequences matching the given pattern
        """
        
        combination_indices = self.__find_combination_indices(sequence, pattern)
        data = {'Combination': [pattern] * len(combination_indices), 
                'Index': [list(tup) for tup in combination_indices]}
        df = pd.DataFrame(data)
        df_filtered = df[df['Index'].apply(lambda x: self.__is_strictly_increasing(x))]
        
        return df_filtered
    
#     def __find_subsequences(self, sequence, pattern):
#         """
#         Finds subsequences in the input sequence that match the given pattern.

#         Parameters:
#         - sequence: Input sequence to search for subsequences
#         - pattern: Target pattern to search for within the sequence

#         Returns:
#         - DataFrame containing subsequences matching the given pattern
#         """
#         combination_indices = self.__find_combination_indices(sequence, pattern)
#         data = {'Combination': [pattern] * len(combination_indices), 
#                 'Index': [list(tup) for tup in combination_indices]}
#         df = pd.DataFrame(data)
#         df_filtered = df[df['Index'].apply(lambda x: all(x[i] < x[i + 1] for i in range(len(x) - 1)))]

#         return df_filtered

    def __calculate_sum_between_indexes(self, indexes, intervals_list):
        """
        Calculates the sum between indexes in a given intervals list.

        Parameters:
        - indexes: List of indexes to calculate the sum between
        - intervals_list: List of intervals to compute the sum

        Returns:
        - List of sums calculated between the specified indexes in the intervals list
        """
            
        sum_list = []
        for i in range(len(indexes) - 1):
            start_index, end_index = indexes[i], indexes[i + 1]
            sublist = intervals_list[start_index:end_index]
            sum_list.append(sum(sublist))
            
        return sum_list
#     def __calculate_sum_between_indexes(self, indexes, intervals_list):
#         """
#         Calculates the sum between indexes in a given intervals list.

#         Parameters:
#         - indexes: List of indexes to calculate the sum between
#         - intervals_list: List of intervals to compute the sum

#         Returns:
#         - List of sums calculated between the specified indexes in the intervals list
#         """
#         intervals = np.array(intervals_list)
#         return [intervals[start:end].sum() for start, end in zip(indexes[:-1], indexes[1:])]


    def __match_indexes_to_intervals(self, sequence, df_pattern, intervals_list):
        """
        Matches indexes to intervals in the sequence and DataFrame pattern.

        Parameters:
        - sequence: Input sequence associated with the DataFrame pattern
        - df_pattern: DataFrame containing pattern information
        - intervals_list: List of intervals related to the sequence

        Returns:
        - DataFrame with intervals matched to corresponding indexes
        """
        df_pattern['SumBetweenIndexes'] = df_pattern['Index'].apply(lambda x: self.__calculate_sum_between_indexes(x, intervals_list))
        return df_pattern

    def __find_pattern_occurrences(self, data_sequence, data_intervals_list, frequent_patterns):
        """
        Finds occurrences of frequent patterns within the data sequences along with their intervals.

        This method locates occurrences of given frequent patterns within the sequences provided, considering associated intervals.

        Parameters:
        - data_sequence: Sequence data for pattern occurrence analysis
        - data_intervals_list: List of intervals corresponding to the sequence data
        - frequent_patterns: List of frequent patterns to search for in the sequences

        Returns:
        - DataFrame containing occurrences of frequent patterns with their respective interval information
        """
        # Iterate through each sequence and its associated intervals to find occurrences of frequent patterns
        # Match identified patterns with intervals and compile the results into a DataFrame
        # Group and aggregate interval information for each pattern occurrence

        # Example logic:
        # 1. Find subsequences in the sequence data that match the frequent patterns
        # 2. Match the identified subsequences with intervals from the intervals list
        # 3. Aggregate and structure the information into a DataFrame with pattern occurrences and their intervals

        # Return the DataFrame containing occurrences of frequent patterns with interval
        df_seq_patterns_concatenated = pd.DataFrame(columns=["Combination", "MeanSumIntervals"])
        for seq_index, (sequence,intervals_list) in enumerate(zip(data_sequence,data_intervals_list)):

            df_seq_patterns = pd.DataFrame(columns=["Combination", "Index", "SumBetweenIndexes"])

            for freq, pattern in frequent_patterns:
                df_pattern = self.__find_subsequences(sequence, pattern)
                if not df_pattern.empty:
                    df_pattern = self.__match_indexes_to_intervals(sequence, df_pattern, intervals_list)
                    df_seq_patterns = pd.concat([df_seq_patterns, df_pattern])

            df_seq_patterns['Combination'] = df_seq_patterns['Combination'].astype(str)
            # df_seq_patterns = df_seq_patterns.groupby('Combination', group_keys=True)['SumBetweenIndexes'].apply(lambda x: pd.DataFrame(x.tolist()).mean().tolist()).reset_index(name='SumBetweenIndexes')
            if self.aggreg_method=="min":
                df_seq_patterns = df_seq_patterns.groupby('Combination', group_keys=True)['SumBetweenIndexes'].apply(lambda x: pd.DataFrame(x.tolist()).min().tolist()).reset_index(name='SumBetweenIndexes')
            elif self.aggreg_method=="max":
                df_seq_patterns = df_seq_patterns.groupby('Combination', group_keys=True)['SumBetweenIndexes'].apply(lambda x: pd.DataFrame(x.tolist()).max().tolist()).reset_index(name='SumBetweenIndexes')
            elif self.aggreg_method=="mean":
                df_seq_patterns = df_seq_patterns.groupby('Combination', group_keys=True)['SumBetweenIndexes'].apply(lambda x: pd.DataFrame(x.tolist()).mean().tolist()).reset_index(name='SumBetweenIndexes')
            # elif self.aggreg_method=="median":
            else:
                print("error in aggregation of intervals")
            df_seq_patterns = df_seq_patterns.rename(columns={'Combination': 'Combination', 'SumBetweenIndexes': 'MeanSumIntervals'})
            df_seq_patterns['seqIndex'] = seq_index

            df_seq_patterns_concatenated = pd.concat([df_seq_patterns_concatenated, df_seq_patterns])
            
        return df_seq_patterns_concatenated



    def __standardize_and_train_kmeans(self, dataframe,n_clust):
        """
        Standardizes the input data and trains a KMeans clustering model.

        This method standardizes the input data using StandardScaler and then trains a KMeans
        clustering model based on the standardized data.

        Parameters:
        - dataframe: Input DataFrame to be standardized and used for KMeans training
        - n_clust: Number of clusters for the KMeans algorithm

        Returns:
        - Trained KMeans clustering model
        - Fitted StandardScaler instance
        """
        scaler = StandardScaler()
        scaled_values=scaler.fit_transform(dataframe)
        scaled_dataframe = pd.DataFrame(scaled_values, columns=dataframe.columns)
        if self.algo_clustering=="kmeans":
            clustering_algo = KMeans(n_clusters=n_clust,n_init="auto")
        elif self.algo_clustering=="spectral_clust":
            clustering_algo = SpectralClustering(n_clusters=n_clust)
        clustering_algo.fit(scaled_dataframe.values)
        
        return clustering_algo , scaler


    def __meanSumIntervals_2_clusters(self, pattern_occurrences,n_clust,len_dataset):
        """
        Generates KMeans instances and associated information for pattern occurrences.

        This method processes pattern occurrences, creates KMeans clustering instances, and computes
        cluster frequencies for each pattern based on the occurrences.

        Parameters:
        - pattern_occurrences: DataFrame containing pattern occurrences and interval information
        - n_clust: Number of clusters for the KMeans algorithm
        - len_dataset: Length of the dataset related to the pattern occurrences

        Returns:
        - Dictionary mapping patterns to their respective KMeans instances, scaler instances,
          occurrences count, and cluster frequencies
        """
        # Initialize an empty dictionary to store information related to KMeans instances for each pattern
        # Iterate through pattern occurrences grouped by pattern to process each pattern's occurrences
        # For each pattern:
        #   - Split 'MeanSumIntervals' column into separate columns based on the number of elements in the list
        #   - Create DataFrame with intervals columns concatenated with the original DataFrame
        #   - Standardize the data and train a KMeans model for clustering
        #   - Compute cluster frequencies and store information in the dictionary

        # Example logic:
        # 1. Process pattern occurrences, group by pattern, and iterate through each pattern
        # 2. Standardize and train a KMeans model for clustering occurrences of each pattern
        # 3. Calculate cluster frequencies based on the occurrences and dataset length

        # Return a dictionary mapping patterns to their respective KMeans instances,
        # scaler instances, occurrences count, and cluster frequencies
    
        pattern_occurrences['Combination'] = pattern_occurrences['Combination'].astype(str)

        dict_k_means_instances={}

        dict_by_pattern = {x:y for x,y in pattern_occurrences.groupby('Combination')}

        for pattern, df_pattern in dict_by_pattern.items(): 
            
            # Split 'MeanSumIntervals' column into separate columns based on the number of elements in the list
            intervals_columns = df_pattern['MeanSumIntervals'].apply(pd.Series)
            intervals_columns.columns = [f'Interval_{i}' for i in range(len(intervals_columns.columns))]

            # Concatenate the original DataFrame with the new intervals columns
            df_pattern = pd.concat([df_pattern[['Combination', 'seqIndex']], intervals_columns], axis=1)

            K_means_instance_fitted, scaler_instance_fitted = self.__standardize_and_train_kmeans(df_pattern.drop(columns=['Combination', 'seqIndex']),n_clust)

            clusters = K_means_instance_fitted.predict(df_pattern.drop(columns=['Combination', 'seqIndex']).values)


            # Count frequencies of each cluster label
            cluster_frequencies = [round(x / len_dataset,4) for x in [list(clusters).count(i) for i in range(n_clust)]]

            # Store KMeans instance, pattern occurrences count, and cluster frequencies in the dictionary
            dict_k_means_instances[pattern] = [K_means_instance_fitted,scaler_instance_fitted, len(df_pattern), cluster_frequencies]
            
        return dict_k_means_instances



       
    def fit(self, X):
        """
        Fits the Seq2patterns model to the input data.

        This method fits the Seq2patterns model by extracting top-k frequent patterns from the input data,
        identifying occurrences of these patterns, and creating KMeans instances for pattern clustering.

        Parameters:
        - X: Input data for model fitting, containing sequence and interval information

        Returns:
        - None
        """
        # Extract top-k frequent patterns from the input data sequences
        # Identify occurrences of these patterns within the sequences along with associated intervals
        # Generate KMeans instances for clustering pattern occurrences
        
        data_seq = X
        
        self.freq_patterns = self.__top_k_frequent_patterns(
            data_seq,
            self.nb_of_frequent_patterns,
            self.min_len_of_frequent_pattern,
            self.kmeans_is_closed
        )
        pattern_occurrences = self.__find_pattern_occurrences(
            data_seq['Obs_seq'],
            data_seq['Intervals_seq'],
            self.freq_patterns
        ) 


        self.scaler_kmeans_dict = self.__meanSumIntervals_2_clusters(
            pattern_occurrences,
            self.n_clust,
            len(data_seq)
        )

    
    def transform(self,X):
        """
        Transforms the input data based on the fitted Seq2patterns model.

        This method transforms the input data using the fitted Seq2patterns model by identifying
        pattern occurrences in the provided sequences and clustering them using previously trained KMeans models.

        Parameters:
        - X: Input data for transformation, containing sequence and interval information

        Returns:
        - Transformed DataFrame with clustered pattern occurrences and associated frequencies
        """
        # Check if the Seq2patterns model has been fitted with frequent patterns and KMeans instances
        # Identify pattern occurrences in the input sequences and intervals
        # Cluster the identified pattern occurrences using previously trained KMeans models
        # Calculate and append cluster frequencies to the transformed data

        # Example logic:
        # 1. Verify if the Seq2patterns model has been fitted with frequent patterns and KMeans instances
        # 2. Find pattern occurrences in the input sequences and intervals
        # 3. Cluster the identified pattern occurrences using previously trained KMeans models
        # 4. Compute and append cluster frequencies to the transformed data

        # Return a DataFrame with clustered pattern occurrences and associated frequencies

        sample_test = X
        
        if self.freq_patterns is None or self.scaler_kmeans_dict is None:
            raise ValueError("Seq2patterns INSTANCE NOT FITTED")
            
        test_pattern_occurrences = self.__find_pattern_occurrences(
            sample_test['Obs_seq'],
            sample_test['Intervals_seq'], 
            self.freq_patterns)

        test_pattern_occurrences_clustered=pd.DataFrame()

        rows = []

        for index, row in test_pattern_occurrences.iterrows():
            intervals_columns = pd.Series(row['MeanSumIntervals'])
            intervals_columns.index = [f'Interval_{i}' for i in range(len(intervals_columns))]

            updated_row = pd.concat([pd.Series({'Combination': row['Combination'], 'seqIndex': row['seqIndex']}), intervals_columns])

            for pattern, scaler_kmeans_and_freq in self.scaler_kmeans_dict.items():
                kmeans = scaler_kmeans_and_freq[0]
                scaler= scaler_kmeans_and_freq[1]

                if row['Combination'] == pattern:
                    cluster_labels = kmeans.predict(updated_row.drop(['Combination', 'seqIndex']).values.reshape(1, -1))
                    updated_row['Cluster'] = cluster_labels[0]

                    # Extract only the required columns
                    updated_row = updated_row[['Combination', 'seqIndex', 'Cluster']]
                    updated_row['Combination'] = str(updated_row['Combination']) + "c"+ str(updated_row['Cluster'])

                    # Append the selected columns to the list
                    row_dict = updated_row.to_dict()
                    row_dict['freq'] = scaler_kmeans_and_freq[3][updated_row['Cluster']]
                    rows.append(row_dict)
                    
                    
        return pd.DataFrame(rows)
    

    
    def fit_transform(self,X):
        """
        Fits the Seq2patterns model to the input data and transforms it.

        This method fits the Seq2patterns model by extracting top-k frequent patterns from the input data,
        identifying occurrences of these patterns, and creating KMeans instances for pattern clustering.
        Then, it transforms the input data based on the fitted model by clustering pattern occurrences.

        Parameters:
        - X: Input data for model fitting and transformation, containing sequence and interval information

        Returns:
        - Transformed DataFrame with clustered pattern occurrences and associated frequencies
        """
        
        self.fit(X)
        return self.transform(X)