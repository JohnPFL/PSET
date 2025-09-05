from abc import abstractmethod, ABC
from os import stat
from typing import Type
import random
import warnings

class ScoreUtilityInterface(ABC):
    @abstractmethod
    def clean_data(data):
        pass

class ScoreUtility(ScoreUtilityInterface):
    @staticmethod
    def is_string(value):
        return isinstance(value, str)

    @staticmethod
    def initialize_rows_to_drop(data):
            
        # Apply the function to both columns 'a' and 'b' and get boolean DataFrames
        is_string_a = data['anchor_score'].apply(ScoreUtility.is_string)
        is_string_b = data['homophones_score'].apply(ScoreUtility.is_string)
        is_string_c = data['synonyms_score'].apply(ScoreUtility.is_string)

        # Combine the boolean DataFrames to identify rows where either column 'a' or 'b' is a string
        rows_to_drop = is_string_a | is_string_b | is_string_c

        return rows_to_drop
    
    @staticmethod
    def clean_data(data):
        rows_to_drop = ScoreUtility.initialize_rows_to_drop(data)
        # Drop those rows from the DataFrame
        data = data[~rows_to_drop]
        return data

class ScoreComparator:
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        self.data = data
        self.score_utility_obj = score_utility_obj
        self.score_utility_obj.clean_data(self.data)
        self.total_rows = len(self.data)
        self.exclude_anchor_score()

    def exclude_anchor_score(self):
        # Exclude "anchor_score" column
        self.data = self.data.drop(columns=['a'], errors='ignore')

    def compare_scores(self):
        # Compare spl_var_score and synonym_score
        b_c_score = (self.data['homophones_score'] > self.data['synonyms_score']).sum()
        c_b_score = (self.data['synonyms_score'] > self.data['homophones_score']).sum()
        b_c_prevalence = b_c_score / self.total_rows
        c_b_prevalence = c_b_score / self.total_rows

        return b_c_prevalence, c_b_prevalence

class ScoreComparatorExtended(ScoreComparator):
    def __init__(self, data, score_utility_obj: Type[ScoreUtilityInterface] = ScoreUtility()):
        super().__init__(data, score_utility_obj)
    
    def check_for_non_matching_scores(self):
        homophones_score_rows = (self.data['homophones_score'] > self.data['phon_dist_score']) & (self.data['homophones_score'] > self.data['synonyms_score']) & (self.data['homophones_score'] > self.data['graph_dist_score'])
        synonyms_score_rows = (self.data['synonyms_score'] > self.data['homophones_score']) & (self.data['synonyms_score'] > self.data['phon_dist_score']) & (self.data['synonyms_score'] > self.data['graph_dist_score'])
        phon_dist_score_rows = (self.data['phon_dist_score'] > self.data['homophones_score']) & (self.data['phon_dist_score'] > self.data['synonyms_score']) & (self.data['phon_dist_score'] > self.data['graph_dist_score']) 
        graph_dist_score_rows = (self.data['graph_dist_score'] > self.data['homophones_score']) & (self.data['graph_dist_score'] > self.data['synonyms_score']) & (self.data['graph_dist_score'] > self.data['phon_dist_score'])
        none_condition_met = ~(homophones_score_rows | synonyms_score_rows | phon_dist_score_rows | graph_dist_score_rows)
        non_condition = None
        n_non_matching_conditions = 0
        if none_condition_met.any():
            n_non_matching_conditions = len(self.data[none_condition_met])
            non_condition = self.data[none_condition_met]
        return non_condition, n_non_matching_conditions


    def compare_scores(self):
        if 'phon_dist_score' in self.data.columns:
            non_matching_condition, n_non_matching_conditions = self.check_for_non_matching_scores()
            if n_non_matching_conditions > 0:
                print(f'Non-matching conditions found: {n_non_matching_conditions}, rows: {non_matching_condition}')
                warnings.warn('Non-matching conditions found')

                score_cols = ['homophones_score', 'synonyms_score', 'phon_dist_score', 'graph_dist_score']

                for idx in non_matching_condition.index:
                    row = self.data.loc[idx]
                    if row['homophones_score'] == 1.0:
                        self.data.loc[idx, ['synonyms_score', 'phon_dist_score', 'graph_dist_score']] = 0.0
                    else:
                        if all(row[col] != 1.0 for col in ['synonyms_score', 'phon_dist_score', 'graph_dist_score']):
                            chosen_col = random.choice(['synonyms_score', 'phon_dist_score', 'graph_dist_score'])
                            self.data.loc[idx, chosen_col] = 1.0

                all_zero_condition = self.data[['homophones_score', 'synonyms_score', 'phon_dist_score', 'graph_dist_score']].sum(axis=1) == 0.0
                print(f"Dropping {all_zero_condition.sum()} rows where all scores are 0.")
                self.data = self.data[~all_zero_condition]
                self.total_rows = len(self.data)

            homophones_score = ((self.data['homophones_score'] > self.data['phon_dist_score']) & 
                    (self.data['homophones_score'] > self.data['synonyms_score']) & 
                    (self.data['homophones_score'] > self.data['graph_dist_score'])).sum()
            synonyms_score = ((self.data['synonyms_score'] > self.data['homophones_score']) & 
                    (self.data['synonyms_score'] > self.data['phon_dist_score']) & 
                    (self.data['synonyms_score'] > self.data['graph_dist_score'])).sum()
            phon_dist_score = ((self.data['phon_dist_score'] > self.data['homophones_score']) & 
                    (self.data['phon_dist_score'] > self.data['synonyms_score']) & 
                    (self.data['phon_dist_score'] > self.data['graph_dist_score'])).sum()
            graph_dist_score = ((self.data['graph_dist_score'] > self.data['homophones_score']) & 
                    (self.data['graph_dist_score'] > self.data['synonyms_score']) & 
                    (self.data['graph_dist_score'] > self.data['phon_dist_score'])).sum()

            homophones_prevalence = homophones_score / self.total_rows
            synonyms_prevalence = synonyms_score / self.total_rows
            phon_dist_prevalence = phon_dist_score / self.total_rows
            graph_dist_prevalence = graph_dist_score / self.total_rows
            return homophones_prevalence, synonyms_prevalence, phon_dist_prevalence, graph_dist_prevalence