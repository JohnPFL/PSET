import math
from scipy.stats import ttest_ind


class StatisticsCalculator:
    def __init__(self, data):
        self.data = data

    def calculate_average(self):
        if not self.data:
            return None
        return sum(self.data) / len(self.data)

    def calculate_standard_deviation(self):
        if not self.data:
            return None
        mean = self.calculate_average()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        std_deviation = math.sqrt(variance)
        return std_deviation

    def calculate_other_statistics(self):
        if not self.data:
            return None
        # You can add more statistical calculations here based on your needs
        # For example, median, minimum, maximum, etc.
        median = sorted(self.data)[len(self.data) // 2]
        minimum = min(self.data)
        maximum = max(self.data)
        return {
            'median': median,
            'minimum': minimum,
            'maximum': maximum,
            # Add more statistics as needed
        }

class IndependentTTestCalculator:
    def __init__(self, group1, group2):
        self.group1 = group1
        self.group2 = group2

    def calculate_standard_deviation(self, data):
        if not data:
            return None
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_deviation = (variance)**0.5
        return std_deviation

    def perform_t_test(self):
        if not self.group1 or not self.group2:
            return None

        std_group1 = self.calculate_standard_deviation(self.group1)
        std_group2 = self.calculate_standard_deviation(self.group2)

        if std_group1 == 0 or std_group2 == 0:
            # Handle zero standard deviation
            return None

        t_statistic, p_value = ttest_ind(self.group1, self.group2)
        return t_statistic, p_value


if __name__ == "__main__":
    # Example usage
    group1 = [23, 21, 18, 25, 27, 29, 30, 32]
    group2 = [28, 30, 32, 35, 38, 40, 42, 45]

    t_test_calculator = IndependentTTestCalculator(group1, group2)
    t_statistic, p_value = t_test_calculator.perform_t_test()

    print(f"Group 1: {t_test_calculator.group1}")
    print(f"Group 2: {t_test_calculator.group2}")
    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")

    # Interpret the result based on the p-value
    significance_level = 0.05
    if p_value < significance_level:
        print("The difference between the groups is statistically significant.")
    else:
        print("There is no significant difference between the groups.")
        
    # Example usage
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    calculator = StatisticsCalculator(data)

    average = calculator.calculate_average()
    std_deviation = calculator.calculate_standard_deviation()
    other_statistics = calculator.calculate_other_statistics()

    print(f"Data: {data}")
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_deviation}")
    print(f"Other Statistics: {other_statistics}")
