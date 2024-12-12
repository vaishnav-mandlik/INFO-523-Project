import pandas as pd
# import numpy as np

frequent_itemsets = pd.read_csv('fp_frequent_itemsets_sample.csv')
association_rules = pd.read_csv('fp_association_rules_sample.csv')

print(frequent_itemsets.head())
print(association_rules.head())

# for sorting rules by lift
rules_sorted_by_lift = association_rules.sort_values('lift', ascending=False)

strong_rules = rules_sorted_by_lift[rules_sorted_by_lift['lift'] > 1]

print(strong_rules)
