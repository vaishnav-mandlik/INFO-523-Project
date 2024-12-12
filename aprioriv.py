import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('apriori.csv')  
basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# basket.to_csv('basket.csv',index = False)
basket_sample = basket.sample(frac=0.1, random_state=42)  

print(f"Original dataset size: {basket.shape}")
print(f"Sampled dataset size: {basket_sample.shape}")

min_support = 0.01 
frequent_itemsets = apriori(basket_sample, min_support=min_support, use_colnames=True)
print(f"Number of frequent itemsets: {len(frequent_itemsets)}")

min_lift = 1.0 
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
print(f"Number of association rules: {len(rules)}")

frequent_itemsets.to_csv('frequent_itemsets_sample.csv', index=False)
rules.to_csv('association_rules_sample.csv', index=False)

print("Apriori processing complete.")