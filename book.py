#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: - 
# 	Kitabi Duniya , a famous book store in India, which was established before Independence, the growth of the company was incremental year by year, but due to online selling of books and wide spread Internet access its annual growth started to collapse, seeing sharp downfalls, you as a Data Scientist help this heritage book store gain its popularity back and increase footfall of customers and provide ways the business can improve exponentially, apply Association Rule Algorithm, explain the rules, and visualize the graphs for clear understanding of solution.
# 

# # 1.1.	Objective :- 
#     Book store to gain its popularity back and increase footfall of customers and provide ways the business can improve exponentially, by applying Association Rule Algorithm.

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#book = []
#with open("D:\\360Digi\\book.csv") as f:
#    book = f.read()
book = pd.read_csv("D:\\360Digi\\book.csv")
book


# In[ ]:





# # 3.Data Pre-processing
#  EDA

# In[12]:


book.info()


# In[3]:


book.hist(grid=True, rwidth=0.9, figsize=(10,10)) 


# In[17]:


book.boxplot(grid=True,figsize=(10,5))


# In[18]:


sns.pairplot(book)
plt.figure(figsize=(8,8))
plt.show()


# In[9]:


a = book.corr(method ='pearson')
sns.heatmap(a>0.85,annot=True)
#Since there is no correlation between variables


# In[ ]:





# In[4]:


from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(book, min_support = 0.05, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(1, 11)), height = frequent_itemsets.support[1:11], color ='rgmyk')
plt.xticks(list(range(1, 11)), frequent_itemsets.itemsets[1:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules


# In[5]:


rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


# In[6]:


################################# Extra part ###################################
#Redudancy is defined as the storing of same data multiple time#
#
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)


# # Summary:
# 1- Above the 10 unique Rule that we get by Apply Apriori Algo.
# 
# 2- Antecedent support variable tells us probability of antecedent product alone.
# 
# 3- The Support Value is the value of the two Product(Antecedents and Consequents)
# 
# 4- Confidence is an indication of how often the rule has been found to be True.
# 
# 5-The ratio of the observed support to that expected if X and Y were independent.

# In[ ]:




