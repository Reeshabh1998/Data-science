#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: -
# A Mobile Phone manufacturing company wants to launch its three brand new phone into the market, but before going with its traditional marketing approach this time it want to analyze the data of its previous model sales in different regions and you have been hired as an Data Scientist to help them out, use the Association rules concept and provide your insights to the company’s marketing team to improve its sales.
# 

# # Objective :- 
#     use the Association rules concept and provide your insights to the company’s marketing team to improve its sales.
#     

# In[3]:


import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
#book = []
#with open("D:\\360Digi\\book.csv") as f:
#    book = f.read()
myphonedata = pd.read_csv("D:\\360Digi\\myphonedata.csv")
myphonedata


# In[4]:


myphonedata = myphonedata.iloc[:,3:]
myphonedata


# # EDA

# In[9]:


dataset1 =myphonedata.copy()


# In[11]:


zero =[]
one = []
for i in dataset1.columns:
    zero.append(list(dataset1[i].value_counts())[0])
    one.append(list(dataset1[i].value_counts())[1])


# In[13]:


count_df = pd.DataFrame([zero,one], columns=dataset1.copy().columns)


# In[14]:


count_df.head()


# In[20]:


myphonedata.hist(grid=True, rwidth=0.9, figsize=(10,10)) 


# In[22]:


a = myphonedata.corr(method ='pearson')
sns.heatmap(a>0.85,annot=True)


# In[23]:


myphonedata.boxplot(grid=True,figsize=(10,5))


# In[ ]:





# In[ ]:





# In[24]:


from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(myphonedata, min_support = 0.05, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(1, 11)), height = frequent_itemsets.support[1:11], color ='rgmyk')
plt.xticks(list(range(1, 11)), frequent_itemsets.itemsets[1:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules


# In[25]:


################################# Extra part ###################################
#Redudancy is defined as the storing of same data multiple time#
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




