import pandas as pd

print("In Combining Columns")
# Importing the dataset
data = pd.read_csv('train_with_no_NaN.csv')

#Product of (review score)^2 and number of reviews
X=data['number_of_reviews']*data['review_scores_rating']


data.drop('number_of_reviews',1,inplace=True)
data.drop('review_scores_rating',1,inplace=True)
#data.append({'final_review_score': X}, ignore_index=True)
data['final_review_score'] = X
data.to_csv('train_with_cc_no_NaN.csv' , encoding="utf-8")
print("File Created :  train_with_cc_no_NaN.csv ")

import Add_Sentiments

