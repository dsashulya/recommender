import numpy as np
import pandas as  pd


uid = 48043
item = 2
rating = 5
ratings = pd.read_csv('data/ratings_reshaped.csv')
ratings = ratings.append({'u_id': uid, 'i_id': item, 'rating': rating}, ignore_index=True)
print(ratings.tail())
print(max(ratings.u_id.tolist()))