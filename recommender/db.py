from surprise import SVD, Dataset, Reader
import pandas as pd

class DataBase():
    def __init__(self, users, sights, ratings, svd):
        self.users = users
        self.sights = sights
        self.ratings = ratings
        self.svd = svd
        self.max_uid = max(self.ratings.u_id.tolist())
        self.cols = ['item_id', 'title', 'rating', 'reviews', 'link', 'img']

    def get_attractions(self):
        return sorted(self.sights[['item_id', 'title']].to_dict(orient='records'), key=lambda x: x['title'])

    def add_user(self, username, item, rating, retrain=False):
        try: 
            uid = self.users[username]
        except:
            uid = self.max_uid + 1
            
        if not ((self.ratings['u_id'] == uid) & (self.ratings['i_id'] == item)).any():
            self.ratings = self.ratings.append({'u_id': uid, 'i_id': item, 'rating': rating}, ignore_index=True)
            self.users[username] = uid
            
            if retrain:
                self.retrain_svd()
        self.max_uid = max(self.ratings.u_id.tolist())
        return uid
    
    def add_rating(self, user, item, rating):
        self.ratings = self.ratings.append({'u_id': user, 'i_id': item, 'rating': rating}, ignore_index=True)
        self.retrain_svd()
    
    def get_user_id(self, username):
        try:
            return self.users[username]
        except:
            return None
    

    def get_rated(self, user_id):
        items = self.ratings[self.ratings['u_id'] == user_id].rename(columns={'rating': 'urating'})
        ratings = pd.merge(items, self.sights, how='left', left_on='i_id', right_on='item_id')[self.cols + ['urating']]
        return set(items.i_id), ratings.to_dict(orient='records')
    
    def get_items_info(self, items):
        return self.sights.loc[items][self.cols.extend(['user_rating'])].to_dict(orient='records')
        

    def retrain_svd(self):
        train = Dataset.load_from_df(self.ratings, Reader(rating_scale=(1, 5)))
        train = train.build_full_trainset()
        
        self.svd = SVD(n_factors=7, n_epochs=100, lr_all=0.005, reg_all=0.005, biased=True).fit(train)
        
