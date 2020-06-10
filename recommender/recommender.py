import pandas as pd

class Recommender():
    def __init__(self, db):
        self.cols = ['item_id', 'title', 'rating', 'reviews', 'link', 'img']
        self.recommendations = db.sights[self.cols].to_dict(orient='records')
    
    def lda(self, db, popularity, topics):
        
        # if one topic selected get the sights that scored highest in it
        if len(topics) == 1:
            sorting_values = [f'pop_topic{topics[0]}'] if popularity else [f'unpop_topic{topics[0]}']
            
            recommend = db.sights.sort_values(by=sorting_values, ascending=False)
            recommend['index'] = [x for x in range(recommend.shape[0])]
            self.recommendations = recommend[self.cols].to_dict(orient='records')
            return
        
        # if multiple topics selected
        select = [f'topic{i}' for i in topics]
        select.append('pop_score' if popularity else 'unpop_score')
        test = db.sights[select]
        coef = []
        for tup in test.values:
            new_coef = min(tup[:-1]) / max(tup[:-1]) if max(tup[:-1]) > 0 else 0
            new_coef *= min(tup[:-1]) * tup[-1]
            coef.append(new_coef)
            
        db.sights['coef'] = coef
        recommend = db.sights.sort_values(by=['coef'], ascending=False)
        recommend['index'] = [x for x in range(recommend.shape[0])]
        
        
        self.recommendations = recommend[self.cols].to_dict(orient='records')
        return
    
    def funksvd(self, db, user_id):
        rated, _ = db.get_rated(user_id)
        items = db.sights.index.tolist()
        predictions = []
        for item in items:
            predictions.append(db.svd.predict(uid=user_id, iid=item))
        predictions.sort(key=lambda x: x[3], reverse=True)
        pred = [x[1] for x in predictions]
        
        print(pred[:10])
        
        self.recommendations = db.sights.copy()
        self.recommendations['item_id'] = pd.Categorical(self.recommendations['item_id'], pred)
        self.recommendations.sort_values('item_id', inplace=True)
        self.recommendations.drop(list(rated), inplace=True)
        
        self.recommendations = self.recommendations[self.cols].to_dict(orient='records')
        
        