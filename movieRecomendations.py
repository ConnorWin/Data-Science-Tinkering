import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch and format data
data = fetch_movielens(min_rating=4.0)

#print data sets
print(repr(data['train']))
print(repr(data['test']))

#create model with loss='weighted approximate-rank pairwise' 
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30,num_threads=2)

def sample_recommendation(model, data, user_ids):
    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user input
    for user_id in user_ids:
        #movies they like now
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies the model preditcs they like
        scores = model.predict(user_id, np.argsort(n_items))
        #rank in order of most to least liked
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
sample_recommendation(model, data, [3, 25, 450])

