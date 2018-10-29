#By Tahiya Chowdhury

# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD, evaluate, SlopeOne, SVDpp, KNNBaseline
from collections import defaultdict 

import warnings; warnings.simplefilter('ignore')


# In[2]:


movie_db = pd. read_csv('~/movies_metadata.csv')
movie_db.head()


# In[3]:


movie_db['genres'] = movie_db['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[4]:


vote_counts = movie_db[movie_db['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movie_db[movie_db['vote_average'].notnull()]['vote_average'].astype('int')


# In[5]:


C = vote_averages.mean()  #C = the mean average vote across the dataset
C


# In[6]:


m = vote_counts.quantile(0.90)  #minimum votes required to be eigible for the list
m


# In[7]:


movie_db['year'] = pd.to_datetime(movie_db['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[8]:


eligible = movie_db[(movie_db['vote_count'] >= m) & (movie_db['vote_count'].notnull()) & (movie_db['vote_average'].notnull())][['title', 'year', 'genres', 'vote_count', 'vote_average', 'popularity' ]]

eligible['vote_count'] = eligible['vote_count'].astype('int')
eligible['vote_average'] = eligible['vote_average'].astype('int')

eligible.shape


# In[9]:


def weighted_rating(x):
    
    v = x['vote_count']
    R = x['vote_average']
    
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[10]:


eligible['wr'] = eligible.apply(weighted_rating, axis=1)


# In[11]:


eligible = eligible.sort_values('wr', ascending=False)


# In[12]:


eligible.head(250)


# In[13]:


gen = movie_db.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
gen.name = 'genre'

gen_movie_db = movie_db.drop('genres', axis=1).join(gen) # identifying the main genre of a movie


# In[14]:


gen_movie_db


# In[16]:


def genre_list(genre):
    
    df = gen_movie_db[gen_movie_db['genre'] == genre]
    
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90)
    
    eligible = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'genre', 'vote_count', 'vote_average', 'popularity']]
    eligible['vote_count'] = eligible['vote_count'].astype('int')
    eligible['vote_average'] = eligible['vote_average'].astype('int')
    
    eligible['wr'] = eligible.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    eligible = eligible.sort_values('wr', ascending=False).head(250)
    
    return eligible


# In[18]:


# recommendation list for a particular genre
genre_list('Crime').head(10)


# In[19]:


genre_list('Animation').head(10)  


# In[20]:


credits = pd.read_csv('~/credits.csv')
keywords = pd.read_csv('~/keywords.csv')


# In[21]:


movie_db = movie_db.drop([19730, 29503, 35587])


# In[22]:


keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

movie_db['id'] = movie_db['id'].astype('int')


# In[23]:


movie_db.shape


# In[24]:


movie_db = movie_db.merge(credits, on='id')

movie_db = movie_db.merge(keywords, on='id')


# In[25]:


movie_db.shape


# In[26]:


movie_db['cast'] = movie_db['cast'].apply(literal_eval)
movie_db['crew'] = movie_db['crew'].apply(literal_eval)

movie_db['keywords'] = movie_db['keywords'].apply(literal_eval)

movie_db['cast_size'] = movie_db['cast'].apply(lambda x: len(x))
movie_db['crew_size'] = movie_db['crew'].apply(lambda x: len(x))


# In[27]:


def filter_director(x):   # finding the director from the crew list
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[28]:


movie_db['director'] = movie_db['crew'].apply(filter_director)


# In[29]:


movie_db['cast'] = movie_db['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

movie_db['cast'] = movie_db['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)  # finding the top 3 cast members from the cast list


# In[30]:


movie_db['keywords'] = movie_db['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[31]:


key_word = movie_db.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)

key_word.name = 'keyword'            


# In[32]:


key_word = key_word.value_counts()  

key_word[:10]


# In[33]:


key_word = key_word[key_word > 5]      # finding frequently appeared keywords


# In[34]:


key_word[:10]


# In[35]:


def filter_keywords(x):        
    words = []
    for i in x:
        if i in key_word:
            words.append(i)
    return words


# In[36]:


movie_db['keywords'] = movie_db['keywords'].apply(filter_keywords)


# In[37]:


movie_db['keywords']=movie_db['keywords'].to_frame()  #converting keyword list to a dataframe


# In[38]:


movie_db['keywords']


# In[39]:


movie_db['genres']= movie_db['genres'].to_frame()           #converting genre list to a dataframe


# In[40]:


movie_db['genres']


# In[41]:


movie_db['cast']= movie_db['cast'].to_frame()       #converting cast list to a dataframe


# In[42]:


movie_db['cast']


# In[43]:


movie_db['director']= movie_db['director'].to_frame()          #converting director list to a dataframe


# In[44]:


movie_db['director']


# In[45]:


movie_db['combined'] =  movie_db['cast'] + movie_db['genres']


# In[46]:


movie_db['combined']


# In[47]:


movie_db['combined']= movie_db['combined'] + movie_db['keywords']


# In[48]:


movie_db['combined']


# In[49]:


movie_db['director'] = movie_db['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))


# In[50]:


movie_db['director'] = movie_db['director'].apply(lambda x: [x, x, x])


# In[51]:


movie_db['director']


# In[52]:


movie_db['combined']= movie_db['combined'] + movie_db['director']


# In[53]:


movie_db['combined']           #top 3 cast, keyword, genre and director in one dataframe


# In[54]:


movie_db['combined'] = movie_db['combined'].apply(lambda x: ' '.join(x))


# In[55]:


movie_db['combined']


# In[56]:


count = CountVectorizer(analyzer='word', decode_error='ignore', encoding='utf-8',ngram_range=(1, 2), min_df=0, stop_words='english')


# In[57]:


count


# In[58]:


term_document_matrix = count.fit_transform(movie_db['combined'])  # returns term-document matrix by feature extraction


# In[59]:


cosine_sim = cosine_similarity(term_document_matrix, term_document_matrix)


# In[60]:


cosine_sim


# In[61]:


movie_db = movie_db.reset_index()


# In[62]:


titles = movie_db['title']

indice = pd.Series(movie_db.index, index = movie_db['title'])


# In[63]:


def movie_recommendation(title):          # to get the recommendation for a particular movie
    idx = indice[title]
    
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    
    sim_score = sim_score[1:26]
    movie_indice = [i[0] for i in sim_score]
    
    movies = movie_db.iloc[movie_indice][['title', 'year', 'genres', 'vote_count', 'vote_average', 'popularity']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    
    eligible = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title', 'year', 'genres', 'vote_count', 'vote_average', 'popularity', ]]
    eligible['vote_count'] = eligible['vote_count'].astype('int')
    eligible['vote_average'] = eligible['vote_average'].astype('int')
    eligible['wr'] = eligible.apply(weighted_rating, axis=1)
    eligible = eligible.sort_values('wr', ascending=False)
    
    return eligible


# In[64]:


#recommendation based on a particular movie
movie_recommendation('Memento').head(10)


# In[65]:


movie_recommendation('The Godfather').head(10)


# In[66]:


#Collaborative Filtering based Recommendation
reader = Reader()


# In[67]:


ratings = pd.read_csv('~/ratings_small.csv')
ratings.head()


# In[68]:


n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies)


# In[69]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)
data.split(n_folds=5)


# In[70]:


svd = SVD()
perf1= evaluate(svd, data, measures=['RMSE', 'MAE'])


# In[71]:


slp = SlopeOne()
perf2= evaluate(slp, data, measures=['RMSE', 'MAE'])


# In[72]:


knn = KNNBaseline()
perf3= evaluate(knn, data, measures=['RMSE', 'MAE'])


# In[73]:


trainset = data.build_full_trainset()
svd.train(trainset)


# In[74]:


testset = trainset.build_anti_testset()  
predictions = svd.test(testset)


# In[79]:


#predicting ratings for unseen movies using SVD
df = pd.DataFrame(predictions)
df= df.sort_values(['est','uid'], ascending=[False, True])
df


# In[80]:


def get_top_n_recommendation(predictions, n=10):
    
    top_n = defaultdict(list)
    
    for uid, iid, r_ui, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# In[81]:


top_n = defaultdict(list)    # getting top-10 recommendation by predicted ratings 

top_n = get_top_n_recommendation(predictions, n=10)

for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

