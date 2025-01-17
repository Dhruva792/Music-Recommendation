import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
tracks = pd.read_csv('C://Users/dhruv/AppData/Local/Programs/Python/Python312/spotify_millsongdata.csv')
tracks.head()
tracks.shape
tracks.info()
tracks.dropna(inplace = True)
tracks.isnull().sum().plot.bar()
tracks = tracks.drop(['link', 'text'], axis = 1)
#data preprocessing
a =  tracks['artist'] + ' ' + tracks['song']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(a)
model = TSNE(n_components = 2, random_state = 0,init='random')
tsne_data = model.fit_transform(tfidf_matrix[:500])
plt.figure(figsize = (7, 7))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
#plt.show()
#find similarities
tracks['song'].nunique(), tracks.shape
tracks.drop_duplicates(subset=['song'], keep='first', inplace=True)
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['artist'])
tracks = tracks.head(10000)
def get_similarities(song_name, data):
        text_array1 = song_vectorizer.transform(data[data['song']==song_name]['artist']).toarray()
        num_array1 = data[data['song']==song_name].select_dtypes(include=np.number).to_numpy()
        if text_array1.shape[0] == 0 or num_array1.shape[1] == 0:
                return pd.Series([])
        sim = []
        for idx, row in data.iterrows():
                name = row['song']
                text_array2 = song_vectorizer.transform(data[data['song']==name]['artist']).toarray()
                num_array2 = data[data['song']==name].select_dtypes(include=np.number).to_numpy()
                text_sim = cosine_similarity(text_array1, text_array2)[0][0]
                num_sim = cosine_similarity(num_array1, num_array2)[0][0]
                sim.append(text_sim + num_sim)
        return sim
def recommend_songs(song_name, data=tracks):
        if tracks[tracks['song'] == song_name].shape[0] == 0:
                print('Some songs you may like:\n')
                for s in data.sample(n=5)['song'].values:
                        print(s)
                return
        data['similarity_factor'] = get_similarities(song_name, data)
        data.sort_values(by=['similarity_factor'],ascending = False,inplace=True)
        print(data[['song', 'artist']][2:7])
rec=input("Enter your favorite song:")
recommend_songs(rec)


