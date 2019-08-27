import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv("movie_dataset.csv")
# print(data.head())
# print(data.columns)
features = ['keywords', 'director', 'cast', 'genres']

for feature in features:
    data[feature]=data[feature].fillna("")


def combine_feature(row):
    try:
       return row['keywords']+row['director']+row['cast']+row['genres']
    except:
        print("Error in ", row)


data["combined"] = data.apply(combine_feature, axis=1)
# print(data['combined'].head())

cv = CountVectorizer()


count_matrix = cv.fit_transform(data["combined"])
similarity_scores = cosine_similarity(count_matrix)
# print(similarity_scores)


def get_title_from_index(index):
    return data[data.index == index]["title"].values[0]


def get_index_from_title(title):
    return data[data.title == title]["index"].values[0]

movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(similarity_scores[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break


