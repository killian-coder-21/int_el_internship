import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movie = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge both datasets on title
movies = movie.merge(credits, on='title')

# Keep important columns
movies = movies[['id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew']]

# Drop rows with null values
movies.dropna(inplace=True)

movies.head()

# Process 'overview' column

print (movies["overview"][0])

#Split string into list of separate words, remove 

movies["overview"] = movies["overview"].apply(lambda x: x.split())



#Process keywords column 

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies["keywords"] = movies["keywords"].apply(convert)
print (movies["keywords"])

#Process genres column 

movies["genres"] = movies["genres"].apply(convert)
print (movies["genres"])


#Process cast column, keep only top 3 actors from each 

def convert3(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L[:3]

movies["cast"] = movies["cast"].apply(convert3)
print (movies["cast"])

print (movies["crew"][0])

#Clean 
#Check entries for unterminated strings

for row in movies["crew"]:
  if row[-1] != "]":
    
    print (row)
#Looks like some of the data got cut off, try to remove last botched entry and close off 

corrected_rows=[]
for row in movies["crew"]:
  if row[-1] != "]":
    y = 0
    while row[-y] != "{":
      y+=1
    print (y)
    row = row[:(-y-2)] + "]"
    corrected_rows.append(row)
  else:
    corrected_rows.append(row)

movies["crew"] = corrected_rows



print (movies["crew"][28])

from os import lockf
#Change crew column into director column 

def get_director(obj):
  L = []
  loc = 0
  for i in ast.literal_eval(obj):
    if i["job"]=="Director":
      L.append(i["name"])
      break
    loc += 1
  return (L)

movies["crew"] = movies["crew"].apply(get_director)
print (movies["crew"])

# Create 'tags' column by combining overview + keywords + genres + cast + crew
movies['tags'] = movies['overview'] + movies['cast'] + movies['crew'] + movies['keywords']

# Final dataset with relevant columns
movies = movies[['id', 'title', 'tags']]

# Remove spaces from tags
movies['tags'] = movies['tags'].apply(lambda x: [i.replace(" ", "") for i in x])

# Stemming
ps = PorterStemmer()

def stemming(text):
    l = []
    for i in text:
        l.append(ps.stem(i))
    return " ".join(l)

movies['tags'] = movies['tags'].apply(stemming)

# Vectorization
vectorizer = CountVectorizer(max_features=500, stop_words='english')
vectors = vectorizer.fit_transform(movies['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

def Recommendation_system(movie_title):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    
    for i in distances[1:20]:
        print(movies.iloc[i[0]].title)

pickle.dump(movies, open('model.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
