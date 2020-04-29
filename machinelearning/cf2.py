import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances

# console output settings
desired_width = 320
pd.set_option('display.width', desired_width)

pd.set_option('display.max_columns', 10)



# User's data
users_cols = ['userId', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./datasets/u.user', sep='|', names=users_cols, parse_dates=True)
# Ratings
rating_cols = ['userId', 'movieId', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./datasets/u.data', sep='\t', names=rating_cols)
# Movies
movies = pd.read_csv('./datasets/movies.csv', sep=',', encoding='latin-1')

###stripping year from movie title *Source Medium*
movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses.
movies['year'] = movies.year.str.extract('(\d\d\d\d)', expand=False)
# Note that expand=False simply means do not add this adjustment as an additional column to the data frame.
# Removing the years from the 'title' column.
movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending white space characters that may have appeared, using lambda function.
movies['title'] = movies['title'].apply(lambda x: x.strip())

# Merging movie data with their ratings

ratings.drop("unix_timestamp", inplace=True, axis=1)
movie_ratings = pd.merge(movies, ratings)
# merging movie_ratings data with the User's dataframe
df1 = pd.merge(movie_ratings, users)

sns.heatmap(df1.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')
plt.title('Heatmap of Movie Data')
plt.show()
plt.savefig('./heatmap.png')
# pre-processing
# dropping colums that aren't needed
df1.drop(df1.columns[[3, 4, 7]], axis=1, inplace=True)

# Pivot Table(This creates a matrix of users and movie_ratings)
ratings_matrix = ratings.pivot_table(index=['movieId'], columns=['userId'], values='rating').reset_index(drop=True)
ratings_matrix.fillna(0, inplace=True)

# Cosine Similarity(Creates a cosine matrix of similaraties ..... which is the pairwise distances
# between two items )

movie_similarity = 1 - pairwise_distances(ratings_matrix.to_numpy(), metric="cosine")
np.fill_diagonal(movie_similarity, 0)
ratings_matrix = pd.DataFrame(movie_similarity)



# Recommender





if __name__ == '__main__':

    try:
        #user_inp = input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp ="Toy Story"
        #user_inp = sys.argv[1]
        inp = movies[movies['title'] == user_inp].index.tolist()
        inp = inp[0]
        movies['similarity'] = ratings_matrix.iloc[inp]
        movies.columns = ['movie_id', 'title', 'genre', 'year', 'similarity']
        movies.head(3)
    except:
        print("...")
df = movies.sort_values(["similarity"], ascending=False)[1:10]
df.drop(df.columns[[0, 3]], axis=1, inplace=True)
df.similarity = df.similarity.apply(lambda x:round(x, 2))
example_string = df.to_string()
output_file = open('file.txt', 'w')
output_file.write(example_string)
output_file.close()




# df.to_csv('results.csv', header=names, index=False)
