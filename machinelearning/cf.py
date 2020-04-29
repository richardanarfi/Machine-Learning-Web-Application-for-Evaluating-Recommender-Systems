import math
import os
import random
import sys
from operator import itemgetter

import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.tools as tls
from surprise import Reader

random.seed(0)


class UserBasedCF:
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self, n_sim_user=20, n_rec_movie=10):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = n_sim_user
        self.n_rec_movie = n_rec_movie

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Number of similar users to consider = %d' % self.n_sim_user, file=outfile)
        print('Number of movies to recommend = %d\n' %
              self.n_rec_movie, file=outfile)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print('loading %s(%s)' % (filename, i), file=outfile)
        fp.close()
        print('loaded %s succussfully' % filename, file=outfile)

    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('\ndone splitting training and test set', file=outfile)
        print('train set = %s' % trainset_len, file=outfile)
        print('test set = %s' % testset_len, file=outfile)

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        print('\nbuilding movie-users inverse table...', file=outfile)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print('movie-users inverse table succussfully built', file=outfile)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('\ntotal movie number = %d' % self.movie_count, file=outfile)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print('\nbuilding user co-rated movies matrix...', file=outfile)

        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print('co-rated movies matrix succussfully built!', file=outfile)

        # calculate similarity matrix
        print('\ncalculating user similarity matrix...', file=outfile)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating user similarity factor(%d)' %
                          simfactor_count, file=outfile)

        print('calculation of user similarity matrix(similarity factor) done',
              file=outfile)
        print('\nTotal similarity factor number = %d' %
              simfactor_count, file=outfile)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
        # return the N best movies

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('\nEvaluation start...', file=outfile)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for %d users' % i, file=outfile)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('\nprecision=%.4f\nrecall=%.4f\ncoverage=%.4f\npopularity=%.4f' %
              (precision, recall, coverage, popularity), file=outfile)


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def predictions(data, rec_inp=10, split_inp=.20):
    trainset, testset = train_test_split(data, test_size=split_inp)
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    top_n = get_top_n(predictions, n=rec_inp)
    for uid, user_ratings in top_n.items():
        uid, [iid for (iid, _) in user_ratings]
        return predictions


from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold, train_test_split


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def generate_graph(data, user_inp):
    kf = KFold(n_splits=5)
    algo = SVD()
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        # precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
        results = []
        for i in range(2, 11):
            precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=2.5)

            # Precision and recall can then be averaged over all users
            prec = sum(prec for prec in precisions.values()) / len(precisions)
            rec = sum(rec for rec in recalls.values()) / len(recalls)
            results.append({'K': i, 'Precision': prec, 'Recall': rec})
    Rec = []
    Precision = []
    Recall = []
    for i in range(0, user_inp):
        Rec.append(results[i]['K'])
        Precision.append(results[i]['Precision'])
        Recall.append(results[i]['Recall'])
    fig = plt.figure()
    plt.plot(Rec, Precision)
    plt.xlabel('# of Recommendations')
    plt.ylabel('Precision')
    plt2 = plt.twinx()
    plt2.plot(Rec, Recall, 'r')
    plt.ylabel('Recall')
    for tl in plt2.get_yticklabels():
        tl.set_color('r')
    plt.show()
    plt.savefig('./recallprecision.png')
    fig = plt.Figure()
    ax = fig.gca()
    ax2 = ax.twinx()
    ax2.plot(Rec,Precision)
    ax.plot(Rec,Precision)
   # canvas = FigureCanvas(fig)
    plotly_fig = tls.mpl_to_plotly(fig)
    py.plot(plotly_fig, filename='Precision')




if __name__ == '__main__':
    # Dataset
    datafile = sys.argv[1]
    ratingfile = os.path.join('machinelearning', 'datasets', datafile)

    # open file to write output to    
    outfile = open("results.txt", "w")

    # User input
    train_perc = float(sys.argv[2])  # percentage of data to be used for training
    n_sim_users = int(sys.argv[3])  # number of similar users to consider
    n_movie_rec = int(sys.argv[4])  # number of movies to recommend to target users

    # Create object, calculate similarities for recommendation and evaluate 
    usercf = UserBasedCF(n_sim_users, n_movie_rec)
    usercf.generate_dataset(ratingfile, train_perc)
    usercf.calc_user_sim()
    usercf.evaluate()
    outfile.close()

reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(ratingfile, reader=reader)
generate_graph(data, n_movie_rec - 1)
