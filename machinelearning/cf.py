''' Matrix Factorization - User Based Collaborative Filtering '''

import math
import os
import random
import sys
from collections import defaultdict
from operator import itemgetter

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

        print('Number of similar users to consider = %d' % self.n_sim_user, file=OUTFILE)
        print('Number of movies to recommend = %d\n' % self.n_rec_movie, file=OUTFILE)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        f_p = open(filename, 'r')
        for i, line in enumerate(f_p):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print('loading %s(%s)')
        f_p.close()
        print('loaded %s succussfully')

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

        print('\ndone splitting training and test set')
        print('train set = %s' % trainset_len, file=OUTFILE)
        print('test set = %s' % testset_len, file=OUTFILE)



    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        print('\nbuilding movie-users inverse table...')
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
        print('movie-users inverse table succussfully built')

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('\ntotal movie number = %d' % self.movie_count, file=OUTFILE)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print('\nbuilding user co-rated movies matrix...')

        for movie, users in movie2users.items():
            for _u in users:
                usersim_mat.setdefault(_u, defaultdict(int))
                for _v in users:
                    if _u == _v:
                        continue
                    usersim_mat[_u][_v] += 1
        print('co-rated movies matrix succussfully built!')

        # calculate similarity matrix
        print('\ncalculating user similarity matrix...')
        simfactor_count = 0
        print_step = 2000000

        for _u, related_users in usersim_mat.items():
            for _v, count in related_users.items():
                usersim_mat[_u][_v] = count / math.sqrt(
                    len(self.trainset[_u]) * len(self.trainset[_v]))
                simfactor_count += 1
                if simfactor_count % print_step == 0:
                    print('calculating user similarity factor(%d)' % simfactor_count, file=OUTFILE)

        print('calculation of user similarity matrix(similarity factor) done',)
        print('\nTotal similarity factor number = %d' % simfactor_count, file=OUTFILE)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        _k = self.n_sim_user
        _n = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:_k]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:_n]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('\nEvaluation start...', file=OUTFILE)

        #  variables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # variables for coverage
        all_rec_movies = set()
        # variables for popularity
        # popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for %d users')
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                # popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += self.n_rec_movie
            test_count += len(test_movies)

        prec = round(hit / (1.0 * rec_count), 3)
        rec = round(hit / (1.0 * test_count), 3)
        cov = round(len(all_rec_movies) / (1.0 * self.movie_count), 3)
        # popularity = popular_sum / (1.0 * rec_count)

        # print('\nprecision=%.4f\nrecall=%.4f\ncoverage=%.4f\npopularity=%.4f' %
               # (precision, recall, coverage, popularity), file=OUTFILE)
        print('\nprecision=%.4f\nrecall=%.4f\ncoverage=%.4f' % (prec, rec, cov), file=OUTFILE)
        print('precision=%.4f\nrecall=%.4f\ncoverage=%.4f' % (prec, rec, cov), file=GRAPHFILE)

        print('precision=%.4f' % (prec), file=GRAPHFILE2)

        print('recall=%.4f' % (rec), file=GRAPHFILE3)






if __name__ == '__main__':
    # Dataset
    DATAFILE = sys.argv[1]
    RATINGFILE = os.path.join('machinelearning', 'datasets', DATAFILE)

    # open file to write output to
    OUTFILE = open("results.txt", "w")
    GRAPHFILE = open("graph_data.txt", "w")

    GRAPHFILE2 = open("graph_data2.txt", "w")

    GRAPHFILE3 = open("graph_data3.txt", "w")

    # User input
    TRAIN_PERC = float(sys.argv[2])  # percentage of data to be used for training
    N_SIM_USERS = int(sys.argv[3])  # number of similar users to consider
    N_MOVIE_REC = int(sys.argv[4])  # number of movies to recommend to target users

    # Create object, calculate similarities for recommendation and evaluate
    for num_sim_users in [N_SIM_USERS - 5, N_SIM_USERS, N_SIM_USERS + 5]:
        usercf = UserBasedCF(num_sim_users, N_MOVIE_REC)
        usercf.generate_dataset(RATINGFILE, TRAIN_PERC)
        usercf.calc_user_sim()
        usercf.evaluate()

    # close file
    OUTFILE.close()
    GRAPHFILE.close()
    GRAPHFILE2.close()
    GRAPHFILE3.close()
