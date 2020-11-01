import numpy as np
import pandas as pd
import subprocess
import re

from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

import config as cfg
from metrics import average_precision_at_k


class BaseRecommender:
    """
    Base blank class to build recommendation systems classes on top of it
    """

    def __init__(self):
        pass

    def fit(self, train_df, col_user=cfg.USER_COL, col_item=cfg.ITEM_COL, col_rating=cfg.DEFAULT_RATING_COL) -> None:
        """
        perform train procedure for the recommendation algorithm
        :param col_item: name for items column
        :param col_user: name for user column
        :param col_rating: name for rating column
        :param train_df: pandas data frame with users, items and ratings columns
        :return: None
        """
        self.train_df = train_df.copy()
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating

    def predict(self, test_df, k=cfg.DEFAULT_K):
        """
        predicts recommendations for test users
        :param test_df: pandas data frame with users, items and ratings columns
        :param k: number of items to predict per user
        :return: prediction pandas data frame with two columns first contains user's
        and second contains list of recommended items
        """
        pass

    def get_uii_matrix(self):
        self.users = list(np.sort(self.train_df[self.col_user].unique()))
        self.items = list(np.sort(self.train_df[self.col_item].unique()))
        ratings = list(self.train_df[self.col_rating])
        usdtype = pd.api.types.CategoricalDtype(categories=self.users)
        rows = self.train_df[self.col_user].astype(usdtype).cat.codes
        itemdtype = pd.api.types.CategoricalDtype(categories=self.items)
        cols = self.train_df[self.col_item].astype(itemdtype).cat.codes

        return sparse.csr_matrix((ratings, (rows, cols)), shape=(len(self.users), len(self.items)))


class MostPopularRecommender(BaseRecommender):
    """
    Performs most popular recommendations for the given dataset
    """

    def fit(self, train_df, col_user=cfg.USER_COL, col_item=cfg.ITEM_COL, col_rating=cfg.DEFAULT_RATING_COL):
        """
        Train the most popular items over the data set
        :param train_df: pandas data frame with users, items and ratings columns
        :param col_item: name for items column
        :param col_user: name for user column
        :param col_rating: name for rating column
        :return: None
        """
        BaseRecommender.fit(self, train_df, col_user, col_item, col_rating)
        self.top_ratings = train_df.groupby(by=self.col_item).count().reset_index().sort_values(by=col_rating,
                                                                                                ascending=False)

    # change sum to count cause we use non-binary rating
    def predict(self, test_df, k=cfg.DEFAULT_K):
        """
        Recommend k items for each user in test df
        :param test_df: pandas DataFrame with test users and truth values
        :param k: int number of items to recommend
        :return: pandas DataFrame with list of items to recommend to user
        """
        top_items = [self.top_ratings[self.col_item].values.tolist()[0:k]] * test_df[self.col_user].nunique()
        prediction = pd.DataFrame({self.col_user: test_df[self.col_user].unique(),
                                   self.col_item: top_items})

        return prediction


class TruncatedSVDRecommender(BaseRecommender):
    """
    Implements the collaborative filtering approach to recommend items based upon sklearn TruncatedSVD
    """

    def get_uii_matrix(self):
        """
        performs building of user item sparse matrix
        :return: scipy.sparse.csr_matrix containing users and items interactions
        """
        users = list(np.sort(self.train_df[self.col_user].unique()))
        items = list(np.sort(self.train_df[self.col_item].unique()))
        ratings = list(self.train_df[self.col_rating])
        user_dtype = pd.api.types.CategoricalDtype(categories=users)
        rows = self.train_df[self.col_user].astype(user_dtype).cat.codes
        item_dtype = pd.api.types.CategoricalDtype(categories=items)
        cols = self.train_df[self.col_item].astype(item_dtype).cat.codes

        return sparse.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(items))), users, items

    def fit(self, train_df, col_user=cfg.USER_COL, col_item=cfg.ITEM_COL,
            col_rating=cfg.DEFAULT_RATING_COL, n_components=800):
        """
        performs training of collaborative filtering recommender
        :param train_df:  pandas DataFrame with train data
        :param col_user: str column name for user
        :param col_item: str column name for item
        :param col_rating: str column name for ratings
        :param n_components: int number of components to use to performs TruncatedSVD
        :return: None
        """
        BaseRecommender.fit(self, train_df, col_user, col_item, col_rating)
        self.uii_matrix, self.users, self.items = self.get_uii_matrix()
        self.trunc_svd = TruncatedSVD(n_components=n_components)
        self.user_matrix = self.trunc_svd.fit_transform(self.uii_matrix)

    def predict(self, test_df, k=cfg.DEFAULT_K, batch_size=100):
        """
        performs recommendations for test users in test set
        recommends
        :param test_df: pandas DataFrame with test_users and truth recommendations
        :param k: int number of items to recommend
        :param batch_size: int size of batch to reduce memory usage while perform dense calculation
        :return: pandas DataFrame with k recommendations for each user in test_df
        """
        test_users_indices = [self.users.index(user) for user in
                              test_df[self.col_user].values if user in self.users]

        final = pd.DataFrame(columns=[self.col_user, self.col_item])
        intervals = [i * batch_size for i in range(len(test_users_indices) // batch_size + 1)] + [
            len(test_users_indices) + 1]
        for i in tqdm(range(len(intervals) - 1)):
            dense_matrix = self.user_matrix[test_users_indices[intervals[i]: intervals[i + 1]]].dot(
                np.diag(self.trunc_svd.singular_values_)).dot(
                self.trunc_svd.components_)

            user_list = []
            for ind, row in enumerate(dense_matrix):
                rec_items = [self.items[j] for j in row.argsort()[::-1] if row[j] > 0][:k]
                user_dict = {self.col_user: self.users[test_users_indices[ind + intervals[i]]],
                             self.col_item: rec_items}
                user_list.append(user_dict)

            result = pd.DataFrame.from_records(user_list)
            final = final.append(result, ignore_index=True)

        return final


class ALSRecommender(BaseRecommender):
    """
    implement alternating least squares algorithm implementation based on implicit library
    """

    def fit(self, train_df,
            col_user=cfg.USER_COL,
            col_item=cfg.ITEM_COL,
            col_rating=cfg.DEFAULT_RATING_COL,
            factors=100,
            confidence=5,
            regularization=0.1):
        """
        Trains implicit ALS recommender on train data
        :param train_df: pandas DataFrame with train data
         :param col_user: str column name for user
        :param col_item: str column name for item
        :param col_rating: str column name for ratings
        :param factors: int number of factors to use in ALS model
        :param confidence: int as described in implicit documentation
        :param regularization: float higher values mean stronger regularization
        :return: None
        """
        BaseRecommender.fit(self, train_df, col_user, col_item, col_rating)
        self.train_df[self.col_rating] = train_df[self.col_rating] * confidence
        self.uii_matrix = self.get_uii_matrix()
        self.als = AlternatingLeastSquares(factors=factors, use_gpu=False, regularization=regularization)
        self.als.fit(self.uii_matrix.T)

    def predict(self, test_df, k=cfg.DEFAULT_K):
        """
        recommend k items for each user in test_df
        :param test_df: pandas DataFrame with test_users and truth recommendations
        :param k: int number of items to recommend
        :return: pandas DataFrame with k recommendations for each user in test_df
        """
        test_users_indices = [self.users.index(user) for user in
                              test_df[self.col_user].values if user in self.users]

        prediction_records = []
        for item in test_users_indices:
            doc = {self.col_user: self.users[item],
                   self.col_item: [self.items[it[0]] for it in self.als.recommend(item,
                                                                                  self.uii_matrix,
                                                                                  k,
                                                                                  filter_already_liked_items=False)]}
            prediction_records.append(doc)
        prediction = pd.DataFrame.from_records(prediction_records)

        return prediction


class StratifyMostPopularRecommender(BaseRecommender):
    """
    StarSpace embeddings with train mode 0
    and clustering of KMeans above them. A
    nd most popular recommendations on top of it all
    """

    def __init__(self):
        """
        installs starspace if it's not in the work directory
        """
        super().__init__()
        try:
            subprocess.call(['starspace'])
            print('Starspace is already installed')
        except FileNotFoundError:
            print('Installing StarSpace')
            subprocess.call(['wget', 'https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip'])
            subprocess.call(['unzip', 'boost_1_63_0.zip'])
            subprocess.call(['sudo', 'mv', 'boost_1_63_0', '/usr/local/bin'])
            subprocess.call(['rm', '-rf', 'Starspace'])
            subprocess.call(['git', 'clone', 'https://github.com/facebookresearch/Starspace.git'])
            subprocess.call(['make', '-C', '/content/Starspace'])
            print('StarSpace installed')

    def fit(self, train_df, col_user=cfg.USER_COL, col_item=cfg.ITEM_COL, col_rating=cfg.DEFAULT_RATING_COL, clusters=500,
            starspace_df=None):
        """
        performs training of starspace user's representation
        and clustering them to get clusters for most popular predictions
        :param train_df: pandas DataFrame with train set
        :param starspace_df: pandas DataFrame prepared for starspace embeddings training
        :param col_user: str column name for user
        :param col_item: str column name for items
        :param col_rating: str column name for ratings
        :param clusters: int number of clusters to build
        :return: None
        """
        BaseRecommender.fit(self, train_df, col_user, col_item, col_rating)
        self.starspace_train_df = starspace_df

        try:
            vectors = pd.read_csv('pagespace.tsv', sep='\t', header=None)

        except FileNotFoundError:
            string_orders = [[str(product) for product in orders] for orders in self.starspace_train_df[self.col_item].values]
            string_orders = [' '.join(order) for order in string_orders]
            user_labels = [' __label__' + str(user_id) for user_id in self.starspace_train_df[self.col_user].values]
            orders_to_save = [string_orders[i] + user_labels[i] for i in range(len(string_orders))]

            with open('input.txt', 'w') as in_file:
                for item in orders_to_save:
                    in_file.write(item + '\n')

            del string_orders
            del user_labels
            del orders_to_save

            subprocess.call(['starspace',
                             'train',
                             '-trainFile',
                             'input.txt',
                             '-model',
                             'pagespace',
                             '-trainMode',
                             '0'])

            vectors = pd.read_csv('pagespace.tsv', sep='\t', header=None)

        vectors.rename(columns={0: 'id'}, inplace=True)

        list_ind = []
        for qr in vectors.id.values:
            if re.search('__label__', str(qr)):
                list_ind.append(True)
            else:
                list_ind.append(False)

        user_vecs = vectors.copy().iloc[list_ind]
        user_vecs['id'] = user_vecs.id.apply(lambda x: int(re.sub('__label__', '', x)))
        user_vecs = user_vecs.sort_values(by='id', ascending=True)
        self.user_ids = user_vecs.id.values
        self.user_vectors = user_vecs.iloc[:, 1:].values
        self.num_clusters = clusters
        self.model = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_labels = self.model.fit_predict(self.user_vectors)
        self.clusters_df = pd.DataFrame.from_dict({self.col_user: self.user_ids, 'cluster_id': self.cluster_labels})
        self.recommenders = []
        for cluster in tqdm(range(self.num_clusters)):
            users = self.clusters_df.query("cluster_id == @cluster")['user_id'].values
            mpr = MostPopularRecommender()
            train_users = self.train_df.query(f'{self.col_user} in @users')
            mpr.fit(train_users, self.col_user, self.col_item, self.col_rating)
            self.recommenders.append(mpr)

    def predict(self, test_df, k=cfg.DEFAULT_K):
        """
        performs recommendations for test users in test set
        :param test_df: pandas DataFrame with users and truth values
        :param k: int number of items to recommend
        :return: pandas DataFrame with k recommendations for each user in test_df
        """
        results_frame = pd.DataFrame(columns=[self.col_user, self.col_item])
        for cluster in tqdm(range(self.num_clusters)):
            users = self.clusters_df.query("cluster_id == @cluster")['user_id'].values
            test_users = test_df.query(f'{self.col_user} in @users')
            pred = self.recommenders[cluster].predict(test_users, k=k)
            results_frame = results_frame.append(pred, ignore_index=True)

        return results_frame


def _change_frames(row):
    """
    helper function to perform transformation of read from drive dataframes
    :param row: str in format of '[a , b, ... , c]' where a,b, .. c are int
    :return: list of int
    """
    return [int(element) for element in row.strip('[]').split(', ')]


class KingOfTheHillRecommender:
    """
    Class performs the king of the hill ensemble
    by training the classifier for best model
    over the results on the validation set
    """

    def __init__(self, col_user=cfg.USER_COL, col_item=cfg.ITEM_COL):
        """
        reads the needed datasets from the drive
        :param col_user: str name for user column
        :param col_item: str name for items column
        """
        self.col_user = col_user
        self.col_item = col_item
        self.als_results = pd.read_csv(cfg.ALS_RESULTS, index_col=0)
        self.mpr_results = pd.read_csv(cfg.MPR_RESULTS, index_col=0)
        self.stratify_mpr_results = pd.read_csv(cfg.STRATIFY_MPR_RESULTS, index_col=0)
        self.svd_results = pd.read_csv(cfg.SVD_RESULTS, index_col=0)
        self.als_results[self.col_item] = self.als_results[self.col_item].apply(_change_frames)
        self.mpr_results[self.col_item] = self.mpr_results[self.col_item].apply(_change_frames)
        self.stratify_mpr_results[self.col_item] = self.stratify_mpr_results[self.col_item].apply(_change_frames)
        self.svd_results[self.col_item] = self.svd_results[self.col_item].apply(_change_frames)
        self.als_results.sort_values(by=self.col_user, inplace=True)
        self.mpr_results.sort_values(by=self.col_user, inplace=True)
        self.stratify_mpr_results.sort_values(by=self.col_user, inplace=True)
        self.svd_results.sort_values(by=self.col_user, inplace=True)

        self.mpr_train = pd.read_csv(cfg.MPR_TRAIN, index_col=0)
        self.stratify_mpr_train = pd.read_csv(cfg.STRATIFY_MPR_TRAIN, index_col=0)
        self.als_train = pd.read_csv(cfg.ALS_TRAIN, index_col=0)
        self.svd_train = pd.read_csv(cfg.SVD_TRAIN, index_col=0)
        self.mpr_train.sort_values(by=self.col_user, inplace=True)
        self.stratify_mpr_train.sort_values(by=self.col_user, inplace=True)
        self.als_train.sort_values(by=self.col_user, inplace=True)
        self.svd_train.sort_values(by=self.col_user, inplace=True)
        self.mpr_train[self.col_item] = self.mpr_train[self.col_item].apply(_change_frames)
        self.stratify_mpr_train[self.col_item] = self.stratify_mpr_train[self.col_item].apply(_change_frames)
        self.als_train[self.col_item] = self.als_train[self.col_item].apply(_change_frames)
        self.svd_train[self.col_item] = self.svd_train[self.col_item].apply(_change_frames)
        self.train_results_list = [self.mpr_train, self.svd_train, self.als_train, self.stratify_mpr_train]

        self.vectors = pd.read_csv(cfg.PATH_VECS, sep='\t', header=None)
        self.vectors.rename(columns={0: "id"}, inplace=True)
        list_ind = []
        for qr in self.vectors.id.values:
            if re.search('__label__', str(qr)):
                list_ind.append(True)
            else:
                list_ind.append(False)

        self.user_vecs = self.vectors.copy().iloc[list_ind]
        self.user_vecs['id'] = self.user_vecs.id.apply(lambda x: int(re.sub('__label__', '', x)))
        self.user_vecs = self.user_vecs.sort_values(by='id', ascending=True)
        self.user_ids = self.user_vecs.id.values
        self.user_vectors = self.user_vecs.iloc[:, 1:].values

        self.classifier = LGBMClassifier(objective='multiclass', num_leaves=16, learning_rate=0.01,
                                         n_estimators=1000)

        self.model_list = [self.mpr_results, self.svd_results, self.als_results, self.stratify_mpr_results]

    def fit(self, validation_df):
        validation_df.sort_values(by=self.col_user, inplace=True)
        truth = validation_df[self.col_item].values
        evaluations = []
        for i in tqdm(range(validation_df.shape[0])):
            max_metric = 0
            for j, model in enumerate(self.train_results_list):
                evaluation = average_precision_at_k(truth[i], model[self.col_item].values[i])
                if evaluation > max_metric:
                    max_metric = j
            evaluations.append(max_metric)
        X_train, X_valid, y_train, y_valid = train_test_split(self.user_vectors,
                                                              evaluations,
                                                              test_size=0.1,
                                                              stratify=evaluations)
        self.classifier.fit(X_train, y_train, verbose=4, eval_set=(X_valid, y_valid), early_stopping_rounds=5)

    def predict(self, test_df):
        test_vecs = self.user_vecs.merge(test_df, right_on=self.col_user, left_on='id').sort_values(by='id')
        test_vectors = test_vecs.drop(columns=['id', self.col_user, self.col_item]).values
        test_users = test_vecs.reset_index()['id'].values
        prediction = self.classifier.predict(test_vectors)
        results = []
        for i, pred in enumerate(prediction):
            results.append(
                {self.col_user: test_users[i], self.col_item: self.model_list[pred][self.col_item].values[i]})
        return pd.DataFrame.from_records(results)
