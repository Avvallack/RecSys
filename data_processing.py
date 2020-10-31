import pandas as pd
import requests
from zipfile import ZipFile
import io

from config import *


FILE_NAMES = [ORDERS, TRAIN_PRODUCTS_ORDERS, TEST_PRODUCTS_ORDERS]


def load_data(path=DATA_PATH):
    """
    Load dataset from web zip archive and stores it into memory
    :param path: path to zip archive with data
    :return: ZipFile object
    """
    resp = requests.get(path)
    stringIO = io.BytesIO(resp.content)
    return ZipFile(stringIO)


def raw_set_preparation(zipfile, file_names=FILE_NAMES):
    raw_orders = pd.read_csv(zipfile.open(file_names[0]))
    raw_order_product_train = pd.read_csv(zipfile.open(file_names[1]))
    raw_order_product_test = pd.read_csv(zipfile.open(file_names[2]))
    return raw_orders, raw_order_product_train, raw_order_product_test


def train_test_orders_split(orders):
    train_orders = orders.query("eval_set == 'prior'").copy()
    test_orders = orders.query("eval_set == 'train'").copy()
    return train_orders, test_orders


def train_test_preparation(orders, train_products, test_products):
    train_orders, test_orders = train_test_orders_split(orders)
    train_set = train_orders.merge(train_products)[['user_id', 'product_id', 'add_to_cart_order']]
    train_set_ratings = train_set.groupby(by=['user_id', 'product_id']).count().reset_index()
    train_set_ratings.rename(columns={'add_to_cart_order': DEFAULT_RATING_COL}, inplace=True)
    test_set = test_orders.merge(test_products)[['user_id', 'product_id']]
    test_set_prepared = test_set.groupby('user_id').aggregate(lambda x: list(x)).reset_index()
    return train_set_ratings, test_set_prepared


def starspace_preparation(orders, train_products):
    train_vecs = orders.merge(train_products)
    train_vecs = train_vecs[['order_id', 'product_id', 'user_id']]
    vectors = train_vecs.groupby(by=['user_id', 'order_id']).aggregate(lambda x: list(x)).reset_index()
    return vectors


def train_validation_split(orders, train_products):
    train_orders, _ = train_test_orders_split(orders)
    train_set = train_orders.merge(train_products)[['user_id', 'product_id', 'order_id']]
    valid_orders = train_set.groupby('user_id').agg('max')['order_id'].values
    valid_set = train_set.query('order_id in @valid_orders').groupby(by=['user_id']).agg(lambda x: list(x))
    valid_set = valid_set.reset_index()[['user_id', 'product_id']]
    train_set_ratings = train_set.query("order_id not in @valid_orders").groupby(by=['user_id', 'product_id']).count()
    train_set_ratings = train_set_ratings.reset_index()
    train_set_ratings.rename(columns={'order_id': 'rating'}, inplace=True)
    return train_set_ratings, valid_set
