import json
import re
from pyspark import SparkContext
import numpy as np
from scipy import stats
import sys


def func3(d):
    new_d = {}
    if 'reviewText' in d:
        new_d['overall'] = d['overall']
        new_d['verified'] = d['verified']
        new_d['reviewText'] = d['reviewText']
    return new_d


def func4(d):
    words = []
    line = []
    if 'reviewText' in d:
        line = d['reviewText'].split()
    for word in line:
        if re.match(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', word):
            words.append(word.lower())
    return words

# def func4(d):
#     words = []
#     line = ""
#     if 'reviewText' in d:
#         line = d['reviewText']
#         words = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', line.lower())
#     return words


def func7(d):
    final_tuple_each_rev = ()
    words = []
    if 'reviewText' in d:
        words = func4(d)
    dict_words = dict(brd_cast_var.value)
    new = {k: 0 for k, v in dict_words.items()}
    total_words_in_review = len(words)
    # if 'overall' in d:
    #     rating = d['overall']
    # rating = d['overall']

    if len(words) > 0:
        rating = d['overall']
        if 'verified' in d and d['verified'] is True:
            verified = 1
        else:
            verified = 0

        for w in words:
            if w in dict_words.keys():
                new[w] += 1
        final_tuple_each_rev = ((k, (v / total_words_in_review, rating, verified)) for k, v in new.items())
    return final_tuple_each_rev


def func8(d):
    X = [v[0] for v in d[1]]
    y = [v[1] for v in d[1]]
    X = np.array(X).reshape((-1, 1))
    y = np.array(y)

    X_mean = np.mean(X)
    X_std = np.std(X)
    X = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    X_std_mean = np.mean(X)
    y_std_mean = np.mean(y)

    betas = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    beta0 = y_std_mean - X_std_mean * betas[0]
    betas = np.array([beta0, betas[0]])
    y_pred = beta0 + betas[1] * X
    y_pred = y_pred.transpose()
    rss = np.sum((y_pred - y) ** 2)
    m = 1
    dof = len(y) - (m + 1)
    s_sqr = rss / dof
    var = np.sum((X - X_std_mean) ** 2)
    denr = np.sqrt(s_sqr / var)
    t_stats = betas[1] / denr
    plt_beta = stats.t.sf(abs(t_stats), df=dof)
    # if plt_beta > 0.5:
    #     p_val = 2 * (1 - plt_beta)
    # else:
    #     p_val = 2 * plt_beta
    p_val = 2 * (plt_beta) * 1000
    tup = tuple((d[0], betas[1], p_val))
    return tup


def func9(d):
    z = np.array(d[1])
    X = z[:, (0, 2)]
    y = z[:, 1]
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    X_std_mean = np.mean(X)
    y_std_mean = np.mean(y)
    betas = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    beta0 = y_std_mean - np.sum((np.multiply(X_std_mean, betas)))
    y_pred = beta0 + (betas.dot(X.transpose()))
    betas = np.concatenate(([beta0], betas), 0)

    y_pred = y_pred.transpose()
    rss = np.sum((y_pred - y) ** 2)
    m = 2
    dof = len(y) - (m + 1)
    s_sqr = rss / dof
    my_mean = np.mean(X[:, 0])
    var = np.sum(((X[:,0]- my_mean)) ** 2)

    denr = np.sqrt(s_sqr / var)
    t_stats = betas[1] / denr
    plt_beta = stats.t.sf(abs(t_stats), df=dof)
    # if plt_beta > 0.05:
    p_val = 2 * (plt_beta) * 1000
    # else:
    #     p_val = 2 * plt_beta
    tup = tuple((d[0], betas[1], p_val))
    return tup


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Incorrect number of args")
    #     sys.exit(-1)
    input_file = sys.argv[1]
    # print("Path---",input_file)
    # input_file = r"C:\Users\Yojana\Desktop\Sem1\BigData\Assignment3\Software_5.json.gz"
    sc = SparkContext('local', 'First App')

rdd = sc.textFile(input_file)
rdd1 = rdd.map(lambda line: json.loads(line))
rdd2 = rdd1.map(lambda d: func3(d))
rdd3 = rdd2.flatMap(lambda d: func4(d))
rdd4 = rdd3.map(lambda word: (word, 1))
rdd5 = rdd4.reduceByKey(lambda v1, v2: v1 + v2)
rdd6 = rdd5.sortBy(lambda x: x[1], False)
brd_cast_var = sc.broadcast(rdd6.take(1000))
rdd8 = rdd2.flatMap(lambda d: func7(d))
rdd9 = rdd8.groupByKey().mapValues(list)
rdd10 = rdd9.map(lambda d: func8(d))
rdd11 = rdd10.sortBy(lambda x: x[1], False)
print("The top 20 word positively correlated with rating")
print(rdd11.take(20))
rdd12 = rdd10.sortBy(lambda x: x[1], True)
print("The top 20 word negatively correlated  with rating")
print(rdd12.take(20))
rdd13 = rdd9.map(lambda d: func9(d))
rdd14 = rdd13.sortBy(lambda x: x[1], False)
print("The top 20 words positively related to rating, controlling for verified")
print(rdd14.take(20))
rdd15 = rdd13.sortBy(lambda x: x[1], True)
print("The top 20 words negatively related to rating, controlling for verified")
print(rdd15.take(20))
