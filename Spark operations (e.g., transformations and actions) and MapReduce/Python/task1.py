from pyspark import SparkContext
import json
import re
import sys


def sub_tasks(review_rdd, output_file, stopwords, year, m, n):
    total_reviews = review_rdd.map(lambda x: x['review_id']).count()

    y_reviews = review_rdd.filter(lambda x: str(year) in x['date']).count()

    distinct_users = review_rdd.map(lambda x: x['user_id']).distinct().count()

    top_reviewers = review_rdd.map(lambda x: (x['user_id'], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(int(m))

    pun_list = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")"]

    stopwords_list = []
    with open(str(stopwords), 'rt') as stopwords_file:
        for word in stopwords_file:
            stopwords_list.append(word.strip())

    top_words = review_rdd.flatMap(lambda x: x['text'].lower().strip().split()) \
        .filter(lambda x: x not in pun_list) \
        .map(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x)) \
        .filter(lambda x: x not in stopwords_list and len(x) > 0) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: (-x[1], x[0])).map(lambda x: x[0]).take(int(n))

    output = {
        "A": total_reviews,
        "B": y_reviews,
        "C": distinct_users,
        "D": top_reviewers,
        "E": top_words,
    }

    with open(str(output_file), "w+") as fileout:
        json.dump(output, fileout, indent=4)


def main():
    if len(sys.argv) != 7:
        print("Please enter all the required agruments")
        exit(-1)

    sc = SparkContext('local[*]', 'task1')

    review_rdd = sc.textFile(str(sys.argv[1])).map(lambda x: json.loads(x)).cache()

    sub_tasks(review_rdd, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

    sc.stop()


if __name__ == "__main__": main()
