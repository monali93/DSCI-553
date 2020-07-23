import itertools
from pyspark import SparkContext
import math
import time
import sys


def frequent_itemsets_function(baskets, candidate_itemsets):
    baskets = list(baskets)
    set_frequencies = []

    for c in list(candidate_itemsets):
        support = 0
        for b in baskets:
            if set(c).issubset(set(b)):
                support += 1
        if support > 0:
            set_frequencies.append((c, support))
    return set_frequencies


def filter_basket(basket, frequent_singleton):
    return sorted(list(set(basket).intersection(frequent_singleton)))


def frequent_in_basket(setslist, baskts, supportthreshold):
    frequent_list = []
    for s in setslist:
        support = 0
        for b in baskts:
            if set(s).issubset(set(b)):
                support += 1
        if support >= supportthreshold:
            frequent_list.append(tuple(s))
    return frequent_list


def generate_set(setsize, cand):
    candidate_list = []
    for i in range(len(cand) - 1):
        for j in range(i + 1, len(cand)):
            if cand[i][0:setsize - 2] == cand[j][0:setsize - 2]:
                candidate = list(set(cand[i]) | set(cand[j]))
                candidate.sort()
                if candidate not in cand:
                    candidate_list.append(candidate)
            else:
                break
    return candidate_list


def apriori(baskets, distinct_single, basket_count, support_threshold):
    baskets = list(baskets)
    sup_threshold = math.ceil(support_threshold * len(baskets) / basket_count)

    set_size = 1
    all_size_sets = []
    candidate_items = []

    for d in distinct_single:
        support = 0
        for b in baskets:
            if d in b:
                support += 1
        if support >= sup_threshold:
            candidate_items.append(d)
    all_size_sets.extend(zip(candidate_items))

    basket = []
    for bask in baskets:
        basket.append(filter_basket(bask, candidate_items))

    set_size += 1

    pairs_items = list(itertools.combinations(sorted(candidate_items), 2))
    pairs_freq = frequent_in_basket(pairs_items, basket, sup_threshold)

    set_size += 1
    temp_set = pairs_freq

    while len(temp_set) != 0:
        all_size_sets.extend(temp_set)
        candidates = generate_set(set_size, temp_set)
        frequents = frequent_in_basket(candidates, basket, sup_threshold)
        temp_set = frequents
        set_size += 1

    return all_size_sets


def main():
    if len(sys.argv) != 5:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()
    sc = SparkContext('local[*]', 'task2')
    csv_rdd = sc.textFile(str(sys.argv[3]))
    limit = int(sys.argv[1])

    basket = csv_rdd.filter(lambda x: 'user_id,business_id' not in x) \
        .map(lambda x: (x.split(","))).map(lambda x: (x[0], x[1])).distinct().map(lambda a: (a[0], [a[1]])) \
        .reduceByKey(lambda a, b: a + b).map(lambda x: set(x[1])) \
        .filter(lambda x: len(x) > limit).persist()

    basket_count = basket.count()
    distinct_single = sorted(basket.flatMap(lambda x: x).distinct().collect())

    support_threshold = int(sys.argv[2])

    candidate_itemsets = basket.mapPartitions(lambda x: (
        apriori(x, distinct_single, basket_count, support_threshold))) \
        .map(lambda x: (tuple(x), 1)).reduceByKey(lambda x, y: x + y).keys().collect()

    frequent_itemsets = basket.mapPartitions(
        lambda x: frequent_itemsets_function(x, candidate_itemsets)).reduceByKey(lambda x, y: (x + y)).filter(
        lambda x: x[1] >= support_threshold).keys().collect()

    sc.stop()

    f = open(str(sys.argv[4]), 'w')
    f.write("Candidates:")
    f.write("\n")

    sorted_candidate_itemSet = managingItemSets(candidate_itemsets, distinct_single)
    sorted_frquent_itemSet = managingItemSets(frequent_itemsets, distinct_single)

    for items in sorted_candidate_itemSet:
        s = ""
        for item in items:
            s = s + str(item)
        s = s.replace("[", "(").replace("]", "),")
        if len(items):
            f.write(s.rstrip(","))
            f.write('\n\n')
    f.write("Frequent Itemsets:")
    f.write("\n")
    for items in sorted_frquent_itemSet:
        s = ""
        for item in items:
            s = s + str(item)

        s = s.replace("[", "(").replace("]", "),")
        if len(items):
            f.write(s.rstrip(","))
            f.write('\n\n')
    f.close()

    end = time.time()
    print("Duration: " + str(end - start))


def managingItemSets(itemSets, distinct_single):
    final_list = []
    l = 1
    while l != len(distinct_single):
        list_temp = []
        for b in itemSets:
            if len(b) == l:
                list_temp.append(sorted(b))
        final_list.append(sorted(list_temp))
        l = l + 1
    return final_list


if __name__ == "__main__": main()
