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


def frequent_in_basket(setslist, baskts, supportthreshold):
    frequent_list = []
    for s in setslist:
        support = 0
        for b in baskts:
            if set(s).issubset(set(b)):
                support += 1
        if support >= supportthreshold:
            frequent_list.append(s)

    return frequent_list


def generate_set(setsize, cand):
    candidate_list = []
    for x in itertools.combinations(cand, 2):
        union = set(x[0]).union(set(x[1]))
        if len(union) == setsize:
            if union not in candidate_list:
                candidate_list.append(union)
    return candidate_list


def apriori(basket, distinct_single, basket_count, support_threshold):

    basket = list(basket)

    sup_threshold = math.ceil(support_threshold * len(basket) / basket_count)

    set_size = 1
    all_size_sets = []

    candidate_items = []
    for d in distinct_single:
        support = 0
        for b in basket:
            if d in b:
                support += 1
        if support >= sup_threshold:
            candidate_items.append((d,))
    all_size_sets.extend(candidate_items)

    set_size += 1
    temp_set = candidate_items


    while len(temp_set) != 0:
        candidates = generate_set(set_size, temp_set)
        frequents = frequent_in_basket(candidates, basket, sup_threshold)
        temp_set = frequents
        set_size += 1
        if len(temp_set):
            all_size_sets.extend(temp_set)
    return all_size_sets


def case(csvrdd):
    if int(sys.argv[1]) == 1:
        basket = csvrdd.filter(lambda x: 'user_id,business_id' not in x) \
            .map(lambda x: (x.split(",")[0], x.split(",")[1])).distinct().groupByKey().values().persist()
        return basket

    elif int(sys.argv[1]) == 2:
        basket = csvrdd.filter(lambda x: 'user_id,business_id' not in x) \
            .map(lambda x: (x.split(",")[1], x.split(",")[0])).distinct().groupByKey().values().persist()
        return basket
    print("Invalid Argument")
    return None


def main():
    if len(sys.argv) != 5:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()
    sc = SparkContext('local[*]', 'task1')
    csv_rdd = sc.textFile(str(sys.argv[3]))

    basket = case(csv_rdd)
    basket_count = basket.count()
    distinct_single = basket.flatMap(lambda x: x).distinct().collect()

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
