import itertools
import sys
import time
from collections import defaultdict
import copy
import heapq

from pyspark import SparkContext


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


class Graph:
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.vertices = defaultdict(float)
        # self.level = defaultdict(int)
        # self.parent = defaultdict(list)
        # self.visited = []
        # self.queue = []
        self.new_graph = defaultdict(list)
        self.edge_btw_dict = defaultdict(float)
        self.all_parents = set()
        self.all_vertices = set()

    def create_adjacency_list(self, u, v):
        self.graph[u].append(v)
        return self.graph

    def add_vertices(self, v):
        self.vertices[v] = 1.0
        return self.vertices

    def modified_bfs(self, root, grp):
        queue = []
        visited = []
        level = defaultdict(int)
        parent = defaultdict(list)

        queue.append(root)
        visited.append(root)
        level[root] = 0
        t = [[], 1, 0]
        parent[root] = t
        while queue:
            s = queue.pop(0)
            adjacent_nodes = grp[s]
            for node in adjacent_nodes:
                if node not in visited:
                    queue.append(node)
                    visited.append(node)
                if level.get(node, None) is None or level.get(node) > level.get(s):
                    sp = parent[s][1]
                    # update the value if exist otherwise update it
                    pn = parent.get(node)
                    if pn is not None:
                        pn[0].append(s)
                        pn[1] += sp
                    else:
                        parent[node] = [[s], sp, level[s] + 1]
                        level[node] = level[parent[node][0][0]] + 1
        return parent

    def edge_betweeness(self, root, prt):
        vertices = copy.deepcopy(self.vertices)
        # edge_btw_dict = copy.deepcopy(self.edge_btw_dict)
        edge_btw_dict = dict()
        prt_list = [x[0] for x in list(prt.values())]
        all_parents = set(itertools.chain(*prt_list))
        all_vertices = set(list(prt.keys()))
        leaf_nodes = list(all_vertices.difference(all_parents))
        temp = [(-1 * prt[node][2], node) for node in leaf_nodes]
        myqueue = []
        for node in temp:
            heapq.heappush(myqueue, node)
        # print("\nmyqueue", myqueue)
        btw_queue = []
        btw_visited = []
        btw_queue.extend(leaf_nodes)
        btw_visited.extend(leaf_nodes)
        while myqueue:
            n = heapq.heappop(myqueue)[1]
            # print("node n", n)
            # print("\n\n vertices", vertices, "\n\nand n: ", n)
            parent_list = prt.get(n)[0]
            if parent_list is not None:
                len_parents = len(parent_list)
                for p in parent_list:
                    if p not in btw_visited:
                        heapq.heappush(myqueue, (-1 * prt[p][2], p))
                        btw_visited.append(p)
                    num = prt[p][1]
                    den = 0
                    for pat in parent_list:
                        den += prt[pat][1]
                    edge = sorted([n, p])
                    edge_btw_dict[tuple(edge)] = float(float(vertices[n]) * float(num / den))
                    vertices[p] = vertices[p] + edge_btw_dict[tuple(edge)]
        return edge_btw_dict


def remove_highest_betweness_edge(edge_betweeness_list, graph):
    edge = edge_betweeness_list[0][0]
    #print("edge1 : ", edge[1], "edge2 : ", edge[0], "modularity : ", edge_betweeness_list[0][1])
    graph[edge[0]].remove(edge[1])
    graph[edge[1]].remove(edge[0])
    return graph


def detect_community(a):
    community_list = []
    queue = []

    remaining_vertices = list(a.keys())
    while len(remaining_vertices) > 0:
        visited = []
        vertex = remaining_vertices[0]
        queue.append(vertex)
        visited.append(vertex)
        while queue:
            parent_node = queue.pop(0)
            for child_node in a[parent_node]:
                if child_node not in visited:
                    queue.append(child_node)
                    visited.append(child_node)
        visited.sort()
        community_list.append(visited)
        remaining_vertices = list(set(a.keys()).difference(set(itertools.chain(*community_list))))
    return community_list


def compute_modularity(communities, adjacency_list, g, total_edges_original):
    temp_sum = 0
    count = 0
    sum_track = []
    for community in communities:
        for edge in itertools.combinations(community, 2):
            if edge[1] in adjacency_list[edge[0]]:
                edge_in_orginal_graph = 1
                count += 1
            else:
                edge_in_orginal_graph = 0
            ki = len(g[edge[0]])
            kj = len(g[edge[1]])
            temp_sum += float(edge_in_orginal_graph - (ki * kj / (2 * total_edges_original)))
    return communities, float(temp_sum / (2 * total_edges_original))


def communities_formation(total_edge_betweeness_list, graph, adjacency_list, org_edges):
    edge_betweeness = total_edge_betweeness_list

    g = remove_highest_betweness_edge(total_edge_betweeness_list, graph)

    communities = detect_community(g)
    # print("communities : ", communities)

    total_edges_original = len(total_edge_betweeness_list)

    communities, modularity = compute_modularity(communities, adjacency_list, g, org_edges)

    return communities, modularity


def main():
    global adjacency_list

    threshhold = int(sys.argv[1])
    csv_file = str(sys.argv[2])
    out_file = str(sys.argv[3])
    out_file1 = str(sys.argv[4])

    start = time.time()
    sc = SparkContext('local[*]', 'task2')
    csv_rdd = sc.textFile(csv_file)

    usr_bus_dict = csv_rdd.filter(lambda x: 'user_id,business_id' not in x) \
        .map(lambda x: x.split(',')) \
        .combineByKey(to_list, append, extend).mapValues(lambda x: sorted(x)).collectAsMap()

    edges = [tuple(pair) for pair in itertools.permutations(sorted(usr_bus_dict.keys()), 2) if
             len(set(usr_bus_dict[pair[0]]) & set(usr_bus_dict[pair[1]])) >= threshhold]

    # print(edges)

    vertices = list(set(itertools.chain(*edges)))

    g = Graph()

    for edge in edges:
        adjacency_list = g.create_adjacency_list(edge[0], edge[1])

    graph = adjacency_list

    # print("no of vertices derived from edges", len(a))

    for vertex in vertices:
        v = g.add_vertices(vertex)
    # print("no of vertices", len(v))

    total_edge_betweeness_dict = defaultdict(float)
    for i in vertices:
        p = g.modified_bfs(i, graph)
        d = g.edge_betweeness(i, p)
        for k, v in d.items():
            total_edge_betweeness_dict[k] += v

    # print("final dictionary", total_edge_betweeness_dict)
    total_edge_betweeness_dict = {k: float(v / 2) for k, v in total_edge_betweeness_dict.items()}
    total_edge_betweeness_list = sorted(total_edge_betweeness_dict.items(), key=lambda x: (-x[1], x[0][0]))
    original_edges = len(total_edge_betweeness_dict)

    with open(out_file, 'w+') as fileout:
        for c in total_edge_betweeness_list:
            fileout.writelines(str(c[0]) + ", " + str(c[1]) + "\n")

    def recompute_betweeness(communities, total_edge_betweeness_dict, vertices):
        total_edge_betweeness_dict = defaultdict(float)
        for v in vertices:
            p = g.modified_bfs(v, graph)
            d = g.edge_betweeness(v, p)
            for k, v in d.items():
                total_edge_betweeness_dict[k] += v
        total_edge_betweeness_dict = {k: float(v / 2) for k, v in total_edge_betweeness_dict.items()}
        total_edge_betweeness_list = sorted(total_edge_betweeness_dict.items(), key=lambda x: (-x[1], x[0][0]))
        return total_edge_betweeness_list

    adj = copy.deepcopy(adjacency_list)
    communities, modularity = communities_formation(total_edge_betweeness_list, graph, adjacency_list, original_edges)
    total_edge_betweeness_list = recompute_betweeness(communities, total_edge_betweeness_dict, vertices)
    max_modularity = float("-inf")
    perfect_communities = None

    while 1:
        if max_modularity < modularity:
            max_modularity = modularity
            perfect_communities = communities
        else:
            break
        communities, modularity = communities_formation(total_edge_betweeness_list, graph, adj,
                                                        original_edges)
        total_edge_betweeness_list = recompute_betweeness(communities, total_edge_betweeness_dict, vertices)
        #print("recompute betweenness : ", total_edge_betweeness_list)

    perfect_communities = sorted(perfect_communities, key=lambda x: (len(x), x[0]))

    #sorted(perfect_communities, key=lambda x: len(x))

    with open(out_file1, 'w+') as fileout:
        for c in perfect_communities:
            fileout.writelines(str(c)[1:-1] + "\n")

    end = time.time()
    #print("maximum modularity : ", max_modularity)
    print("Duration: " + str(end - start))


if __name__ == "__main__": main()
