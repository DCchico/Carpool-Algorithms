import networkx as nx
import sys
import copy
import random
import matplotlib.pyplot as plt


def create_hood(graph, idx, hood_size):
    hood = dict()
    for i in range(idx, hood_size + idx):
        for j in range(i + 1, idx + hood_size):
            try:
                overhead1 = nx.dijkstra_path_length(graph, source=i, target=j, weight='weight')
                overhead2 = nx.dijkstra_path_length(graph, source=graph.nodes[i]['dst'], target=graph.nodes[j]['dst'],
                                                    weight='weight')
            except nx.NetworkXNoPath:
                total_cost_c2 = graph.nodes[i]['cost'] + graph.nodes[i]['cost']
                route_c2 = (i, j)
                hood[route_c2] = total_cost_c2
                continue
            overhead = overhead1 + overhead2
            raw_cost = graph.nodes[i]['cost'] + graph.nodes[j]['cost']
            if raw_cost > overhead + graph.nodes[i]['cost']:
                if graph.nodes[j]['cost'] > graph.nodes[i]['cost']:
                    total_cost_c2 = graph.nodes[i]['cost'] + overhead
                    route_c2 = (j, i)
                elif graph.nodes[j]['cost'] <= graph.nodes[i]['cost']:
                    total_cost_c2 = graph.nodes[j]['cost'] + overhead
                    route_c2 = (i, j)
            else:
                total_cost_c2 = raw_cost
                route_c2 = (i, j)
            hood[route_c2] = total_cost_c2
    return hood


def heapPermutation(a, size, n, result):
    if size == 1:
        result.append([i for i in a])
        return
    for i in range(size):
        heapPermutation(a, size - 1, n, result)
        if size & 1:
            a[0], a[size - 1] = a[size - 1], a[0]
        else:
            a[i], a[size - 1] = a[size - 1], a[i]


def opt_combine(hood, hood_size, n):
    result = []
    opt_cost_c2 = sys.maxsize
    heapPermutation([i for i in range(n, hood_size + n)], hood_size, hood_size, result)
    for p in result:
        cost_c2 = 0
        for i in range(0, hood_size, 2):
            if (p[i], p[i + 1]) not in hood:
                cost_c2 = sys.maxsize
                break
            else:
                cost_c2 += hood[(p[i], p[i + 1])]
        if opt_cost_c2 > cost_c2:
            opt_cost_c2 = cost_c2
    return opt_cost_c2


# Mathematically optimized solution to iterate through all combinations
# This method also uses the hood dictionary provided by function: create_hood
# Complexity: (N-1)(N-3)......(3)(1) = O(N^(hood_size/2))
def opt_calc(hood, traverse_l, accum_cost=0):
    if not traverse_l:
        global opt_cost
        if accum_cost < opt_cost:
            opt_cost = accum_cost
    for i in range(1, len(traverse_l)):
        if (traverse_l[0], traverse_l[i]) in hood:
            cost_opt = hood[(traverse_l[0], traverse_l[i])]
        else:
            cost_opt = hood[(traverse_l[i], traverse_l[0])]
        traverse_l_cp = copy.deepcopy(traverse_l)
        traverse_l_cp.pop(0)
        traverse_l_cp.pop(i - 1)
        opt_calc(hood, traverse_l_cp, accum_cost + cost_opt)


# Semi-iterative algorithm: iterate through each pair a time (aka ain't check different combinations)
# Complexity: (N-1) + (N-3) + ......+ 3 + 1 = O(N^2 / 2)
def semi_opt_calc(hood, traverse_l, accum_cost=0):
    if not traverse_l:
        return accum_cost
    cost_opt = sys.maxsize
    for i in range(1, len(traverse_l)):
        if (traverse_l[0], traverse_l[i]) in hood:
            cost_opt_t = hood[(traverse_l[0], traverse_l[i])]
        else:
            cost_opt_t = hood[(traverse_l[i], traverse_l[0])]
        if cost_opt_t < cost_opt:
            cost_opt = cost_opt_t
            idx = i
    traverse_l.pop(0)
    traverse_l.pop(idx - 1)
    return semi_opt_calc(hood, traverse_l, accum_cost + cost_opt)


# ------------------------------------------------------------------------------------------------------------------
# Genetic-based Algorithms
# Five steps:
# 1. Initialization: create population
# 2. Find smallest half of nodes as parents
# 3. Reproduce n children by exchanging half of the chromosomes
# 4. Delete n of last generation
# 5. Optional mutation phase
def gene_init(hood, hood_sz):
    population = dict()
    chrom = list()
    while len(population) < hood_sz:
        cst = 0
        while 1:
            x, y = random.choice(list(hood.keys()))
            if not (any(x in pair for pair in chrom) or any(y in pair for pair in chrom)):
                chrom.append((x, y))
                cst += hood[(x, y)]
            if len(chrom) == hood_sz / 2:
                break
        population[tuple(chrom)] = cst
        del chrom[:]
    return population


def next_gen(hood, population, hood_sz):
    selected_gen_sz = (int)(hood_sz / 2)
    ex_gene = hood_sz / 4
    new_chroms = list()
    idx = 0
    for chrom in sorted(population, key=population.get):
        if idx < selected_gen_sz and idx % 2 == 0:
            chrom1 = chrom
        elif idx < selected_gen_sz and idx % 2 != 0:
            chrom2 = chrom
            new_c1 = list()
            new_c2 = list()
            for i in range(selected_gen_sz):
                if i < ex_gene:
                    new_c1.append(chrom2[i])
                    new_c2.append(chrom1[i])
                else:
                    if not (any(chrom1[i][0] in pair for pair in new_c1) or any(chrom1[i][1] in pair for pair in new_c1)):
                        new_c1.append(chrom1[i])
                    else:
                        while 1:
                            x, y = random.choice(list(hood.keys()))
                            if not (any(x in pair for pair in new_c1) or any(y in pair for pair in new_c1)):
                                new_c1.append((x, y))
                                break
                    if not (any(chrom2[i][0] in pair for pair in new_c2) or any(chrom2[i][1] in pair for pair in new_c2)):
                        new_c2.append(chrom2[i])
                    else:
                        while 1:
                            x, y = random.choice(list(hood.keys()))
                            if not (any(x in pair for pair in new_c2) or any(y in pair for pair in new_c2)):
                                new_c2.append((x, y))
                                break
            new_chroms.append(new_c1)
            new_chroms.append(new_c2)
        elif idx > (int)(selected_gen_sz / 2) * 2:
            del population[chrom]
        idx += 1
    for new_chrom in new_chroms:
        new_chrom = tuple(new_chrom)
        if new_chrom not in population:
            cst = 0
            for x, y in new_chrom:
                if (x, y) in hood:
                    cst += hood[(x, y)]
                else:
                    cst += hood[(y, x)]
            population[tuple(new_chrom)] = cst
    return population


def gene_based_algo(hood, hood_sz):
    population = gene_init(hood, hood_sz)
    last_min = min(population.values())
    freeze = 0
    while 1:
        print(population)
        max_cost = max(population.values())
        min_cost = min(population.values())
        if max_cost == min_cost or freeze == 5:
            return population
        elif last_min == min_cost:
            freeze += 1
        last_min = min(population.values())
        population = next_gen(hood, population, hood_sz)


# --------------------------------------------------------------------------------------------------
# simple carpool: nearest neighbors carpool together
def carpool_2(graph, idx):
    try:
        overhead1 = nx.dijkstra_path_length(graph, source=idx, target=idx + 1, weight='weight')
        overhead2 = nx.dijkstra_path_length(graph, source=graph.nodes[idx]['dst'], target=graph.nodes[idx + 1]['dst'],
                                            weight='weight')
    except nx.NetworkXNoPath:
        total_cost_c2 = graph.nodes[idx]['cost'] + graph.nodes[idx + 1]['cost']
        route_c2 = []
        return total_cost_c2, route_c2
    if graph.nodes[idx + 1]['cost'] > graph.nodes[idx]['cost']:
        total_cost_c2 = graph.nodes[idx]['cost'] + overhead1 + overhead2
        route_c2 = [idx + 1, idx, graph.nodes[idx]['dst'], graph.nodes[idx + 1]['dst']]
    else:
        total_cost_c2 = graph.nodes[idx + 1]['cost'] + overhead1 + overhead2
        route_c2 = [idx, idx + 1, graph.nodes[idx + 1]['dst'], graph.nodes[idx]['dst']]
    return total_cost_c2, route_c2
# -----------------------------------------------------------------------------------------------------


# Main----------------------------------------------------------------------------------
file = open("road-chesapeake.txt", 'r')
# file = open("small_example.txt", 'r')
G = nx.Graph()
# Note that the sparsity should be greater than hood_size
sparsity = 15
hood_size = 8

# Generate Graph for the road network
for line in file:
    (u, v) = line.split()
    u = int(u)
    v = int(v)
    G.add_edge(u, v, weight=abs(u-v))

# Simulate src and dst for nodes
cluster = 0
wait = 0
for n in sorted(list(G)):
    if wait == (sparsity + hood_size):
        wait = 0
        cluster = 0
    if cluster == hood_size:
        wait += 1
        continue
    if n + sparsity + hood_size - 1 - cluster in G:
        cluster = cluster + 1
        G.nodes[n]['dst'] = n + sparsity

# print edge information
'''
for u, v, weight in G.edges.data('weight'):
    if weight is not None:
        print("[{}, {}] = {}".format(u, v, weight))
'''

nx.draw(G)
plt.show()

# print node information
print(sorted(list(G.nodes(data=True))))
total = 0
for n in G:
    if 'dst' in G.nodes[n]:
        try:
            cost = nx.dijkstra_path_length(G, source=n, target=G.nodes[n]['dst'], weight='weight')
        except nx.NetworkXNoPath:
            G.nodes[n]['dst'] = list(G.neighbors(n))[0]
            cost = nx.dijkstra_path_length(G, source=n, target=G.nodes[n]['dst'], weight='weight')
        total += cost
        G.nodes[n]['cost'] = cost
        print("cost [{}, {}] = {}".format(n, G.nodes[n]['dst'], G.nodes[n]['cost']))
print("Sum of the regular total cost is {}".format(total))


total = 0
carpool_size = 2
wait = 0
for n in sorted(list(G)):
    if wait == (sparsity + 2 * hood_size):
        wait = 0
    elif wait > 0:
        wait += 1
        continue
    if 'dst' in G.nodes[n]:
        wait += 1
        group = int(hood_size / carpool_size)
        remain = hood_size - group * carpool_size
        for i in range(n, n + group * carpool_size, carpool_size):
            total_cost, route = carpool_2(G, i)
            total += total_cost
            # print("carpool cost = {} ".format(total_cost) + str(route))
        for i in range(n + group * carpool_size, n + group * carpool_size + remain):
            total_cost = G.nodes[i]['cost']
            total += total_cost
            # print("Regular cost = " + str(total_cost))
print("Sum of simple carpool cost = {}".format(total))

'''
total = 0
carpool_size = 2
wait = 0
for n in sorted(list(G)):
    if wait == (sparsity + 2 * hood_size):
        wait = 0
    elif wait > 0:
        wait += 1
        continue
    if 'dst' in G.nodes[n]:
        wait += 1
        group = int(hood_size / carpool_size) * carpool_size
        remain = hood_size - group * carpool_size
        total_cost = opt_combine(create_hood(G, n, hood_size), hood_size, n)
        total += total_cost
        for i in range(n + group * carpool_size, n + group * carpool_size + remain):
            total_cost = G.nodes[i]['cost']
            total += total_cost
            print("Regular cost = " + str(total_cost))
print("sum of iterative best carpool cost = {}".format(total))
'''

total = 0
carpool_size = 2
wait = 0
opt_cost = sys.maxsize
for n in sorted(list(G)):
    if wait == (sparsity + 2 * hood_size):
        wait = 0
    elif wait > 0:
        wait += 1
        continue
    if 'dst' in G.nodes[n]:
        wait += 1
        group = int(hood_size / carpool_size) * carpool_size
        remain = hood_size - group * carpool_size
        opt_calc(create_hood(G, n, hood_size), [i for i in range(n, n + hood_size)])
        total += opt_cost
        opt_cost = sys.maxsize
        for i in range(n + group * carpool_size, n + group * carpool_size + remain):
            total_cost = G.nodes[i]['cost']
            total += total_cost
            print("Regular cost = " + str(total_cost))
print("sum of ideal traverse carpool cost = {}".format(total))


total = 0
carpool_size = 2
wait = 0
for n in sorted(list(G)):
    if wait == (sparsity + 2 * hood_size):
        wait = 0
    elif wait > 0:
        wait += 1
        continue
    if 'dst' in G.nodes[n]:
        wait += 1
        group = int(hood_size / carpool_size) * carpool_size
        remain = hood_size - group * carpool_size
        opt_cost = semi_opt_calc(create_hood(G, n, hood_size), [i for i in range(n, n + hood_size)])
        total += opt_cost
        opt_cost = sys.maxsize
        for i in range(n + group * carpool_size, n + group * carpool_size + remain):
            total_cost = G.nodes[i]['cost']
            total += total_cost
            print("Regular cost = " + str(total_cost))
print("sum of semi-ideal traverse carpool cost = {}".format(total))


total = 0
carpool_size = 2
wait = 0
for n in sorted(list(G)):
    if wait == (sparsity + 2 * hood_size):
        wait = 0
    elif wait > 0:
        wait += 1
        continue
    if 'dst' in G.nodes[n]:
        wait += 1
        group = int(hood_size / carpool_size) * carpool_size
        remain = hood_size - group * carpool_size
        p = gene_based_algo(create_hood(G, n, hood_size), hood_size)
        total += min(p.values())
        for i in range(n + group * carpool_size, n + group * carpool_size + remain):
            total_cost = G.nodes[i]['cost']
            total += total_cost
            print("Regular cost = " + str(total_cost))
print("sum of gene-based algorithm carpool cost = {}".format(total))