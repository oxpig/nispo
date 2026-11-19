"""Exact additive, lexicographic path maxima in a tree or forest.

The caller proves the chemical scope. This operation has no naming heuristics,
path cap, or preferred-name rules. It retains every tied maximum, in the same
endpoint and path order as exhaustive leaf-pair enumeration.
"""
from itertools import combinations, product


def maximum_weight_leaf_paths(adjacency, node_weights, edge_weights, *, minimum_rank=None):
    """Score paths by dynamic programming, then expand only maximal ties.

    Weights are equally sized integer tuples, added componentwise and compared
    lexicographically. Internal vertices are not eligible endpoints. A forest
    singleton is ignored, matching the existing leaf-pair enumerator, unless
    it is the entire supplied graph.
    """
    if not adjacency:
        return None, ()
    if len(adjacency) == 1:
        node = next(iter(adjacency))
        rank = node_weights[node]
        return rank, ((node,),) if minimum_rank is None or rank >= minimum_rank else ()

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    parent, order = {}, []
    for root in sorted(adjacency):
        if root in parent:
            continue
        parent[root] = None
        queue = [root]
        for node in queue:
            order.append(node)
            for child in adjacency[node]:
                if child != parent[node]:
                    if child in parent:
                        raise ValueError('maximum_weight_leaf_paths requires a forest')
                    parent[child] = node
                    queue.append(child)

    down, choices = {}, {}
    maximum, forks = None, []
    for node in reversed(order):
        weight = node_weights[node]
        arms = []
        if len(adjacency[node]) <= 1:
            arms.append((weight, None))
        for child in adjacency[node]:
            if parent.get(child) == node:
                edge = edge_weights[min(node, child), max(node, child)]
                arms.append((add(weight, add(edge, down[child])), child))
        best = max(rank for rank, _ in arms)
        first = tuple(child for rank, child in arms if rank == best)
        down[node], choices[node] = best, first
        if len(arms) < 2:
            continue
        if len(first) > 1:
            second_rank, second = best, None
        else:
            second_rank = max(rank for rank, _ in arms if rank != best)
            second = tuple(child for rank, child in arms if rank == second_rank)
        rank = tuple(a + b - c for a, b, c in zip(best, second_rank, weight))
        if maximum is None or rank > maximum:
            maximum, forks = rank, []
        if rank == maximum:
            # Keep a factored tie, not the potentially quadratic arm pairs.
            forks.append((node, first, second))

    if maximum is None or minimum_rank is not None and maximum < minimum_rank:
        return maximum, ()

    def endpoints(node, child):
        if child is None:
            yield node
            return
        stack = [child]
        while stack:
            current = stack.pop()
            for choice in choices[current]:
                if choice is None:
                    yield current
                else:
                    stack.append(choice)

    def arm_path(endpoint, junction):
        path = []
        while endpoint != junction:
            path.append(endpoint)
            endpoint = parent[endpoint]
        return path

    paths = []
    for node, first, second in forks:
        pairs = combinations(first, 2) if second is None else product(first, second)
        for left, right in pairs:
            for start, end in product(endpoints(node, left), endpoints(node, right)):
                path = arm_path(start, node) + [node] + arm_path(end, node)[::-1]
                paths.append(tuple(path if start < end else path[::-1]))
    return maximum, tuple(sorted(paths))
