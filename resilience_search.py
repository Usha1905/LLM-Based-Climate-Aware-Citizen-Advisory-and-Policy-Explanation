#!/usr/bin/env python
# coding: utf-8

# In[5]:


# module2_resilience_search_ucs_only_small.py
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random
from math import ceil

random.seed(42)  # reproducible

#Debug / safety knob (ONLY progress, no hard cap)
PRINT_EVERY = 1000  # print progress after these many expansions

# 1) Build synthetic graph
def build_synthetic_city(num_nodes=10, k=3, p=0.15):
    G0 = nx.watts_strogatz_graph(num_nodes, k, p, seed=42)
    # Relabel to Zone0..ZoneN-1 and make it undirected graph
    mapping = {i: f"Zone{i}" for i in range(num_nodes)}
    G = nx.relabel_nodes(G0, mapping)
    G = nx.Graph(G)  # ensure Graph type

    # Simulate vulnerability scores
    for node in G.nodes():
        deg = G.degree(node)
        base = min(1.0, 0.15 * deg + random.gauss(0.35, 0.15))
        vuln_score = max(0.0, min(1.0, base + random.gauss(0, 0.08)))
        if vuln_score >= 0.66:
            lvl = 'High'
        elif vuln_score >= 0.33:
            lvl = 'Med'
        else:
            lvl = 'Low'
        area_km2 = round(max(0.2, random.gauss(1.2, 0.4)), 3)
        intervene_cost = int(max(3, round(4 + area_km2 * 2 + random.uniform(-1, 3))))
        resilience_gain_est = round(vuln_score * area_km2, 4)

        G.nodes[node].update({
            'vuln_score': round(vuln_score, 3),
            'vuln_level': lvl,
            'area_km2': area_km2,
            'intervene_cost': intervene_cost,
            'resilience_gain_est': resilience_gain_est
        })

    # Edge weights
    for u, v in G.edges():
        G.edges[u, v]['weight'] = round(max(1.0, random.gauss(2.5, 1.0)), 2)

    return G

# Build smaller graph
G = build_synthetic_city(num_nodes=10, k=3, p=0.15)
START = 'Zone0'
BUDGET = 50  # smaller but still enough for a few interventions + travel

# Determine high-vuln nodes and goal threshold (50% of high-vuln nodes)
high_nodes = [n for n in G.nodes() if G.nodes[n]['vuln_level'] == 'High']
total_high = len(high_nodes)
if total_high == 0:
    scores = sorted(G.nodes(data=True), key=lambda x: -x[1]['vuln_score'])
    high_nodes = [n for n, _ in scores[:max(1, int(0.3 * len(G)))]]
    total_high = len(high_nodes)

TARGET_FRACTION = 0.5
TARGET_COUNT = ceil(TARGET_FRACTION * total_high)

print(f"Total nodes: {len(G)}; High-vuln nodes: {total_high}; "
      f"Target (>= {int(TARGET_FRACTION*100)}%): {TARGET_COUNT}")
print("High-vuln nodes:", high_nodes)

# Helper utilities
def score_coverage(covered_set):
    covered_high = [n for n in covered_set if G.nodes[n]['vuln_level'] == 'High']
    count = len(covered_high)
    gain = sum(G.nodes[n]['resilience_gain_est'] for n in covered_high)
    return count, gain

# UCS (Uniform Cost Search) ONLY
def ucs_search(start=START, budget=BUDGET):
    frontier = []
    # entries: (g_cost, node, covered_frozenset, path_list_of_actions)
    start_state = (0.0, start, frozenset(), [])
    heapq.heappush(frontier, start_state)
    seen_best = {}   # (node, covered_frozenset) -> best g
    expansions = 0

    while frontier:
        g, node, covered_fs, path = heapq.heappop(frontier)
        expansions += 1

        if expansions % PRINT_EVERY == 0:
            print(f"[UCS] Expanded {expansions} states | "
                  f"frontier={len(frontier)} | g={g:.2f}", flush=True)

        covered = set(covered_fs)
        key = (node, covered_fs)
        if key in seen_best and g > seen_best[key] + 1e-9:
            continue
        seen_best[key] = g

        # Goal test
        covered_count, _ = score_coverage(covered)
        if covered_count >= TARGET_COUNT and g <= budget:
            print(f"[UCS] Goal reached after {expansions} expansions. "
                  f"Cost={g:.2f}, coverage={covered_count}/{total_high}", flush=True)
            return {
                'success': True,
                'cost': g,
                'path': path,
                'covered': covered,
                'expansions': expansions
            }

        if g > budget:
            continue

        # Action 1: Intervene at current node (if high-vuln & not already covered)
        if node not in covered and G.nodes[node]['vuln_level'] == 'High':
            int_cost = G.nodes[node]['intervene_cost']
            new_g = g + int_cost
            if new_g <= budget:
                new_covered_fs = frozenset(set(covered) | {node})
                heapq.heappush(
                    frontier,
                    (new_g, node, new_covered_fs,
                     path + [f"Intervene({node})"])
                )

        # Action 2: Move to neighbors
        for nbr in G.neighbors(node):
            w = G.edges[node, nbr]['weight']
            new_g = g + w
            if new_g <= budget:
                heapq.heappush(
                    frontier,
                    (new_g, nbr, covered_fs,
                     path + [f"Move({node}->{nbr},cost={w})"])
                )

    print("[UCS] Frontier exhausted. No solution found within budget.")
    return {
        'success': False,
        'cost': float('inf'),
        'path': [],
        'covered': set(),
        'expansions': expansions
    }

# Run UCS and summarize
print("\nRunning UCS search only...")
ucs_result = ucs_search(START, BUDGET)
print("UCS finished. Success:", ucs_result['success'],
      "| cost:", ucs_result['cost'],
      "| expansions:", ucs_result.get('expansions'))

# Evaluation helpers
def evaluate_solution(sol):
    if not sol['success']:
        return {'success': False}
    covered = sol['covered']
    hc, gain = score_coverage(covered)
    coverage_pct = 100.0 * (hc / total_high) if total_high > 0 else 0.0
    return {
        'success': True,
        'cost': sol['cost'],
        'path_len': len(sol['path']),
        'covered_count': hc,
        'covered_gain': round(gain, 4),
        'coverage_pct': round(coverage_pct, 2),
        'path': sol['path'],
        'covered_nodes': sorted(list(covered))
    }

eval_ucs = evaluate_solution(ucs_result)
print("\n--- UCS Solution Summary ---")
print(eval_ucs)

# Visualization
def draw_solution(graph, sol, title):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 6))
    color_map = []
    labels = {}
    for n in graph.nodes():
        lvl = graph.nodes[n]['vuln_level']
        if lvl == 'High':
            color_map.append('#ff9999')  # light red
        elif lvl == 'Med':
            color_map.append('#fff399')  # light yellow
        else:
            color_map.append('#cfe6c7')  # light green
        labels[n] = f"{n}\n{graph.nodes[n]['vuln_score']:.2f}|c{graph.nodes[n]['intervene_cost']}"

    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=650)

    if sol.get('success'):
        covered = set(sol.get('covered_nodes', []))
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(covered),
            node_color='red', node_size=900, alpha=0.9
        )

    nx.draw_networkx_labels(graph, pos, labels, font_size=7)
    plt.title(title)
    plt.axis('off')
    plt.show()

draw_solution(
    G,
    eval_ucs,
    f"UCS result — cost {eval_ucs.get('cost')} | coverage {eval_ucs.get('coverage_pct')}%"
)


# Print action sequence (trim)
def print_actions(sol, name, max_actions=60):
    print(f"\n{name} actions (first {max_actions} shown):")
    if not sol['success']:
        print(" No feasible solution within budget.")
        return
    for i, a in enumerate(sol['path'][:max_actions], 1):
        print(f"{i:02d}. {a}")
    if len(sol['path']) > max_actions:
        print(" ... (truncated)")

print_actions(ucs_result, "UCS")


# In[6]:


# module2_resilience_search_astar_only_small.py
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random
from math import ceil

random.seed(42)  # reproducible

# Debug / safety knob (ONLY progress, no hard cap) 
PRINT_EVERY = 1000        # print progress after these many expansions

# 1) Build synthetic graph
def build_synthetic_city(num_nodes=10, k=3, p=0.15):
    G0 = nx.watts_strogatz_graph(num_nodes, k, p, seed=42)
    # Relabel to Zone0..ZoneN-1 and make it undirected graph
    mapping = {i: f"Zone{i}" for i in range(num_nodes)}
    G = nx.relabel_nodes(G0, mapping)
    G = nx.Graph(G)  # ensure Graph type

    # Simulate vulnerability scores (to stand in for Module 1 BN outputs)
    for node in G.nodes():
        # generate a base vulnerability that correlates with degree
        deg = G.degree(node)
        base = min(1.0, 0.15 * deg + random.gauss(0.35, 0.15))
        vuln_score = max(0.0, min(1.0, base + random.gauss(0, 0.08)))
        # classify level
        if vuln_score >= 0.66:
            lvl = 'High'
        elif vuln_score >= 0.33:
            lvl = 'Med'
        else:
            lvl = 'Low'
        area_km2 = round(max(0.2, random.gauss(1.2, 0.4)), 3)  # zone area
        # Intervention cost loosely proportional to area & built density
        intervene_cost = int(max(3, round(4 + area_km2 * 2 + random.uniform(-1, 3))))
        # resilience gain estimate: used for scoring sum-of-benefits
        resilience_gain_est = round(vuln_score * area_km2, 4)

        G.nodes[node].update({
            'vuln_score': round(vuln_score, 3),
            'vuln_level': lvl,
            'area_km2': area_km2,
            'intervene_cost': intervene_cost,
            'resilience_gain_est': resilience_gain_est
        })

    # Attach edge weights (travel/installation overhead)
    for u, v in G.edges():
        # distance-like weight 1..5
        G.edges[u, v]['weight'] = round(max(1.0, random.gauss(2.5, 1.0)), 2)

    return G

# Build smaller graph
G = build_synthetic_city(num_nodes=10, k=3, p=0.15)
START = 'Zone0'
BUDGET = 50  # smaller but still enough for a few interventions + travel

# Determine high-vuln nodes and goal threshold (50% of high-vuln nodes)
high_nodes = [n for n in G.nodes() if G.nodes[n]['vuln_level'] == 'High']
total_high = len(high_nodes)
if total_high == 0:
    # fallback: if no high nodes due to randomness, set target to top-k
    scores = sorted(G.nodes(data=True), key=lambda x: -x[1]['vuln_score'])
    high_nodes = [n for n, _ in scores[:max(1, int(0.3 * len(G)))]]
    total_high = len(high_nodes)

TARGET_FRACTION = 0.5  # <= changed from 0.8
TARGET_COUNT = ceil(TARGET_FRACTION * total_high)

print(f"Total nodes: {len(G)}; High-vuln nodes: {total_high}; "
      f"Target (>= {int(TARGET_FRACTION*100)}%): {TARGET_COUNT}")
print("High-vuln nodes:", high_nodes)

# Helper utilities
def score_coverage(covered_set):
    covered_high = [n for n in covered_set if G.nodes[n]['vuln_level'] == 'High']
    count = len(covered_high)
    gain = sum(G.nodes[n]['resilience_gain_est'] for n in covered_high)
    return count, gain

# Admissible heuristic for A*
def admissible_heuristic(covered_set):
    covered_count, _ = score_coverage(covered_set)
    remaining_needed = max(0, TARGET_COUNT - covered_count)
    if remaining_needed == 0:
        return 0
    # Find intervention costs for high-vuln nodes that are not yet covered
    remaining_high_costs = sorted(
        [G.nodes[n]['intervene_cost'] for n in high_nodes if n not in covered_set]
    )
    take = remaining_high_costs[:remaining_needed]
    return sum(take)

# A* Search ONLY
def astar_search(start=START, budget=BUDGET):
    frontier = []
    # entries: (f, g_cost, node, covered_frozenset, path_list_of_actions)
    start_entry = (admissible_heuristic(frozenset()), 0.0, start, frozenset(), [])
    heapq.heappush(frontier, start_entry)
    seen_best = {}  # (node, covered_frozenset) -> best g
    expansions = 0

    while frontier:
        f, g, node, covered_fs, path = heapq.heappop(frontier)
        expansions += 1

        if expansions % PRINT_EVERY == 0:
            print(f"[A* ] Expanded {expansions} states | "
                  f"frontier={len(frontier)} | g={g:.2f} | f={f:.2f}", flush=True)

        covered = set(covered_fs)
        key = (node, covered_fs)
        if key in seen_best and g > seen_best[key] + 1e-9:
            continue
        seen_best[key] = g

        # Goal test
        covered_count, _ = score_coverage(covered)
        if covered_count >= TARGET_COUNT and g <= budget:
            print(f"[A* ] Goal reached after {expansions} expansions. "
                  f"Cost={g:.2f}, coverage={covered_count}/{total_high}", flush=True)
            return {
                'success': True,
                'cost': g,
                'path': path,
                'covered': covered,
                'expansions': expansions
            }

        if g > budget:
            # No point expanding further from this state
            continue

        # Intervene at current node (if high-vuln and not covered)
        if node not in covered and G.nodes[node]['vuln_level'] == 'High':
            int_cost = G.nodes[node]['intervene_cost']
            new_g = g + int_cost
            if new_g <= budget:
                new_covered_fs = frozenset(set(covered) | {node})
                h = admissible_heuristic(new_covered_fs)
                heapq.heappush(
                    frontier,
                    (new_g + h, new_g, node, new_covered_fs,
                     path + [f"Intervene({node})"])
                )

        # Move to neighbors
        for nbr in G.neighbors(node):
            w = G.edges[node, nbr]['weight']
            new_g = g + w
            if new_g <= budget:
                h = admissible_heuristic(covered_fs)
                heapq.heappush(
                    frontier,
                    (new_g + h, new_g, nbr, covered_fs,
                     path + [f"Move({node}->{nbr},cost={w})"])
                )

    print("[A* ] Frontier exhausted. No solution found within budget.")
    return {
        'success': False,
        'cost': float('inf'),
        'path': [],
        'covered': set(),
        'expansions': expansions
    }

# Run A* and summarize
print("\nRunning A* search only...")
astar_result = astar_search(START, BUDGET)
print("A* finished. Success:", astar_result['success'],
      "| cost:", astar_result['cost'],
      "| expansions:", astar_result.get('expansions'))

# Evaluation helpers
def evaluate_solution(sol):
    if not sol['success']:
        return {'success': False}
    covered = sol['covered']
    hc, gain = score_coverage(covered)
    coverage_pct = 100.0 * (hc / total_high) if total_high > 0 else 0.0
    return {
        'success': True,
        'cost': sol['cost'],
        'path_len': len(sol['path']),
        'covered_count': hc,
        'covered_gain': round(gain, 4),
        'coverage_pct': round(coverage_pct, 2),
        'path': sol['path'],
        'covered_nodes': sorted(list(covered))
    }

eval_astar = evaluate_solution(astar_result)
print("\n--- A* Solution Summary ---")
print(eval_astar)

# Visualization
def draw_solution(graph, sol, title):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 6))
    # Node colors by vuln_level
    color_map = []
    labels = {}
    for n in graph.nodes():
        lvl = graph.nodes[n]['vuln_level']
        if lvl == 'High':
            color_map.append('#ff9999')  # light red
        elif lvl == 'Med':
            color_map.append('#fff399')  # light yellow
        else:
            color_map.append('#cfe6c7')  # light green
        labels[n] = f"{n}\n{graph.nodes[n]['vuln_score']:.2f}|c{graph.nodes[n]['intervene_cost']}"

    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=650)

    # highlight covered nodes if solution exists
    if sol.get('success'):
        covered = set(sol.get('covered_nodes', []))
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(covered),
            node_color='red', node_size=900, alpha=0.9
        )

    nx.draw_networkx_labels(graph, pos, labels, font_size=7)
    plt.title(title)
    plt.axis('off')
    plt.show()

draw_solution(
    G,
    eval_astar,
    f"A* result — cost {eval_astar.get('cost')} | coverage {eval_astar.get('coverage_pct')}%"
)

# Print action sequence (trim)
def print_actions(sol, name, max_actions=60):
    print(f"\n{name} actions (first {max_actions} shown):")
    if not sol['success']:
        print(" No feasible solution within budget.")
        return
    for i, a in enumerate(sol['path'][:max_actions], 1):
        print(f"{i:02d}. {a}")
    if len(sol['path']) > max_actions:
        print(" ... (truncated)")

print_actions(astar_result, "A*")


# In[ ]:




