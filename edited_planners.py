#!/usr/bin/env python
# coding: utf-8

# # Planners — GraphPlan and POP (demonstration)
# 
# This notebook implements a simplified STRIPS-like domain for adaptation actions and demonstrates GraphPlan (fixed horizon) and a simple POP planner producing partial orders and causal links. Visualization included.

# In[1]:


# Basic planner structures and domain definition
from itertools import combinations
import networkx as nx, matplotlib.pyplot as plt, random, json, os
random.seed(42)
plt.rcParams['figure.figsize'] = (6,4)
zones = ['A','B','C','D']
initial_state = {'safe_water': set(), 'green': set(), 'retrofit': set()}
goal = {'safe_water': set(['A','B']), 'green': set(['C']), 'retrofit': set(['D'])}
class Action:
    def __init__(self,name,pre_add={},pre_not={}, add=set(), delete=set(), cost=1, duration=1):
        self.name=name; self.pre_add=pre_add; self.pre_not=pre_not; self.add=add; self.delete=delete; self.cost=cost; self.duration=duration
    def is_applicable(self, state):
        for pred,vals in self.pre_add.items():
            if not vals.issubset(state.get(pred,set())): return False
        for pred,vals in self.pre_not.items():
            if vals.intersection(state.get(pred,set())): return False
        return True
    def apply(self,state):
        ns = {k:set(v) for k,v in state.items()}
        for p in self.add:
            ns.setdefault(p[0], set()).add(p[1])
        for p in self.delete:
            if p[1] in ns.get(p[0], set()):
                ns[p[0]].remove(p[1])
        return ns
    def __repr__(self):
        return self.name
actions = [
    Action('Install_RWH_A', add={('safe_water','A')}, cost=3),
    Action('Plant_Trees_C', add={('green','C')}, cost=2),
    Action('Retrofit_D', add={('retrofit','D')}, cost=5),
    Action('Community_Outreach_B', add={('safe_water','B')}, cost=2)
]
print('Actions:', actions)
print('Initial state:', initial_state)
print('Goal:', goal)


# ## GraphPlan: fixed horizon search
# 
# We build a levelled graph up to horizon H and search for a plan that achieves the goal.

# In[2]:


def graphplan(actions, initial, goal, horizon=4):
    layers = [initial]
    action_layers = []
    for t in range(horizon):
        applicable = []
        for a in actions:
            if a.is_applicable(layers[-1]):
                applicable.append(a)
        action_layers.append(applicable)
        next_state = {k:set(v) for k,v in layers[-1].items()}
        for a in applicable:
            for p in a.add:
                next_state.setdefault(p[0], set()).add(p[1])
            for p in a.delete:
                if p[1] in next_state.get(p[0], set()):
                    next_state[p[0]].remove(p[1])
        layers.append(next_state)
        achieved = True
        for pred,vals in goal.items():
            if not vals.issubset(layers[-1].get(pred,set())):
                achieved = False; break
        if achieved:
            plan = []
            state = initial.copy()
            for t in range(len(action_layers)):
                for a in action_layers[t]:
                    if any((p[0] in goal and p[1] in goal[p[0]]) for p in a.add) and a.is_applicable(state):
                        plan.append((t,a))
                        state = a.apply(state)
            return True, plan
    return False, None

ok, plan = graphplan(actions, initial_state, goal, horizon=4)
print('GraphPlan found plan?', ok)
if ok:
    print('Plan (time,action):', plan)


# ## POP: Partially-Ordered Plan construction (simple causal-link planner)
# 
# We build causal links by selecting actions achieving subgoals and adding ordering constraints.

# In[3]:


def pop_planner(actions, initial, goal):
    chosen = []
    order_edges = []
    causal_links = []
    state = {k:set(v) for k,v in initial.items()}
    for pred, vals in goal.items():
        for v in vals:
            if v in state.get(pred, set()): continue
            cand = next((a for a in actions if (pred,v) in a.add), None)
            if cand is None:
                return False, None
            chosen.append(cand)
            causal_links.append((cand.name, (pred,v)))
    for a in chosen:
        for pred, vals in a.pre_add.items():
            for v in vals:
                provider = next((b for b in chosen if (pred,v) in b.add), None)
                if provider and provider!=a:
                    order_edges.append((provider.name, a.name))
    Gpo = nx.DiGraph()
    for a in chosen:
        Gpo.add_node(a.name)
    Gpo.add_edges_from(order_edges)
    return True, {'actions': [a.name for a in chosen], 'order_edges': order_edges, 'causal_links': causal_links, 'graph': Gpo}

ok, pop_plan = pop_planner(actions, initial_state, goal)
print('POP success?', ok)
if ok:
    print('POP plan details:', pop_plan['actions'], pop_plan['order_edges'], pop_plan['causal_links'])
    plt.figure(figsize=(5,3))
    nx.draw(pop_plan['graph'], with_labels=True, node_color='lightgreen', node_size=1400)
    plt.title('POP partial-order graph (actions as nodes)')
    plt.show()


# ## Demonstration of flexibility
# 
# GraphPlan returned a fixed time-sliced plan (if found). POP returns a partial order that can be revised; e.g., we can delay `Retrofit_D` until resources are available without breaking causal links.

# In[4]:


if ok:
    delay = Action('Delay', add=set(), cost=0)
    pop_plan['graph'].add_node(delay.name)
    pop_plan['graph'].add_edge('Plant_Trees_C', delay.name)
    pop_plan['graph'].add_edge(delay.name, 'Retrofit_D')
    plt.figure(figsize=(6,3))
    nx.draw(pop_plan['graph'], with_labels=True, node_color='lightblue', node_size=1200)
    plt.title('POP graph after inserting a delay (revisable ordering)')
    plt.show()


# In[ ]:




