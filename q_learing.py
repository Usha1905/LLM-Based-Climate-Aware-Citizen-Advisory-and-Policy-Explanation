#!/usr/bin/env python
# coding: utf-8

# In[2]:


# module4_qlearning.py
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, defaultdict

np.random.seed(42)
random.seed(42)

# 1) DISCRETIZATION / MDP SETUP
# State representation: (heat, green_bucket, equity, budget_bucket)
# heat: 0=Low,1=Med,2=High
HEAT_STATES = [0, 1, 2]

# green % buckets: 0,20,40,60,80,100 (5% increments coarse)
GREEN_BUCKETS = np.array([0, 20, 40, 60, 80, 100])
NB_GREEN = len(GREEN_BUCKETS)

# equity: 0=Low,1=Med,2=High
EQUITY_STATES = [0, 1, 2]

# budget buckets (millions): 0, 200, 400, 600, 800, 1000
BUDGET_BUCKETS = np.array([0, 200, 400, 600, 800, 1000])
NB_BUDGET = len(BUDGET_BUCKETS)

# Build state list and index mapping
STATES = []
for h in HEAT_STATES:
    for g in range(NB_GREEN):
        for e in EQUITY_STATES:
            for b in range(NB_BUDGET):
                STATES.append((h, g, e, b))
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)

# Actions
ACTIONS = ['InvestGreen', 'SubsidizeCooling', 'EducateCommunities', 'RelocateHighRisk', 'DoNothing']
N_ACTIONS = len(ACTIONS)

# Hyperparameters
ALPHA = 0.1       # learning rate
GAMMA = 0.9       # discount factor
EPSILON = 0.12    # exploration prob
EPISODES = 2000
EP_LEN = 10       # years per episode
CLIMATE_HEAT_TREND = 0.05  # baseline yearly upward drift to heat (probability shift)

# Costs (millions)
COSTS = {
    'InvestGreen': 200,
    'SubsidizeCooling': 150,
    'EducateCommunities': 50,
    'RelocateHighRisk': 300,
    'DoNothing': 0
}

# Effect magnitudes (expected)
# InvestGreen: +green_pct approx +15-25; reduces heat over time
# SubsidizeCooling: immediate heat index reduction probability
# Educate: small equity gain
# RelocateHighRisk: large equity + reduces exposure (affecting long-term heat risk)
# We implement stochastic transitions reflecting these
EFFECTS = {
    'InvestGreen': {'green_delta': (12, 18), 'equity_delta': (0, 0), 'heat_reduction_prob': 0.35},
    'SubsidizeCooling': {'green_delta': (0, 0), 'equity_delta': (0, 0), 'heat_reduction_prob': 0.6},
    'EducateCommunities': {'green_delta': (0, 4), 'equity_delta': (0.2, 0.35), 'heat_reduction_prob': 0.05},
    'RelocateHighRisk': {'green_delta': (0, 0), 'equity_delta': (0.5, 0.8), 'heat_reduction_prob': 0.25},
    'DoNothing': {'green_delta': (0, 0), 'equity_delta': (0.0, 0.0), 'heat_reduction_prob': 0.0}
}

# Reward function parameters (weights)
W_COST = -1.0          # linear penalty for cost (millions)
W_GREEN = 6.0          # reward per green % gained (scaled)
W_HEAT = -30.0         # penalty for being in high heat state (per-year)
W_EQUITY = 40.0        # reward scaling for equity improvement
W_BUDGET_SHORTAGE = -0.05  # penalty per million below 0 if overdrawn

# Utility functions for discretization / buckets
def clamp(v, low, high):
    return max(low, min(high, v))

def green_bucket_from_pct(pct):
    # pick nearest bucket index
    idx = int(np.argmin(np.abs(GREEN_BUCKETS - pct)))
    return idx

def budget_bucket_from_amount(amount):
    idx = int(np.argmin(np.abs(BUDGET_BUCKETS - amount)))
    return idx

def heat_from_probability(prob):
    # map a latent continuous heat metric to discrete 0/1/2
    if prob < 0.33: return 0
    if prob < 0.66: return 1
    return 2

# Initial scenario (Odisha-like)
INIT_HEAT = 2  # high
INIT_GREEN_PCT = 20
INIT_EQUITY = 0  # low
INIT_BUDGET = 800  # million

INIT_STATE = (INIT_HEAT, green_bucket_from_pct(INIT_GREEN_PCT), INIT_EQUITY, budget_bucket_from_amount(INIT_BUDGET))

# 2) ENVIRONMENT STEP FUNCTION (stochastic)
def step_env(state_idx, action_idx):
    """
    Given state index and action index, sample next state and reward.
    Returns (next_state_idx, reward, info).
    """
    s = STATES[state_idx]
    h_idx, g_idx, e_idx, b_idx = s
    action = ACTIONS[action_idx]
    cost = COSTS[action]

    # Current continuous proxies
    green_pct = GREEN_BUCKETS[g_idx]
    equity_level = e_idx  # 0/1/2
    budget_amount = BUDGET_BUCKETS[b_idx]

    # Apply cost immediately (but budget buckets coarse)
    new_budget = max(0.0, budget_amount - cost)

    # Apply green changes (stochastic within EFFECTS bounds)
    gmin, gmax = EFFECTS[action]['green_delta']
    if gmin == gmax == 0:
        green_change = 0.0
    else:
        green_change = float(np.random.uniform(gmin, gmax))

    new_green_pct = clamp(green_pct + green_change, 0, 100)

    # Equity change (small increments)
    emin, emax = EFFECTS[action]['equity_delta']
    equity_change_cont = float(np.random.uniform(emin, emax))  # continuous
    # map to bucket change: convert continuous 0..1+ to bucket steps
    new_equity_cont = clamp(equity_level + equity_change_cont, 0.0, 2.0)
    # discretize to nearest
    new_equity = int(round(new_equity_cont))
    new_equity = clamp(new_equity, 0, 2)

    # Heat: baseline climate trend increases heat; actions can probabilistically reduce heat
    # Represent a latent heat probability p_heat: higher p -> higher chance to be in Med/High
    # We'll model p_heat from current discrete heat as centers: Low:0.15, Med:0.5, High:0.85
    p_map = {0: 0.15, 1: 0.5, 2: 0.85}
    p_heat = p_map[h_idx]
    # apply baseline upward trend
    p_heat = clamp(p_heat + CLIMATE_HEAT_TREND * np.random.normal(1.0, 0.03), 0.0, 1.0)

    # action effect: with some probability reduce p_heat
    reduction_prob = EFFECTS[action]['heat_reduction_prob']
    if np.random.rand() < reduction_prob:
        # reduce p_heat by a random fraction
        p_heat = clamp(p_heat - np.random.uniform(0.12, 0.28), 0.0, 1.0)

    # Increase green has small cooling effect too (probabilistic)
    if green_change > 0 and np.random.rand() < 0.4:
        p_heat = clamp(p_heat - np.random.uniform(0.03, 0.12), 0.0, 1.0)

    # map back to discrete heat state
    new_heat = heat_from_probability(p_heat)

    # Budget bucket
    new_budget_bucket = budget_bucket_from_amount(new_budget)

    next_state = (new_heat, green_bucket_from_pct(new_green_pct), new_equity, new_budget_bucket)
    next_idx = STATE_TO_IDX[next_state]

    # Reward design:
    # reward = -cost (penalize spend) + W_GREEN * (green_pct gain) + W_EQUITY * (equity bucket increase) + W_HEAT penalty
    green_gain = (new_green_pct - green_pct)
    equity_gain = (new_equity - equity_level)  # typically 0,1,2 differences
    heat_penalty = 0.0
    if new_heat == 2:
        heat_penalty = 1.0
    elif new_heat == 1:
        heat_penalty = 0.4
    else:
        heat_penalty = 0.0

    budget_penalty = 0.0
    if new_budget < 0:
        budget_penalty = W_BUDGET_SHORTAGE * abs(new_budget)

    reward = (W_COST * (cost/1.0)) + (W_GREEN * (green_gain)) + (W_EQUITY * equity_gain) + (W_HEAT * heat_penalty) + budget_penalty

    info = {
        'action': action, 'cost': cost,
        'green_gain': green_gain, 'equity_gain': equity_gain,
        'heat_before': h_idx, 'heat_after': new_heat,
        'budget_before': budget_amount, 'budget_after': new_budget
    }
    return next_idx, reward, info

# 3) Q-LEARNING ALGORITHM (tabular)
def q_learning_train(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                     episodes=EPISODES, ep_len=EP_LEN, verbose=True):
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []
    for ep in range(episodes):
        state_idx = STATE_TO_IDX[INIT_STATE]  # start from same initial each episode (could randomize)
        total_r = 0.0
        for t in range(ep_len):
            # ε-greedy
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                action_idx = int(np.argmax(Q[state_idx]))
            next_idx, r, info = step_env(state_idx, action_idx)
            total_r += r
            # Q update
            best_next = np.max(Q[next_idx])
            Q[state_idx, action_idx] += alpha * (r + gamma * best_next - Q[state_idx, action_idx])
            state_idx = next_idx
        returns.append(total_r)
        if verbose and (ep % max(1, episodes//10) == 0):
            print(f"Episode {ep}/{episodes}, return={total_r:.2f}")
    return Q, returns

# 4) Reactive Baseline
def reactive_policy(state_idx):
    # simple rule: if heat is High -> InvestGreen; if Med -> SubsidizeCooling; else DoNothing
    s = STATES[state_idx]
    h_idx, g_idx, e_idx, b_idx = s
    if h_idx == 2:
        return ACTIONS.index('InvestGreen')
    if h_idx == 1:
        return ACTIONS.index('SubsidizeCooling')
    return ACTIONS.index('DoNothing')

def evaluate_policy(policy_fn, episodes=500, ep_len=EP_LEN):
    returns = []
    final_stats = []
    for ep in range(episodes):
        state_idx = STATE_TO_IDX[INIT_STATE]
        total_r = 0.0
        for t in range(ep_len):
            action_idx = policy_fn(state_idx)
            next_idx, r, info = step_env(state_idx, action_idx)
            total_r += r
            state_idx = next_idx
        returns.append(total_r)
        s = STATES[state_idx]
        final_stats.append({'final_state': s, 'total_reward': total_r})
    return returns, final_stats

# 5) RUN TRAIN + EVAL + PLOTS
def smooth(x, w=50):
    if len(x) < w: return x
    out = []
    dq = deque(maxlen=w)
    for v in x:
        dq.append(v)
        out.append(np.mean(dq))
    return out

def main():
    print("Training Q-learning...")
    Q, returns = q_learning_train(episodes=1500, verbose=True)
    avg_ret = np.mean(returns[-200:])
    print("Training done. avg return (last 200 eps):", avg_ret)

    # evaluate greedy policy (no exploration)
    def greedy_policy(state_idx):
        return int(np.argmax(Q[state_idx]))

    print("Evaluating greedy Q policy vs reactive baseline...")
    q_returns, q_stats = evaluate_policy(greedy_policy, episodes=500)
    base_returns, base_stats = evaluate_policy(reactive_policy, episodes=500)

    print("Q-policy mean return:", np.mean(q_returns), "std:", np.std(q_returns))
    print("Reactive baseline mean return:", np.mean(base_returns), "std:", np.std(base_returns))

    # Plot learning curve
    plt.figure(figsize=(10,5))
    plt.plot(returns, alpha=0.4, label='episode returns')
    plt.plot(smooth(returns, w=50), label='smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Return (cumulative)')
    plt.title('Q-Learning training returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q_learning_returns.png', dpi=150)
    print("Saved q_learning_returns.png")

    # Boxplot comparison
    plt.figure(figsize=(6,5))
    plt.boxplot([q_returns, base_returns], labels=['Q-policy', 'Reactive'])
    plt.ylabel('Episode Return (10-year)')
    plt.title('Policy Comparison')
    plt.tight_layout()
    plt.savefig('policy_comparison_box.png', dpi=150)
    print("Saved policy_comparison_box.png")

    # Quick policy summary: for a selection of states show greedy action
    sample_states = []
    for h in HEAT_STATES:
        for g_idx in [0, 2, 4]:  # low/med/high green
            for e in EQUITY_STATES:
                s = (h, g_idx, e, budget_bucket_from_amount(INIT_BUDGET))
                sample_states.append(s)
    print("\nSample greedy actions for typical states:")
    for s in sample_states:
        idx = STATE_TO_IDX[s]
        a = ACTIONS[int(np.argmax(Q[idx]))]
        print(f"State H{ s[0] } G{ GREEN_BUCKETS[s[1]] } E{ s[2] } -> {a}")

    # Save Q table (numpy)
    np.save('q_table.npy', Q)
    print("Saved q_table.npy")

if __name__ == '__main__':
    main()


# In[ ]:




