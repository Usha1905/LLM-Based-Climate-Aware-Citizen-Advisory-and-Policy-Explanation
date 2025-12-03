#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.independencies import Independencies
import networkx as nx
import matplotlib.pyplot as plt

model = DiscreteBayesianNetwork([
    ('LandUseDensity', 'ImperviousSurface'),
    ('LandUseDensity', 'NDVI_cat'),
    ('HistoricalClimate', 'FloodRisk'),
    ('ImperviousSurface', 'FloodRisk'),
    ('NDVI_cat', 'HeatRisk'),
    ('ImperviousSurface', 'HeatRisk'),
    ('NDVI_cat', 'WaterScarcityRisk'),
    ('HistoricalClimate', 'WaterScarcityRisk'),
    ('HeatRisk', 'HealthRisk'),
    ('WaterScarcityRisk', 'HealthRisk'),
    ('SocioEcon', 'HealthRisk'),
])

cpd_landuse = TabularCPD(
    variable='LandUseDensity',
    variable_card=3,
    values=[[0.3], [0.4], [0.3]],
    state_names={'LandUseDensity': ['Low', 'Medium', 'High']}
)

# HistoricalClimate: None, Flood, Drought, Heatwave, Mixed
cpd_hist = TabularCPD(
    variable='HistoricalClimate',
    variable_card=5,
    values=[[0.5], [0.15], [0.15], [0.15], [0.05]],
    state_names={'HistoricalClimate': ['None', 'Flood', 'Drought', 'Heatwave', 'Mixed']}
)

# SocioEcon: LowIncome, Middle, High
cpd_ses = TabularCPD(
    variable='SocioEcon',
    variable_card=3,
    values=[[0.35], [0.45], [0.20]],
    state_names={'SocioEcon': ['LowIncome', 'Middle', 'High']}
)

# ImperviousSurface depends on LandUseDensity: Low/High
cpd_imp = TabularCPD(
    variable='ImperviousSurface',
    variable_card=2,
    values=[
        # Impervious=Low
        [0.85, 0.5, 0.2],
        # Impervious=High
        [0.15, 0.5, 0.8]
    ],
    evidence=['LandUseDensity'],
    evidence_card=[3],
    state_names={
        'ImperviousSurface': ['Low', 'High'],
        'LandUseDensity': ['Low', 'Medium', 'High']
    }
)

# NDVI_cat depends on LandUseDensity: High/Medium/Low
cpd_ndvi = TabularCPD(
    variable='NDVI_cat',
    variable_card=3,
    values=[
        # NDVI High
        [0.7, 0.35, 0.1],
        # NDVI Medium
        [0.25, 0.45, 0.3],
        # NDVI Low
        [0.05, 0.2, 0.6]
    ],
    evidence=['LandUseDensity'],
    evidence_card=[3],
    state_names={
        'NDVI_cat': ['High', 'Medium', 'Low'],
        'LandUseDensity': ['Low', 'Medium', 'High']
    }
)

# FloodRisk depends on HistoricalClimate & ImperviousSurface
# FloodRisk: No, Yes
# Evidence order: ['HistoricalClimate', 'ImperviousSurface']
cpd_flood = TabularCPD(
    variable='FloodRisk',
    variable_card=2,
    values=[
        # FloodRisk=No
        [
            0.98, 0.90,  # None + Imp: Low/High
            0.30, 0.10,  # Flood + Imp: Low/High
            0.90, 0.70,  # Drought + Imp
            0.95, 0.85,  # Heatwave + Imp
            0.6, 0.4     # Mixed + Imp
        ],
        # FloodRisk=Yes
        [
            0.02, 0.10,
            0.70, 0.90,
            0.10, 0.30,
            0.05, 0.15,
            0.40, 0.60
        ]
    ],
    evidence=['HistoricalClimate', 'ImperviousSurface'],
    evidence_card=[5, 2],
    state_names={
        'FloodRisk': ['No', 'Yes'],
        'HistoricalClimate': ['None', 'Flood', 'Drought', 'Heatwave', 'Mixed'],
        'ImperviousSurface': ['Low', 'High']
    }
)

# HeatRisk depends on NDVI_cat and ImperviousSurface
cpd_heat = TabularCPD(
    variable='HeatRisk',
    variable_card=2,
    values=[
        # HeatRisk=No (NDVI High/Med/Low x Imp Low/High)
        [
            0.98, 0.9,   # NDVI High (Imp Low/High)
            0.85, 0.6,   # NDVI Medium
            0.5, 0.2     # NDVI Low
        ],
        # HeatRisk=Yes
        [
            0.02, 0.1,
            0.15, 0.4,
            0.5, 0.8
        ]
    ],
    evidence=['NDVI_cat', 'ImperviousSurface'],
    evidence_card=[3, 2],
    state_names={
        'HeatRisk': ['No', 'Yes'],
        'NDVI_cat': ['High', 'Medium', 'Low'],
        'ImperviousSurface': ['Low', 'High']
    }
)

# WaterScarcityRisk depends on NDVI_cat and HistoricalClimate
# Build CPT programmatically (more transparent)
p_yes = []
for ndvi in ['High', 'Medium', 'Low']:
    for hist in ['None', 'Flood', 'Drought', 'Heatwave', 'Mixed']:
        if ndvi == 'High':
            if hist == 'Drought':
                p = 0.25
            elif hist == 'Mixed':
                p = 0.35
            else:
                p = 0.05
        elif ndvi == 'Medium':
            if hist == 'Drought':
                p = 0.45
            elif hist == 'Mixed':
                p = 0.55
            else:
                p = 0.12
        else:  # NDVI Low
            if hist == 'Drought':
                p = 0.75
            elif hist == 'Mixed':
                p = 0.8
            else:
                p = 0.4
        p_yes.append(p)

p_no = [1 - p for p in p_yes]

cpd_water = TabularCPD(
    variable='WaterScarcityRisk',
    variable_card=2,
    values=[p_no, p_yes],
    evidence=['NDVI_cat', 'HistoricalClimate'],
    evidence_card=[3, 5],
    state_names={
        'WaterScarcityRisk': ['No', 'Yes'],
        'NDVI_cat': ['High', 'Medium', 'Low'],
        'HistoricalClimate': ['None', 'Flood', 'Drought', 'Heatwave', 'Mixed']
    }
)

# HealthRisk depends on HeatRisk, WaterScarcityRisk, SocioEcon
heat_states = ['No', 'Yes']
water_states = ['No', 'Yes']
ses_states = ['LowIncome', 'Middle', 'High']

vals_no = []
vals_yes = []
for heat in heat_states:
    for water in water_states:
        for ses in ses_states:
            base = 0.02  # baseline
            if heat == 'Yes':
                base += 0.25
            if water == 'Yes':
                base += 0.15
            if ses == 'LowIncome':
                base += 0.2
            elif ses == 'Middle':
                base += 0.05

            base = max(0.01, min(base, 0.95))
            vals_yes.append(base)
            vals_no.append(1 - base)

cpd_health = TabularCPD(
    variable='HealthRisk',
    variable_card=2,
    values=[vals_no, vals_yes],
    evidence=['HeatRisk', 'WaterScarcityRisk', 'SocioEcon'],
    evidence_card=[2, 2, 3],
    state_names={
        'HealthRisk': ['No', 'Yes'],
        'HeatRisk': ['No', 'Yes'],
        'WaterScarcityRisk': ['No', 'Yes'],
        'SocioEcon': ['LowIncome', 'Middle', 'High']
    }
)

# 3. Add CPDs to model & check
model.add_cpds(
    cpd_landuse, cpd_hist, cpd_ses,
    cpd_imp, cpd_ndvi, cpd_flood,
    cpd_heat, cpd_water, cpd_health
)

assert model.check_model(), "Model failed consistency checks. Re-check CPDs."
print("Model is valid and CPDs added.")

# 4. Inference setup
inference = VariableElimination(model)

def query_example(evidence, query_vars):
    q = inference.query(variables=query_vars, evidence=evidence, show_progress=False)
    return q

# Example inference: scenario (dense built area, high impervious, historical floods, low NDVI)
evidence1 = {
    'LandUseDensity': 'High',
    'ImperviousSurface': 'High',
    'HistoricalClimate': 'Flood',
    'NDVI_cat': 'Low',
    'SocioEcon': 'LowIncome'
}
print("\nScenario 1 evidence:", evidence1)
q1 = query_example(
    evidence=evidence1,
    query_vars=['FloodRisk', 'HeatRisk', 'WaterScarcityRisk', 'HealthRisk']
)
print(q1)

#5.Generate synthetic data from this BN (sampling)
sampler = BayesianModelSampling(model)

#forward_sample now directly returns a DataFrame; no return_type argument
synthetic = sampler.forward_sample(
    size=5000,
    include_latents=False,
    seed=42,
    show_progress=False
)
print("\nSynthetic sample head:")
print(synthetic.head())

# Example: learn parameters from synthetic data
estimator_ml = MaximumLikelihoodEstimator(model, synthetic)
cpd_flood_learned = estimator_ml.estimate_cpd('FloodRisk')
print("\nLearned CPD for FloodRisk (MLE):")
print(cpd_flood_learned)

# Bayesian parameter estimation
estimator_bayes = BayesianEstimator(model, synthetic)
cpd_health_bayes = estimator_bayes.estimate_cpd(
    'HealthRisk',
    prior_type='BDeu',
    equivalent_sample_size=10
)
print("\nLearned (Bayesian) CPD for HealthRisk (Bayesian estimator):")
print(cpd_health_bayes)

# 6. D-separation checks (independence)
dsep = model.is_dconnected(
    'LandUseDensity',
    'HealthRisk',
    observed=['NDVI_cat', 'ImperviousSurface']
)
print("\nAre LandUseDensity and HealthRisk d-connected given NDVI and ImperviousSurface? ->", dsep)

dsep2 = model.is_dconnected(
    'LandUseDensity',
    'FloodRisk',
    observed=['ImperviousSurface']
)
print("Are LandUseDensity and FloodRisk d-connected given ImperviousSurface? ->", dsep2)

# 7. Simple visualization
def plot_model(m):
    G = nx.DiGraph()
    G.add_nodes_from(m.nodes())
    G.add_edges_from(m.edges())
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=1800, font_size=8)
    plt.title("Bayesian Network Structure (Climate Vulnerability)")
    plt.tight_layout()
    plt.show()

plot_model(model)

# 8. Helper: map continuous NDVI -> NDVI_cat
def ndvi_to_category(ndvi_value):
    # ndvi_value expected in [-1, 1]
    if ndvi_value >= 0.4:
        return 'High'
    elif ndvi_value >= 0.15:
        return 'Medium'
    else:
        return 'Low'

print("\nNDVI mapping examples:",
      ndvi_to_category(0.6),
      ndvi_to_category(0.2),
      ndvi_to_category(0.05))

# 9. Example: run multiple scenario queries
scenarios = [
    {
        'LandUseDensity': 'High',
        'ImperviousSurface': 'High',
        'HistoricalClimate': 'Flood',
        'NDVI_cat': 'Low',
        'SocioEcon': 'LowIncome'
    },
    {
        'LandUseDensity': 'Low',
        'ImperviousSurface': 'Low',
        'HistoricalClimate': 'None',
        'NDVI_cat': 'High',
        'SocioEcon': 'High'
    },
    {
        'LandUseDensity': 'Medium',
        'ImperviousSurface': 'High',
        'HistoricalClimate': 'Drought',
        'NDVI_cat': 'Low',
        'SocioEcon': 'Middle'
    },
]
for i, ev in enumerate(scenarios, start=1):
    q = query_example(
        evidence=ev,
        query_vars=['FloodRisk', 'HeatRisk', 'WaterScarcityRisk', 'HealthRisk']
    )
    print(f"\nScenario {i} results:\n{q}")


# In[2]:


pip install pgmpy


# In[ ]:




