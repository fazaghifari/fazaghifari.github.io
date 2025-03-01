---
title: 'Simple robust optimization tutorial'
date: 2025-02-03
permalink: /posts/2025/03/rob-opt-en/
tags:
  - robust optimization
  - linear programming
  - lp
---

Let's say that you are working in a local taxi company where they want to develop their in-house navigation software. This navigation software is responsible for showing the taxi drivers the optimum directions to the destination or pick-up points. Naturally, arriving quickly is crucial—after all, as the saying goes, time is money!

It sounds straightforward, right? As the engineer or scientist in the team, you might directly frame the problem as the [shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem). A logical approach would be to solve it using linear programming or Dijkstra’s algorithm to determine the optimal route. But wait, have you considered that your company operates in a bustling metropolitan city? Traffic jams, roadblocks, and unexpected delays can happen anytime, anywhere. The route is full of uncertainties! So, how do you tackle this problem?

There are several ways to model uncertainty, and a probabilistic approach might seem like the natural choice. However, since your team is relatively new and lacks the resources to collect enough traffic data for accurate probability distributions, this approach isn't feasible—at least for now. So instead, you can choose an interval set! An interval set is defined by the lower and upper bound of a possible value:

$$
t_{\dagger} = [\underline{t},\overline{t}].
$$

This allows you to represent travel time on each road segment (graph edge) as an interval rather than a fixed value, accounting for possible variations. 

To make this more concrete, let’s consider a simple routing problem using the following graph (download [here](https://drive.google.com/file/d/1n6yxpwlt8EiIsPzBVQREe0_wQr1mue_y/view?usp=sharing)). Throughout this tutorial, we'll use Python and Gurobi (academic license) as our optimization solver. First, let's import the required libraries.

```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
```
Since the data is in text format, we have to parse the data into the graph components:
```python
def parse_data(filename):
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f]
    k = int(lines[0])  # maximum number of arcs suggested
    nodes = lines[1].split()  # list of nodes
    edges = []  # list of edges
    edge_labels = {}  # edge labels
    for line in lines[2:]:
        splitline = line.split()
        edge = tuple(splitline[:2])
        edges.append(edge)
        edge_labels [edge] = [float(a) for a in splitline[2:]]
    
    edge_list= ["_".join(tup) for tup in edges]
    return k, nodes, edge_list, edge_labels
```
and finally, draw the graph:
```python
k, nodes, edge_list, edge_labels = parse_data("simple_route.dat")

G = nx.DiGraph()
edges = edge_labels.keys()
G.add_edges_from(edges)

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_color='red'
)
plt.show()
```
<p align="center">
  <img width="550" src='/images/rob_opt/simple_route.png' class="center">
</p>
<p align="center">
  <em>Figure 1. Simple routing problem with uncertain weight.</em>
</p>

In this problem, the taxi driver needs to travel from $s$ to $t$ using one of several possible routes, each affected by interval uncertainty. For example, on the path $s \rightarrow 1$, the travel time can range from a minimum of 1.5 minutes to a maximum of 19.7 minutes, depending on factors like traffic congestion or roadblocks. This variability means that the actual time required for each route isn’t fixed but lies within a given interval, adding uncertainty to the decision-making process.

It is important to note that in this tutorial, we simplify the problem a little bit. We determine $k=1$, where $k$ is the maximum number of arcs for which the travel time will be at the upper bound, with the travel time of all other arcs being at the lower bound. Simply put, when the algorithms plan the optimal direction, it only assumes that only one road section will be on the maximum value.

How to cite this article:
```latex
@misc{faza2025robustopttutorial,
   author =       {Faza, Ghifari Adam},
   title =        {Simple robust optimization tutorial},
   month =        {February},
   year =         {2025},
   url =          {https://fazaghifari.github.io/posts/2025/03/rob-opt-en/},
 }
```
