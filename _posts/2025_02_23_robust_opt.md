---
title: 'Simple robust optimization tutorial'
date: 2025-02-23
permalink: /posts/2025/02/rob-opt-en/
tags:
  - robust optimization
  - linear programming
  - lp
---

Let's say that you are working in a local taxi company where they want to develop their in-house navigation software. This navigation software is responsible for showing the taxi drivers the optimum directions to the destination or pick-up points. Naturally, arriving quickly is crucial—after all, as the saying goes, time is money!

It sounds straightforward, right? As the engineer or scientist in the team, you might directly frame the problem as the [shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem). A logical approach would be to solve it using linear programming or Dijkstra’s algorithm to determine the optimal route. But wait, have you considered that your company operates in a bustling metropolitan city? Traffic jams, roadblocks, and unexpected delays can happen anytime, anywhere. The route is full of uncertainties! So, how do you tackle this problem?

We can choose different approaches to model the uncertainty. A probabilistic model would be the natural approach to tackle uncertainty problems. However, since this team is relatively new, you don't have the resources yet to collect all of the time that is required to pass through a road segment to make an accurate distribution. So instead, you can choose an interval set! An interval set is defined by the lower and upper bound of a possible value:

$$
t_{\dagger} = [\underline{t},\overline{t}].
$$

Thus, you can place the interval cost (in this sense, the time) over the road section (graph edges).  Now to make everything's clearer, let's consider a simple routing problem from this graph (download [here](https://drive.google.com/file/d/1n6yxpwlt8EiIsPzBVQREe0_wQr1mue_y/view?usp=sharing)). 


How to cite this article:
```latex
@misc{faza2024mdplptutorial,
   author =       {Faza, Ghifari Adam},
   title =        {Simple robust optimization tutorial},
   month =        {February},
   year =         {2025},
   url =          {https://fazaghifari.github.io/posts/2025/02/rob-opt-en/},
 }
```
