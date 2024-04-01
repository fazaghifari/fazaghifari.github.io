---
title: "Interval Reduced Order Surrogate Modelling Framework for Uncertainty Quantification"
collection: publications
permalink: /publication/interval_pod_pce
excerpt: 'This paper introduces a non-intrusive framework for epistemic surrogate modeling, leveraging interval proper orthogonal decomposition 
(interval POD) and interval polynomial chaos expansion (interval PCE) to handle interval observations, addressing a major limitation in existing frameworks. 
By integrating POD for interval data with PCE for interval observations, the framework allows for the consideration of non-scalar data, 
such as intervals, providing a more comprehensive approach to physical system modeling that captures additional information.'
date: 2024-01-04
venue: ' AIAA SciTech 2024 '
paperurl: 'https://arc.aiaa.org/doi/10.2514/6.2024-0387'
---
Surrogate models are widely used in the engineering community to approximate costly and large evaluation processes, 
such as difficult experiments or expensive simulations. This paper presents a non-intrusive framework for epistemic surrogate modelling, 
which is based on proper orthogonal decomposition (POD) and polynomial chaos expansion (PCE) for interval observations. 
In physical systems modelling, it is important to consider both aleatoric and epistemic uncertainty by constructing an uncertainty model to be 
included in the surrogate model. However, existing frameworks have a major limitation in that they can only handle 
scalar data observations and are not designed for non-deterministic observations like interval data. In many applications, 
the observed data can be inherently non-scalar due to various factors, such as observation uncertainties, conflicting data, or summarized data. 
In our proposed framework, we integrate POD for interval data with PCE for interval observations. 
Firstly, we employ interval POD to obtain an optimally reduced-order basis from the full-order snapshot. 
Then, we approximate this reduced-order basis using a non-intrusive interval PCE method. 
Allowing non-scalar data, such as intervals, is advantageous as it takes into account more information in the physical system modelling.

[Download unformatted version of the paper here](https://www.researchgate.net/publication/377785658_Interval_Reduced_Order_Surrogate_Modelling_Framework_for_Uncertainty_Quantification)
