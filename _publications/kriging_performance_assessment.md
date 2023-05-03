---
title: "Performance assessment of Kriging with partial least squares for high-dimensional uncertainty and sensitivity analysis"
collection: publications
permalink: /publication/kriging_performance_assessment
excerpt: 'This paper aims to assess the potential of Kriging combined with partial least squares (KPLS) for fast uncertainty quantification and sensitivity analysis in high-dimensional problems.'
date: 2023-04-26
venue: ' Structural and Multidisciplinary Optimization '
paperurl: 'https://link.springer.com/article/10.1007/s00158-023-03547-3'
---
This paper aims to assess the potential of Kriging combined with partial least squares (KPLS) for fast uncertainty quantification and sensitivity analysis in high-dimensional problems. Such a fast assessment is especially important in cases that involve a large number of outputs such as uncertain scalar fields or applications in robust and reliability-based optimization. In this regard, the role of the partial least squares is to reduce the dimensionality of the input space to accelerate model construction. We conduct experiments using KPLS on analytical and nonanalytical problems of various complexities and compare various quantities of interest (QOI), i.e., mean, standard deviation, and Sobol sensitivity indices, to those from the original Kriging to perform this assessment. In addition, a comparison with sparse polynomial chaos expansion (PCE) is also performed on nonanalytical problems. Results show that KPLS with four principal components is significantly faster than the ordinary Kriging while yielding comparable accuracy in approximating the statistical moments and Sobol indices. We also observe that KPLS with a proper number of principal components can achieve higher accuracy than Kriging in high-dimensional problems with small sample size, suggesting that the benefit of KPLS is not just on the training time but also accuracy. Finally, we observe no apparent benefits in utilizing KPLS for low-dimensional problems.

[Download paper here](https://link.springer.com/article/10.1007/s00158-023-03547-3)
