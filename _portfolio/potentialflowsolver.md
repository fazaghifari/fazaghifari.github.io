---
title: "Potential Flow Solver"
excerpt: "Python code for solving laminar flow problem past an airfoil."
collection: portfolio
---

For further information, go to the [repository](https://github.com/fazaghifari/PotentialFlow_solver
# Laminar Viscous Solver
This code is a part of final project of Continuum Mechanics II, in Aerospace Engineering ITB

# Contributor:
- Ghifari Adam F
- Faber Yosua O
- Hanief Adnadi

This code was written in Python 3.6

Dependencies: numpy, matplotlib, copy

# How to use
1. Place all files in one folder:
    a. main.py
    b. calc_gamma.m
    c. dataimport.m
    d. geometrical_param.m
    e. gridgen_fcn.py
	f. dom_velo.py
	g. 2410.txt

2. The .txt file (Airfoil coordinates) follows selig format
   (Start from trailing edge and goes CCW)

3. Run main.py on the console 

4. Input known values. Example:

     Freestream velocity: 10 m/s
     Angle of Attack: 0 deg
	 Airfoil file: 2410.txt)
