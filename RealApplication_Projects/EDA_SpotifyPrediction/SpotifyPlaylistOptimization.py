# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:02:19 2024

@author: 2mgia
"""

import numpy as np
import pandas as pd
import cvxpy as cp

df = pd.read_csv("optimizationSpotify(1).csv")

trackPop = df.track_popularity
x = cp.Variable(len(trackPop), boolean=True)

obj = cp.Maximize(sum(x @ trackPop))

constraints = [x.sum()>=25,
               x.sum()<=100,
               ((x @ df.energy)/np.count_nonzero(x @ df.energy)) >= 0.75,
               ((x @ df.danceability)/np.count_nonzero(x @ df.energy)) >= 0.5,
               ((x @ df.speechiness)/np.count_nonzero(x @ df.energy)) >= 0.14,
               ((x @ df.loudness)/np.count_nonzero(x @ df.energy)) <= -2,
               ((x @ df.loudness)/np.count_nonzero(x @ df.energy)) >= -15
               ]
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.GLPK_MI)
print(prob.status)

output=x.value

####next playlist

x2 = cp.Variable(len(trackPop), boolean=True)

obj2 = cp.Maximize(sum(x2 @ trackPop))

constraints2 = [x2.sum()>=20,
               x2.sum()<=100,
               ((x2 @ df.loudness)/np.count_nonzero(x2 @ df.energy)) <= -8,
               ((x2 @ df.speechiness)/np.count_nonzero(x2 @ df.energy)) <= 0.5,
               ((x2 @ df.energy)/np.count_nonzero(x2 @ df.energy)) <= 8
               ]
prob2 = cp.Problem(obj2, constraints2)
prob2.solve(solver=cp.GLPK_MI)
print(prob2.status)

output2=x2.value



               
