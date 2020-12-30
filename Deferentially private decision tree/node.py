from collections import defaultdict
import random
import numpy as np
import math
import statistics
from scipy import stats # for Exponential Mechanism

class node:
    def __init__(self, parent_node, split_value_from_parent, splitting_attribute, tree_level, id, children, svfp_numer=None):
        self._parent_node = parent_node
        self._split_value_from_parent = split_value_from_parent
        self._svfp_numer = svfp_numer
        self._splitting_attribute = splitting_attribute
        #self._level = tree_level # comment out unless needed. saves memory.
        #self._id = id # comment out unless needed. saves memory.
        self._children = children
        self._class_counts = defaultdict(int)
        self._noisy_majority = None
        self._empty = 0 # 1 if leaf and has no records
        self._sensitivity = -1.0

    def add_child(self, child_node):
        self._children.append(child_node)

    def increment_class_count(self, class_value):
        self._class_counts[class_value] += 1

    def set_noisy_majority(self, epsilon, class_values):
        if not self._noisy_majority and not self._children: # to make sure this code is only run once per leaf
            for val in class_values:
                if val not in self._class_counts: self._class_counts[val] = 0

            if max([v for k,v in self._class_counts.items()]) < 1:
                self._empty = 1
                self._noisy_majority = random.choice([k for k,v in self._class_counts.items()])
                return 0 # we dont want to count purely random flips
            else:
                all_counts = sorted([v for k,v in self._class_counts.items()], reverse=True)
                v_0 = 0
                count_difference = all_counts[0] - all_counts[1]
                v = statistics.variance(self._class_counts)
                if (v < v_0):
                    self._sensitivity = math.exp(-1 * count_difference * epsilon) + self.medianMechanism(10,5,1000)
                    v_0 = v
                else:
                    self._sens_of_sens = 1.
                    self._noisy_sensitivity = 1
                    self._noisy_majority = self.expo_mech(epsilon, self._sensitivity, self._class_counts)

                if self._noisy_majority != int(max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))):
                    #print('majority: '+str(self._noisy_majority)+' vs. max_count: '+str( max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))))
                    return 1 # we're summing the flipped majorities
                else:
                    return 0
        else: return 0

    def laplace(self, e, counts):
        noisy_counts = {}
        for label,count in counts.items():
            noisy_counts[label] = max( 0, int(count + np.random.laplace(scale=float(1./(2*e)))) )
        return int(max(noisy_counts.keys(), key=(lambda key: noisy_counts[key])))

    def medianMechanism(self,e,s,cot): ##my propose method Sadat Bin Faruque
        noisy_counts = {}
        m = (160000 * math.log2(1/e) * cot)/e
        a = 1
        d = math.log2((2*s)/a)
        a1 = a/(720*m*math.log2(cot))
        tau = 4/(a1*e*s) * d
        new_counts = 0
        for label,count in np.arange(1,0.5,cot):
            ti = 3/4 + count
            noisy_counts[label] = max( 0, int(count + np.random.laplace(scale=float(tau))))
            if noisy_counts[label] < ti :
                new_counts+= count + np.random.laplace(scale=float(1/cot*a1))
        return new_counts

    def expo_mech(self, e, s, counts):       #used for median mechanism
        weighted = []
        max_count = max([v for k,v in counts.items()])
        
        for label,count in counts.items():
            if count == max_count:
                if s<1.0e-10: power = 50 # e^50 is already astronomical. sizes beyond that dont matter
                else: power = min( 50, (e*1)/(2*s) ) # score = 1
            else:
                power = 0 # score = 0
            weighted.append( [label, math.exp(power)] ) 
        sum = 0.
        for label,count in weighted:
            sum += count
        for i in range(len(weighted)):
            weighted[i][1] /= sum   
        customDist = stats.rv_discrete(name='customDist', values=([lab for lab,cou in weighted], [cou for lab,cou in weighted]))
        best = customDist.rvs()
        #print("best_att examples = "+str(customDist.rvs(size=20)))
        return int(best)
