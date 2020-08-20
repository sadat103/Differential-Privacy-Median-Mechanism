from collections import Counter, defaultdict
import random
import numpy as np
import math
import multiprocessing as multi
from sklearn.model_selection import train_test_split
import node 
import matplotlib.pyplot as plt


MULTI_THREAD = False # Turn this on if you would like to use multi-threading. Warning: if each tree builds too quickly, overhead time will be relatively large.

class DP_Random_Forest:    
    ''' Make a forest of Random Trees, then filter the training data through each tree to fill the leafs. 
    IMPORTANT: the first attribute of both the training and testing data MUST be the class attribute. '''
    def __init__(self, 
                train, # 2D list of the training data where the columns are the attributes, and the first column is the class attribute
                test, # 2D list of the testing data where the columns are the attributes, and the first column is the class attribute
                categs, # the indexes of the categorical (i.e. discrete) attributes, EXCLUDING the class attribute
                num_trees, # the number of trees to build. Since we divide the data among the trees, ensure: num_trees << len(train)
                epsilon, # the total privacy budget. The whole budget will be used in each tree (and thus each leaf), due to using disjoint data.
                ):
        ''' Some initialization '''
        self._categs = categs
        numers = [x+1 for x in range(len(train[0])-1) if x+1 not in categs] # the indexes of the numerical (i.e. continuous) attributes
        self._attribute_domains = self.get_attr_domains(train, numers, categs)
        attribute_indexes = [int(k) for k,v in self._attribute_domains.items()]
        self._max_depth = self.calc_tree_depth(len(numers), len(categs))
        self._num_trees = num_trees

        ''' Some bonus information gained throughout the algorithm '''
        self._missed_records = []
        self._flipped_majorities = []
        self._av_sensitivity = []
        self._empty_leafs = []

        random.shuffle(train)
        class_labels = list(set([int(x[0]) for x in train])) # domain of labels
        actual_labels = [int(x[0]) for x in test] # ordered list of the test data labels
        voted_labels = [defaultdict(int) for x in test] # initialize
        
        if MULTI_THREAD:        
            ''' PERSONALIZE THE NUMBER OF CORES TO USE '''
            num_threads = multi.cpu_count() - 2
            pool = multi.Pool(processes = num_threads)
            processes = []

        subset_size = int(len(train)/self._num_trees) # by using subsets, we don't need to divide the epsilon budget
        for i in range(self._num_trees):
            if MULTI_THREAD:
                processes.append(pool.apply_async(self.build_tree, (attribute_indexes, train[i*subset_size:(i+1)*subset_size], epsilon, class_labels, test)))
            else:
                results = self.build_tree(attribute_indexes, train[i*subset_size:(i+1)*subset_size], epsilon, class_labels, test)

                ''' Collect the predictions and the bonus information '''
                curr_votes = results['voted_labels']
                for rec_index in range(len(test)):
                    for lab in class_labels:
                        voted_labels[rec_index][lab] += curr_votes[rec_index][lab]
                self._missed_records.append(results['missed_records'])
                self._flipped_majorities.append(results['flipped_majorities'])
                self._av_sensitivity.append(results['av_sensitivity'])
                self._empty_leafs.append(results['empty_leafs'])

        if MULTI_THREAD:        
            for i in range(self._num_trees):
                print(str(i), end=' ')
                curr_votes = processes[i].get()['voted_labels']
                for rec_index in range(len(test)):
                    for lab in class_labels:
                        voted_labels[rec_index][str(lab)] += curr_votes[rec_index][str(lab)]
                self._missed_records.append(processes[i].get()['missed_records'])
                self._flipped_majorities.append(processes[i].get()['flipped_majorities'])
                self._av_sensitivity.append(processes[i].get()['av_sensitivity'])
                self._empty_leafs.append(processes[i].get()['empty_leafs'])
            pool.close()

        final_predictions = []
        for i,rec in enumerate(test):
            final_predictions.append( Counter(voted_labels[i]).most_common(1)[0][0] )
        print(final_predictions)
        print(actual_labels)
        counts = Counter([x == y for x, y in zip(final_predictions, actual_labels)])
        self._predicted_labels = final_predictions
        self._accuracy = float(counts[True]) / len(test)


    def get_attr_domains(self, data, numers, categs):
        attr_domains = {}
        transData = np.transpose(data)
        for i in categs:
            attr_domains[str(i)] = [str(x) for x in set(transData[i])]
            print("original domain length of categ att {}: {}".format(i, len(attr_domains[str(i)])))
        for i in numers:
            vals = [float(x) for x in transData[i]]
            attr_domains[str(i)] = [min(vals), max(vals)]
        return attr_domains

    def calc_tree_depth(self, num_numers, num_categs):
        if num_numers<1: # if no numerical attributes
            return math.floor(num_categs/2.) # depth = half the number of categorical attributes
        else:
            ''' Designed using balls-in-bins probability. See the paper for details. '''
            m = float(num_numers)
            depth = 0
            expected_empty = m # the number of unique attributes not selected so far
            while expected_empty > m/2.: # repeat until we have less than half the attributes being empty
                expected_empty = m * ((m-1.)/m)**depth
                depth += 1
            final_depth = math.floor(depth + (num_categs/2.)) # the above was only for half the numerical attributes. now add half the categorical attributes
            ''' WARNING: The depth translates to an exponential increase in memory usage. Do not go above ~15 unless you have 50+ GB of RAM. '''
            return min(15, final_depth) 
    
    def build_tree(self, attribute_indexes, train, epsilon, class_labels, test):
        root = random.choice(attribute_indexes)
        tree = Tree(attribute_indexes, root, self)
        tree.filter_training_data_and_count(train, epsilon, class_labels)
        missed_records = tree._missed_records
        flipped_majorities = tree._flip_fraction
        av_sensitivity = tree._av_sensitivity
        empty_leafs = tree._empty_leafs
        voted_labels = [defaultdict(int) for x in test]
        for i,rec in enumerate(test):
            label = tree.classify(tree._root_node, rec)
            voted_labels[i][label] += 1
        del tree
        return {'voted_labels':voted_labels, 'missed_records':missed_records, 'flipped_majorities':flipped_majorities, 
                'av_sensitivity':av_sensitivity, 'empty_leafs':empty_leafs}


class Tree(DP_Random_Forest):
    ''' Set the root for this tree and then start the random-tree-building process. '''
    def __init__(self, attribute_indexes, root_attribute, pc):
        self._id = 0
        self._categs = pc._categs
        self._max_depth = pc._max_depth
        self._num_leafs = 0

        root = node.node(None, None, root_attribute, 1, 0, []) # the root node is level 1
        attribute_domains = pc._attribute_domains
        
        if root_attribute not in self._categs: # numerical attribute
            split_val = random.uniform(attribute_domains[str(root_attribute)][0], attribute_domains[str(root_attribute)][1])
            left_domain = {k : v if k!=str(root_attribute) else [v[0], split_val] for k,v in attribute_domains.items() }
            right_domain = {k : v if k!=str(root_attribute) else [split_val, v[1]] for k,v in attribute_domains.items() }
            root.add_child( self.make_children([x for x in attribute_indexes], root, 2, '<'+str(split_val), split_val, left_domain) ) # left child
            root.add_child( self.make_children([x for x in attribute_indexes], root, 2, '>='+str(split_val), split_val, right_domain) ) # right child
        else: # categorical attribute
            for value in attribute_domains[str(root_attribute)]: 
                root.add_child( self.make_children([x for x in attribute_indexes if x!=root_attribute], root, 2, value, None, attribute_domains) ) # categorical attributes can't be tested again
        self._root_node = root
                
    ''' Recursively make all the child nodes for the current node, until a termination condition is met. '''
    def make_children(self, candidate_atts, parent_node, current_depth, splitting_value_from_parent, svfp_numer, attribute_domains):
        self._id += 1
        if not candidate_atts or current_depth >= self._max_depth+1: # termination conditions. leaf nodes don't count to the depth.
            self._num_leafs += 1
            return node.node(parent_node, splitting_value_from_parent, None, current_depth, self._id, None, svfp_numer=svfp_numer) 
        else:
            new_splitting_attr = random.choice(candidate_atts) # pick the attribute that this node will split on
            current_node = node.node(parent_node, splitting_value_from_parent, new_splitting_attr, current_depth, self._id, [], svfp_numer=svfp_numer) # make a new node

            if new_splitting_attr not in self._categs: # numerical attribute
                split_val = random.uniform(attribute_domains[str(new_splitting_attr)][0], attribute_domains[str(new_splitting_attr)][1])
                left_domain = {k : v if k!=str(new_splitting_attr) else [v[0], split_val] for k,v in attribute_domains.items() }
                right_domain = {k : v if k!=str(new_splitting_attr) else [split_val, v[1]] for k,v in attribute_domains.items() }
                current_node.add_child( self.make_children([x for x in candidate_atts], current_node, current_depth+1, '<', split_val, left_domain) ) # left child
                current_node.add_child( self.make_children([x for x in candidate_atts], current_node, current_depth+1, '>=', split_val, right_domain) ) # right child
            else: # categorical attribute
                for value in attribute_domains[str(new_splitting_attr)]: # for every value in the splitting attribute
                    child_node = self.make_children([x for x in candidate_atts if x!=new_splitting_attr], current_node, current_depth+1, value, None, attribute_domains)
                    current_node.add_child( child_node ) # add children to the new node
            return current_node


    ''' Record which leaf each training record belongs to, and then set the (noisy) majority label. '''
    def filter_training_data_and_count(self, records, epsilon, class_values):
        ''' epsilon = the epsilon budget for this tree (each leaf is disjoint, so the budget can be re-used). '''
        num_unclassified = 0.
        for rec in records:
            num_unclassified += self.filter_record(rec, self._root_node, class_index=0)
        self._missed_records = num_unclassified
        flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(epsilon, self._root_node, class_values, 0, 0, [])
        self._av_sensitivity = np.mean(sensitivities) # excludes empty leafs

        if self._num_leafs == 0:
            print("\n\n~~~ WARNING: NO LEAFS. num_unclassified = "+str(num_unclassified)+" ~~~\n\n")
            self._empty_leafs = -1.0
        else:
            self._empty_leafs = empty_leafs / float(self._num_leafs)

        if empty_leafs == self._num_leafs:
            print("\n\n~~~ WARNING: all leafs are empty. num_unclassified = "+str(num_unclassified)+" ~~~\n\n")
            self._flip_fraction = -1.0
        else:
            self._flip_fraction = flipped_majorities / float(self._num_leafs-empty_leafs)
    
    def filter_record(self, record, node, class_index=0):
        if not node:
            return 0.00001 # For debugging purposes. Doesn't happen in my experience
        if not node._children: # if leaf
            node.increment_class_count(record[class_index])
            return 0.
        else:
            child = None
            if node._splitting_attribute not in self._categs: # numerical attribute
                rec_val = record[node._splitting_attribute]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else: # categorical attribute
                rec_val = str(record[node._splitting_attribute])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None and node._splitting_attribute in self._categs: # if the record's value couldn't be found:
                #print(str([i._split_value_from_parent,])+" vs "+str([record[node._splitting_attribute],])+" out of "+str(len(node._children)))
                return 1. # For debugging purposes
            elif child is None: # if the record's value couldn't be found:
                return 0.001 # For debugging purposes
            return self.filter_record(record, child, class_index)

    def set_all_noisy_majorities(self, epsilon, node, class_values, flipped_majorities, empty_leafs, sensitivities):
        if node._children:
            for child in node._children:
                flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(
                                                        epsilon, child, class_values, flipped_majorities, empty_leafs, sensitivities)
        else:
            flipped_majorities += node.set_noisy_majority(epsilon, class_values)
            empty_leafs += node._empty
            if node._sensitivity>=0.0: sensitivities.append(node._sensitivity)
        return flipped_majorities, empty_leafs, sensitivities


    def classify(self, node, record):
        if not node:
            return None
        elif not node._children: # if leaf
            return node._noisy_majority
        else: # if parent
            attr = node._splitting_attribute
            child = None
            if node._splitting_attribute not in self._categs: # numerical attribute
                rec_val = record[attr]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else: # categorical attribute
                rec_val = str(record[attr])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None: # if the record's value couldn't be found, just return the latest majority value
                return node._noisy_majority #majority_value, majority_fraction

            return self.classify(child, record)


    
#plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
if __name__ == '__main__':
    data1 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/a.csv', delimiter = ',')
    #data2 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/b.csv', delimiter = ',')
    data3 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/soyabean.csv', delimiter = ',')
    data4 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/foresttypes.csv', delimiter = ',')
    data5 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/breast-cancer-wisconsin.csv', delimiter = ',')
    data6 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/data_banknote_authentication.csv', delimiter = ',')
    data7 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/heartdes.csv', delimiter = ',')
    data8 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/indian_liver_patient_dataset.csv', delimiter = ',')
    data9 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/iris_dataset.csv', delimiter = ',')
    data10 = np.genfromtxt('/home/sadat/Documents/ThesisCode/Thesis/Deferentially private decision tree/seeds_dataset.csv', delimiter = ',')
    for j in np.arange(0.2, 0.8, 0.1):
        train_data1 , test_data1 = train_test_split(data1,test_size=j)
        #train_data2 , test_data2 = train_test_split(data2,test_size=j)
        train_data3 , test_data3 = train_test_split(data3,test_size=j)
        train_data4 , test_data4 = train_test_split(data4,test_size=j)
        train_data5 , test_data5 = train_test_split(data5,test_size=j)
        train_data6 , test_data6 = train_test_split(data6,test_size=j)
        train_data7 , test_data7 = train_test_split(data7,test_size=j)
        train_data8 , test_data8 = train_test_split(data8,test_size=j)
        train_data9 , test_data9 = train_test_split(data9,test_size=j)
        train_data10 , test_data10 = train_test_split(data10,test_size=j)
    
        budget_list = []
        accuracy_list = []
        for i in np.arange(3, 20, 1):
            #forest1forest1 = DP_Random_Forest(train_data1[0:15], test_data1, [16,], 2, 0.1)
            #forest2 = DP_Random_Forest(train_data2[1:4], test_data2, [0,], 2, 0.1)
            #forest3 = DP_Random_Forest(train_data3[1:35], test_data3, [0,], 6, i)
            #forest4 = DP_Random_Forest(train_data4[1:27], test_data4, [0,], 2, 0.1)
            forest5 = DP_Random_Forest(train_data5[1:10], test_data5, [0,], 12, i)
            #forest6 = DP_Random_Forest(train_data6[1:4], test_data6, [0,], 12, i)
            #forest7 = DP_Random_Forest(train_data7[1:13], test_data7, [0,], 6, i)
            #forest8 = DP_Random_Forest(train_data8[1:10], test_data8, [0,], 15,i)
            #forest9 = DP_Random_Forest(train_data9[1:4], test_data9, [0,], 8, i)
            #forest10 = DP_Random_Forest(train_data10[1:7],test_data10, [0,], 15, i)
            budget_list.append(i)
            accuracy_list.append(forest5._accuracy*100)
            print('accuracy = '+str(forest5._accuracy*100))
        print(budget_list)
        print(accuracy_list)
        #print('accuracy = '+str(forest2._accuracy*100))
        #%matplotlib inline
        plt.rcParams['figure.figsize'] = 7, 5
        plt.locator_params(axis = 'x', nbins = 5)
        plt.plot(budget_list, accuracy_list, 'b-', linewidth=3.0, color = '#B0017F')
        plt.title('budget-accuracy curve for fixed trees and ratio' +str(j))
        plt.xlabel('budget')
        plt.ylabel('accuracy')
        plt.rcParams.update({'font.size': 10})
        plt.show()
