import os
import argparse
import yaml
import random
import numpy as np
from scipy.special import comb

from Factory import set_problem
from WeightVector import das_dennis, determine_neighbor
from Population import init_pop, eval_pop
from ReferencePoint import init_ref_point, update_ref_point
from Mutation import lf_mutation, poly_mutation, fix_bound, levy
from Decomposition import tchebycheff
from PriorityFunctions import fixed_priority_values
from AdaptiveStrategy import evolve
from AdaptiveWeightAdjustment import calc_SL,update_EP,kNN_EP,WS_transform,delete_vector,add_vector,init_EP


###################
# parse arguments #
###################

parser = argparse.ArgumentParser()
parser.add_argument('params', type=argparse.FileType('r'))
parser.add_argument('seed', type=int)
args = parser.parse_args()

##################
# set parameters #
##################

params = yaml.safe_load(args.params)                                # read config file

random.seed(args.seed)
np.random.seed(args.seed)

output = params['output']                                           # set output of record in this run

prob_name = params['prob_name']                                     # set optimization problem
problem = set_problem(prob_name)
n_obj = params['n_obj']                                             # set number of objectives
n_var = params['n_var']                                             # set number of variables
xl = params['xl']                                                   # set boundary of variables
xu = params['xu']

sld_n_part = params['sld_n_part']                                   # set number of partitions

n_eval = params['n_eval']                                           # set maximum number of evaluation
n_pop = int( comb(n_obj + sld_n_part - 1, n_obj - 1) )              # compute population size

T = params['T']                                                     # set neighbor size
delta = params['delta']                                             # set probability to select parent from neighbor
nr = params['nr']                                                   # set maximum update counts for one offspring

alpha = params['alpha']                                             # set scaling factor of levy flight mutation
betal = params['betal']                                             # set lower bound for stability parameter of levy flight mutation
betau = params['betau']                                             # set upper bound 
etam = params['etam']                                               # set index parameter of polynomial mutation

alpha_for_param = params['alpha for param']                         # set scaling factor of levy flight to search parameters
beta_for_param = params['beta for param']                           # set stability factor of levy flight to search parameters
n_step = params['n_step']                                           # set number of generations to assess a parameter

wag = params['wag']                                             #set the iteration intervals of utilizing the adaptive weight vector adjustment strategy 
rate_evol = params['rate_evol']                                    #If set this param 0.8,  first 80% generation normal MOEA/D, rest 20% weight adjustment
rate_update_weight = ['rate_update_weight']               # If set this param 0.05, 5% of weight are updated
G_max = n_eval/n_pop                                           #max generation 

#################
# start program #
#################

os.makedirs(f'./{output}', exist_ok=True)                           # create a folder to include running results
os.makedirs(f'./{output}/history/', exist_ok=True)
os.makedirs(f'./{output}/history/{args.seed}', exist_ok=True)

W = das_dennis(sld_n_part, n_obj)                                   # generate a set of weight vectors
W = WS_transform(W)                                                  # vectors are modified to use Tchebycheff efficiently
B = determine_neighbor(W, T)                                        # determine neighbor
X = init_pop(n_pop, n_var, xl, xu)                                  # initialize a population
Y = eval_pop(X, problem)                                            # evaluate fitness
z = init_ref_point(Y)                                               # determine a reference point
EP = init_EP(X, Y, n_pop, n_obj)                                           #initialize External population

P = np.random.uniform(betal, betau, n_pop)                          # generate a set of stability parameters
P_parent = P[:]                                                     # initialize a list to store parameters in parent generations
P_offspring = np.full(n_pop, np.nan)                                # initialize a list to store parameters in offspring generations

I = np.zeros(n_pop)                                                 # initialize a set of indicator vaules
I_parent = I[:]                                                     # initialize a list to store indicators in parent generations
I_offspring = I[:]                                                  # initialize a list to store indicators in offspring generations

n_fe = n_pop

c_gen = 1
while n_fe < n_eval:
    result = np.hstack([Y, X])                                          # record objective values and decision variables
    np.savetxt(f'./{output}/history/{args.seed}/{c_gen}_paretos.csv', result)

    info_gen = np.hstack([n_fe, c_gen])                                 # record information (n_fe and c_gen) about the current generation
    np.savetxt(f'./{output}/history/{args.seed}/{c_gen}_info_gen.csv', info_gen)

    for i in np.random.permutation(n_pop):                              # traverse the population; randomly permuting solutions in the population set

        beta = P[i]

        # if (priority_values[i] >= np.random.uniform()):
        if True:
            n_fe += 1
            xi, fi = X[i, :], Y[i, :]                                   # get current individual and its fitness value

            if random.random() < delta:                                 # determine selection pool by probability
                pool = B[i, :]                                          # neighbor as the pool
            else:
                pool = np.arange(n_pop)                                 # population as the pool

            j = np.random.choice(pool)                                  # select a random individual from pool
            xj, fj = X[j, :], Y[j, :]

            xi_ = fix_bound( lf_mutation(xi, xj, alpha, beta), xl, xu ) # levy flight mutation
            xi_ = fix_bound( poly_mutation(xi_, etam, xl, xu), xl, xu ) # polynomial mutation
            fi_ = problem(xi_)                                          # evaluate offspring
            EP = update_EP(EP, np.array([fi_,xi_]), n_pop, n_obj)        # update External population

            z = update_ref_point(z, fi_)                                # update reference point

            nc = 0                                                      # initialize the update counter
            for k in np.random.permutation(len(pool)):                  # traverse the selection pool

                fk = Y[k, :]                                            # get k-th individual fitness
                wk = W[k, :]                                            # get k-th weight vector

                tch_ = tchebycheff(fi_, wk, z)                          # compute tchebycheff cost of offspring
                tchk = tchebycheff(fk, wk, z)                           # compute tchebycheff cost of parent

                if tch_ <= tchk:                                        # compare tchebycheff cost of offspring and parent
                    X[k] = xi_                                          # update parent
                    Y[k] = fi_
                    nc += 1                                             # cumulate the counter

                    I[i] += np.abs(tch_ - tchk) / tchk                  # cumulate the indicator

                if nc >= nr: break                                      # break if counter arrive the upper limit
                    
    if c_gen>=rate_evol*G_max and c_gen % wag == 0:      # If satisfy this fomula, start AWA
        nus = int(min(len(EP),rate_update_weight*len(Y)))    #number of update subproblem
        X,Y,W = delete_vector(X, Y, W, n_obj, nus)                   #delete vector
        X,Y,W = add_vector(EP, X, Y, W, z, nus)                         #add vector
        B = determine_neighbor(W, T)                                        # re-compute neighbor
    c_gen += 1

    P, P_parent, P_offspring, I, I_parent, I_offspring = \
        evolve(P, P_parent, P_offspring, I, I_parent, I_offspring,
               B, c_gen, n_step,
               betal, betau, alpha_for_param, beta_for_param)           # evolve the stability parameters



result = np.hstack([Y, X])                                              # record objective values and decision variables

np.savetxt(f'./{output}/history/{args.seed}/{c_gen}_paretos.csv', result)
np.savetxt(f'./{output}/{args.seed}_final.csv', result)

info_gen = np.hstack([n_fe, c_gen])                                      # record information (n_fe and c_gen) about the current generation
np.savetxt(f'./{output}/history/{args.seed}/{c_gen}_info_gen.csv', info_gen)
