import numpy as np
from sklearn.neighbors import NearestNeighbors

def calc_SL(Y, pop, n_objs, Y_is_pop=True):
    num = min(n_objs,len(Y))
    neigh = NearestNeighbors(num)
    Y = [item for item in Y]
    pop = [item for item in pop]
    neigh.fit(Y)
    neighbors = neigh.kneighbors(pop, num)[0].tolist()
    if Y_is_pop:
        SL = [np.prod(item[1:num]) for item in neighbors]
    else:
        SL = [np.prod(item[:num-1]) for item in neighbors]
    return SL

def update_EP(EP, solution, pop, n_obj):
    add_flag = False
    if len(EP)==0:
        return np.array([solution])
    remove_list = []
    L = len(EP)
    
    for i in range(L):
        flag = ParetoDominance(solution[0],EP[i][0])
        if flag == 1:
            return EP
        elif flag == -1:
            remove_list.append(EP[i])
            add_flag = True
        elif flag == 0:
            add_flag = True
            
    if len(EP)==0:
        add_flag = True

    if add_flag:
        EP = np.append(EP,[solution],axis=0)
    for item in remove_list:
        np.delete(EP,np.where(EP==item))

    new_EP = kNN_EP(EP, n_obj, pop)
    return new_EP

def kNN_EP(EP, n_objs, pop):
    while(len(EP)>pop*1.5):
        SL = calc_SL(EP[:, 0], EP[:, 0],n_objs)
        EP = np.delete(EP,np.argmin(SL),0)
    return EP.tolist()

def WS_transform(Vector,epsilon=10**9+7):
    small = 1/epsilon
    for i in range(len(Vector)):
        if sum(np.array(Vector[i])==0)==1:
            Vector[i] = [Vector[i][j]+small for j in range(len(Vector[i]))]
        elif sum(np.array(Vector[i])==0)==2:
            continue
        for j in range(len(Vector[i])):
            Vector[i][j] = 1/Vector[i][j]

    for i in range(len(Vector)):
        SUM = np.sum(Vector[i])
        Vector[i] = Vector[i]/SUM
    return Vector

def delete_vector(X, Y, W, n_objs, nus):
    for i in range(nus):
        SL = calc_SL(Y, Y, n_objs)
        X = np.delete(X,np.argmin(SL),0)
        Y = np.delete(Y,np.argmin(SL),0)
        W = np.delete(W,np.argmin(SL),0)
    return X,Y,W

def add_vector(EP, X, Y, W, ref_point, nus, epsilon = 10**-7):
    Vec_sp = []
    FV_sp = []
    ind_sp = []
    EP = np.array(EP)
    objs = EP[:, 0]
    individuals = EP[:, 1]
    SL = calc_SL(Y, EP[:,0], len(ref_point), False)
    for count in range(nus):
        F_sp = objs[np.argmax(SL)]
        F_sp_ideal = sum([1/(F_sp[i] - ref_point[i]+epsilon) for i in range(len(ref_point))])
        ind_sp.append(individuals[np.argmax(SL)])
        FV_sp.append(F_sp)
        Vec_sp.append([(1/(F_sp[i] - ref_point[i]+epsilon))/F_sp_ideal for i in range(len(ref_point))])
        SL.remove(max(SL))

    X = np.append(X,ind_sp,axis=0)
    Y = np.append(Y,FV_sp,axis=0)
    W = np.append(W,Vec_sp,axis=0)
    return X,Y,W

def init_EP(X, Y, n_pop, n_obj):
    EP = np.array([])
    for i in range(n_pop):
        EP = update_EP(EP,np.array([Y[i, :],X[i, :]]), n_pop, n_obj)
    return EP

def ParetoDominance(solution1,solution2):
    dominate1 = False
    dominate2 = False
    
    for i in range(len(solution1)):
        o1 = solution1[i]
        o2 = solution2[i]
        if o1 < o2:
            dominate1 = True

            if dominate2:
                return 0
        elif o1 > o2:
            dominate2 = True

            if dominate1:
                return 0
            
    if dominate1 == dominate2:
        return 0
    elif dominate1:
        return -1
    else:
        return 1
