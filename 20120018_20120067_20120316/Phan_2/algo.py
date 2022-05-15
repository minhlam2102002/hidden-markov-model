import numpy as np

# Forward Algorithm 
def Forward(O, A, B, pi):
    M = O.shape[0]
    N = A.shape[0]
    res = np.zeros((M, N))
    
    # khoi tao xac suat cua trang thai ban dau
    res[0, :] = pi * B[:, O[0]]

    for i in range(1, M):
        for j in range(N):
            res[i, j] = res[i - 1].dot(A[:, j]) * B[j, O[i]]
    return res

# Viterbi Algorithm
def Viterbi(O, A, B, pi):
    T = O.shape[0]
    M = A.shape[0]
 
    table_dp1 = np.zeros((T, M))
    table_dp2 = np.zeros((T - 1, M))
    table_dp1[0, :] = pi * B[:, O[0]]
    path = np.zeros(T, dtype=int)
 
    for i in range(1, T):
        for j in range(M):
            tmp = table_dp1[i - 1] + A[:, j] + B[j, O[i]]
            table_dp1[i, j] = np.max(tmp)
            table_dp2[i - 1, j] = np.argmax(tmp)
 
    path[-1] = np.argmax(table_dp1[-1, :])
    
    for i in range(T - 2, -1, -1):
        path[i] = table_dp2[i, path[i+1]]

    return path

# ham bo tro cho Baum-Welch algorithm
def Backward(O, A, B):
    T = O.shape[0]
    M = A.shape[0]
    res = np.zeros((T, M))

    res[T - 1] = np.ones((M))

    for i in range(T - 2, -1, -1):
        for j in range(M):
            res[i, j] = (res[i + 1] * B[:, O[i + 1]]).dot(A[j, :])

    return res

# Giai thuat Baum-Welch rat kho, nhom em chi co the doc hieu ve phan toan va pseudo code chu khong
# the tu lam 100%, nen da dua theo tai lieu tham khao duoc ghi chu trong file report.
# Baum-Welch Algorithm
def Baum_welch(V, A, B, pi, n=100):
    M = A.shape[0]
    T = len(V)
 
    for n in range(n):
        #tinh toan alpha va beta dua tren forward va backward algorithm
        alpha = Forward(V, A, B, pi)
        beta = Backward(V, A, B)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            den = np.dot(np.dot(alpha[t, :].T, A) * B[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                num = alpha[t, i] * A[i, :] * B[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = num / den
 
        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = B.shape[1]
        den = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        B = np.divide(B, den.reshape((-1, 1)))
 
    return {"A":A, "B":B}