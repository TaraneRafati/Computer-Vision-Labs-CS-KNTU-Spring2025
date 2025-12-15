def mul(A,B):
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    if cols_a != rows_b:
        return []
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for r_a in range(rows_a):
        for c_b in range(cols_b):
            for i in range(cols_a): # cols_a = rows_b
                result[r_a][c_b] += A[r_a][i] * B[i][c_b]    
    return result

A = [[1, 0, 0],
    [0, 0, 3],
    [0, 2, 0]]
B = [[1, 1],
    [0, .5],
    [2, 1/3.0]]
C = [[ 1, 0, 0 ],
    [ 0, 0, 0.5],
    [ 0, 1/3.0, 0]]
print(mul(A,B))
print(mul(B,A))
print(mul(A,C))