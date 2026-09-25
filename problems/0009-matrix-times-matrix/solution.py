def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    m_a = len(a)
    n_a = len(a[0])
    m_b = len(b)
    n_b = len(b[0])

    if n_a != m_b:
        return  -1
    out = [[0]*n_b for _ in range(m_a)]
    for i in range(m_a):
        for j in range(n_b):
            row = a[i]
            col = [b[c][j] for c in range(n_b)]
            res = sum(x*y for x,y in zip(row,col))
            # res = sum((map(lambda x, y: x * y, row, col)))
            out[i][j] = res
    return  out