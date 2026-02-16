using Reduce

# Formal indexed objects as function calls (dimension-free)
ex = :( SUM( A(i,k)*B(k,j), k, 1, n ) + SUM( A(i,k)*B(k,j), k, 1, n ) )

rcall(ex)
# often simplifies to: 2*SUM(A(i,k)*B(k,j), k, 1, n)