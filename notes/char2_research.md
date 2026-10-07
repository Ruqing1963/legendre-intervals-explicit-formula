# Direction 2: characteristic-2 anomaly (research notes, not public)

## Established (in the draft, Section 8)

- Legendre intervals for even q are the subgroup of squares G^2 of the Hayes group, of index 2^{k * #odd j <= l}.
- Every P in I_f is uniquely h^2 + t b^2 with h = f + a, deg a <= floor(d/2), deg b <= floor((d-1)/2); P' = b^2; P squarefree iff gcd(h, b) = 1.
- Over all classes the variance of Psi follows d - 2 (KR) for even q too; the form factor over all characters matches CUE.
- Over G^2 the variance differs from the full-group value by factors 0.33 to 21 (q = 2, 4, 8; d <= 16, 9, 7).
- Var_{G^2} / Var_G = 1 + sum_{eps quadratic, eps != 1} rho(eps), matching to 4 decimals (code/char2_corr.py, data/char2_variance.csv).
- Large rho(eps) concentrate on quadratic characters supported on the axes j > l/2 (generators of order 2), equal across the F_2-basis directions of F_q.

## Refuted hypotheses (q = 2, d = 8, 10, 12; code/char2_probe.py)

- mu(h^2 + t b^2) = +-(-1)^{L(a)} with L affine over F_2 in the coefficients of a: no.
- mu(h^2 + t b^2) depends only on h mod b: no.

## Next steps

1. Identify the order-2 characters supported on axes j > l/2 concretely: for x = 1 + c_1 u + ..., they should depend only on the top coefficients c_j (j > l/2) through Tr_{F_q/F_2}(lambda c_j)-type expressions (check: (1 + omega u^j)^2 = 1 + omega^2 u^{2j} = 1 mod u^{l+1} when 2j > l).
2. Express psi_{2d}(chi eps) - psi_{2d}(chi) as a sum over primes P whose class lies outside ker(eps); relate ker(eps) to a parity condition on the coefficients c_j of P, j > l/2.
3. Compare with Swan's theorem / the Berlekamp discriminant (Carmon 2015): for P with P' = b^2 of low degree the parity of the number of prime factors is governed by Br(P); test whether Tr(Br(P)) correlates with the high coefficients c_j (j > l/2).
4. Literature: arXiv:1909.03778 (prime polynomial values of quadratic functions in short intervals), Carmon 2015, Conrad-Conrad-Gross (prime specialization in genus 0).
