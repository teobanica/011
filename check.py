"""Companion computations for Numeric evidence for the almost Hadamard conjecture."""
from itertools import combinations
import numpy as np
from scipy.linalg import expm, null_space, polar
from mpmath import iv

# Original decimal strings in the original row and column order.
SEEDS = {
    "first 5": ("""
0.1177558477 -0.0435010183 0.3519941135 0.4495841923 -0.4081154482
-0.1658955629 -0.1618508181 -0.4851537663 -0.4606498157 -0.3299830967
0.0091310513 -0.1112249007 0.1984510983 -0.3846833862 0.3111219828
-0.0823212152 0.2555804301 0.3510775314 -0.0482107019 0.5286654690
0.5134228737 -0.2750256902 -0.2711164268 0.3400190279 0.2776289120
""", """
-0.3769646420 -0.3878731899 0.3945558371 -0.2052307160 -0.0362669636
-0.3064960954 -0.4195694067 0.0941880816 0.2527678012 0.2169801871
0.4496128972 -0.5277157165 0.3360879217 -0.2329225368 -0.2356713328
-0.4873067768 -0.2949855746 -0.2112309458 0.3919752332 -0.0093540594
0.1117459925 -0.3558033782 -0.2871659310 -0.0764760274 0.4088604750
"""),
    "second 5": ("""
-0.3144104397 0.4021428975 -0.5104582969 -0.2376099818 -0.0487971305
0.3126946721 0.3719741451 -0.0234492398 -0.2424866115 0.3652137307
0.0621320665 -0.3550835598 0.0537564623 -0.3913907173 -0.3545787983
0.3644112867 -0.1362177237 -0.4277681157 0.3841666070 0.4939179962
0.2535061989 0.5374453422 -0.1783714283 0.1459875088 -0.4147688300
""", """
0.2210942287 0.1267481697 -0.0529505686 0.3021226284 -0.5108720265
0.4069317170 0.1985427718 0.3836492098 -0.4522960257 0.1198145198
0.5094222760 -0.2273756137 -0.3805874713 -0.3319407871 -0.1483592345
0.1222333700 -0.3990347570 -0.2835239513 0.0123532656 -0.1393429153
-0.3369252333 -0.0040606731 -0.3820571197 -0.3955649692 0.0758341632
"""),
    "6": ("""
0.2203292937 -0.1212061858 -0.1901093137 -0.3268648179 -0.3294313804 -0.3020175888
-0.2567938805 -0.4797377729 -0.3915441538 -0.1244197246 0.3343325884 0.0195524829
0.0154267664 0.3235510539 0.0056510381 0.3432282972 0.2830263976 -0.4444178680
-0.3941714879 0.4538217250 -0.3245883488 0.0514702220 -0.4865401914 0.2467354848
0.3283635917 0.1129589444 -0.2552264183 -0.3233170023 0.2973646357 -0.1830230807
-0.1933074529 -0.1524956466 0.4522327706 -0.4234882066 0.1065304451 0.1934227079
""", """
0.4220450798 0.3416454720 0.3790040486 -0.0558438591 0.1677384136 0.3525617469
-0.2148408576 -0.1388231552 -0.0501548827 0.3404884249 0.3669036688 0.3232124009
0.4622432302 -0.0127782647 -0.3824333498 0.3125838214 0.0795376108 0.1884296081
0.0168388336 -0.2011030290 0.1452018991 0.3660765339 -0.1081422984 0.1598492770
0.0498965174 -0.3151825002 0.3409090753 0.3494754766 -0.2592855934 -0.4247460996
0.3794623597 -0.3640980970 -0.0760897718 0.0210580890 -0.3392527660 0.3299618833
"""),
    "7": ("""
0.3719142129 -0.0232374313 -0.1332181138 0.0024925305 0.2343367769 0.0187888129 -0.0378651950
0.2151750385 -0.0318028201 0.2031926126 -0.3778623468 0.2761321315 0.4213540287 0.0639318662
-0.1584312709 0.3700977795 -0.2435333045 -0.5423872817 -0.2186807171 0.1572097300 0.1521211962
0.3893101502 -0.1979115771 0.3436164554 -0.2139833263 0.1229399944 -0.2995375897 -0.3091164076
0.2857583770 0.4260666019 0.2603104373 0.0911616394 0.0191771078 0.2546460598 0.1113315179
-0.2068954982 -0.2993758457 -0.0447117940 -0.0578549409 -0.1802542832 0.4955189897 -0.2998061914
-0.3288851458 0.1042383688 0.3464079061 0.2720573932 0.4264059171 0.3014483055 -0.2045866010
""", """
0.3008950630 0.1615818332 -0.3208946238 0.4014736952 -0.2381613111 0.2708880978 -0.5251512707
-0.0568324314 -0.4733722645 0.2936044862 0.1823395597 -0.3703526361 -0.0652736221 0.1502044646
-0.2870029355 0.1976241763 -0.1278346979 -0.0896686945 -0.1132379336 -0.3034472544 -0.3715460510
-0.3766059327 -0.1017901454 0.0381861214 -0.2483591660 0.3052919080 -0.0563523547 -0.3651096611
-0.1060212078 0.0163430645 -0.3673775132 0.3293701835 0.4951556226 -0.1471530717 0.2476664773
0.2559524550 -0.3518297068 -0.4241497517 -0.2393677000 0.2146003333 -0.0027960302 -0.1474813842
0.1066067098 0.3415042944 0.2279906995 -0.0404343459 -0.0089549425 -0.3344497792 -0.2808287407
"""),
}

def basis_for(n):
    """Frobenius-orthonormal Hermitian basis, with adjacent entries for each pair."""
    unit = np.eye(n)
    basis = [np.diag(row) for row in unit]
    for j, k in combinations(range(n), 2):
        e = np.outer(unit[j], unit[k])
        basis.extend([(e + e.T) / np.sqrt(2), 1j * (e - e.T) / np.sqrt(2)])
    return np.array(basis)


def derivatives(U, basis):
    phase = U / abs(U)
    X = phase.conj().T @ U
    Y = (X + X.conj().T) / 2
    projection = (U @ basis * phase.conj()).real / np.sqrt(abs(U))
    projection = projection.reshape(len(basis), -1)
    hessian = projection @ projection.T - np.einsum(
        "ij,ajk,bki->ab", Y, basis, basis, optimize=True).real
    gradient = -np.einsum("ij,aji->a", X, basis).imag
    return gradient, (hessian + hessian.T) / 2


def newton(U, basis):
    """Refine a seed already close to criticality, removing the phase directions."""
    n = len(U)
    for _ in range(8):
        g, H = derivatives(U, basis)
        if np.linalg.norm(g) < 1e-14:
            return U
        phases = np.concatenate((basis[:n], U.conj().T @ basis[1:n] @ U))
        coordinates = np.einsum("aij,kji->ka", basis, phases).real
        C = null_space(coordinates)
        step = -C @ np.linalg.solve(C.T @ H @ C, C.T @ g)
        U = U @ expm(1j * np.einsum("a,aij->ij", step, basis))
    raise RuntimeError("Newton refinement did not converge")


def best_pair(H, n):
    blocks = [H[j:j+2, j:j+2] for j in range(n, n*n, 2)]
    return max(np.linalg.eigvalsh(block)[-1] for block in blocks)


def conjugate(z):
    return iv.mpc(z.real, -z.imag)


def input_box(real, imag):
    delta = iv.mpf(["-1e-10", "1e-10"])
    return iv.matrix([[iv.mpc(iv.mpf(x) + delta, iv.mpf(y) + delta)
                       for x, y in zip(row_x, row_y)]
                      for row_x, row_y in zip(real, imag)])


def witness_bound(box, B):
    W = box * B
    Z = box * (B * B)
    Q = iv.mpf(0)
    for i in range(box.rows):
        for j in range(box.cols):
            z = box[i, j]
            u = conjugate(z)
            Q += (W[i, j] * u).real**2 / abs(z)**3 - (Z[i, j] * u).real / abs(z)
    return Q


def audit(label, parts):
    real, imag = [[line.split() for line in part.strip().splitlines()] for part in parts]
    A = np.array(real, float) + 1j * np.array(imag, float)
    n = len(A)
    basis = basis_for(n)
    U = newton(polar(A)[0], basis)
    g, H = derivatives(U, basis)
    w = np.linalg.eigvalsh(H)
    left = derivatives(U.conj().T, basis)[1]
    print(label, "relative norm", abs(U).sum() / n**1.5, "gradient", np.linalg.norm(g))
    print("  ascending spectrum", w)
    print("  inertia", np.sum(w < -1e-6), np.sum(abs(w) <= 1e-6), np.sum(w > 1e-6))
    print("  best column and row pairs", best_pair(H, n), best_pair(left, n))
    if label == "first 5":
        B = np.zeros((5, 5), complex)
        for j, k, value in [(0, 1, 1), (0, 3, -1j), (1, 3, -1+1j), (2, 3, -1+1j)]:
            B[j, k], B[k, j] = value, np.conj(value)
    else:
        # Choose a rational witness using the printed matrix. Only its sign
        # certificate below uses interval arithmetic; finding it can be numerical.
        H_printed = derivatives(A, basis)[1]
        top = np.linalg.eigh(H_printed)[1][:, -1]
        direction = np.einsum("a,aij->ij", top, basis)
        B = np.round(direction.real, 3) + 1j * np.round(direction.imag, 3)
    B_interval = iv.matrix([[iv.mpc(str(v.real), str(v.imag)) for v in row] for row in B])
    Q = witness_bound(input_box(real, imag), B_interval)
    assert Q > iv.mpf({"first 5": "0.11910039", "second 5": "0.6530",
                       "6": "0.6506", "7": "2.4955"}[label])
    print("  witness curvature on full decimal box", Q)


def exact_pairs():
    def derivative(x):
        return 3 - 4*x/iv.sqrt(1-x*x) - 2*(1-x)/iv.sqrt((1-x)**2+4)

    lo, hi = "0.53746068", "0.53746069"
    assert derivative(iv.mpf(lo)) > iv.mpf("1.27e-8")
    assert derivative(iv.mpf(hi)) < iv.mpf("-4.46e-8")
    x = iv.mpf([lo, hi])
    b = iv.sqrt(1-x*x)/2
    a = (1+x)/4
    z = iv.mpc((1-x)/4, "-0.5")
    first = [-a, z, -a, conjugate(z)]
    U = iv.matrix([[x, b, b, b, b]] +
                  [[b] + [first[(s-r) % 4] for s in range(4)] for r in range(4)])
    for j, k, low, high in [(0, 1, "-0.8502450", "-0.8502447"),
                            (1, 2, "-0.3694043", "-0.3694040"),
                            (1, 3, "-0.1978585", "-0.1978582")]:
        total, weight, phase_sum = iv.mpf(0), iv.mpf(0), iv.mpc(0)
        for i in range(5):
            v, w = U[i, j], U[i, k]
            p, q = abs(v), abs(w)
            d = q*q/p + p*p/q
            total += p+q
            weight += d
            phase_sum += d * ((conjugate(v)*w)/(p*q))**2
        bound = (weight + abs(phase_sum))/4 - total/2
        assert bound > iv.mpf(low) and bound < iv.mpf(high)
        print("exact model, pair", j, k, bound)




def hermitian(n, entries):
    B = iv.matrix(n)
    for j, k, real, imag in entries:
        B[j, k] = iv.mpc(real, imag)
        B[k, j] = conjugate(B[j, k])
    return B


def exact_saddle():
    lo = "0.53746068222727233271039478"
    hi = "0.53746068222727233271039479"
    def Fprime(x):
        return 3 - 4*x/iv.sqrt(1-x*x) - 2*(1-x)/iv.sqrt((1-x)**2+4)
    assert Fprime(iv.mpf(lo)) > 0 and Fprime(iv.mpf(hi)) < 0
    x = iv.mpf([lo, hi])
    b = iv.sqrt(1-x*x)/2
    a = (1+x)/4
    z = iv.mpc((1-x)/4, "-0.5")
    row = [-a, z, -a, conjugate(z)]
    U = iv.matrix([[x, b, b, b, b]] +
                  [[b] + [row[(s-r) % 4] for s in range(4)] for r in range(4)])
    triple = hermitian(5, [(1, 2, 0, 1), (2, 3, 0, 1)])
    q = witness_bound(U, triple)
    assert iv.mpf("0.5674395") < q and q < iv.mpf("0.5674400")
    print("exact order-five triple", q)

    generators = []
    for j, k in list(combinations(range(5), 2))[:8]:
        generators.extend([hermitian(5, [(j, k, 1, 0)]),
                           hermitian(5, [(j, k, 0, 1)])])
    diag = [witness_bound(U, B) for B in generators]
    H = iv.matrix(16)
    for i in range(16):
        H[i, i] = diag[i]
        for j in range(i):
            mixed = (witness_bound(U, generators[i] + generators[j]) - diag[i] - diag[j])/2
            H[i, j] = H[j, i] = mixed
    for k in range(16):
        pivot = H[k, k]
        print("Hessian pivot", k+1, pivot)
        if k == 13:
            assert iv.mpf("1.58") < pivot and pivot < iv.mpf("1.59")
        else:
            assert iv.mpf("-3.02") < pivot and pivot < iv.mpf("-0.07")
        for i in range(k+1, 16):
            for j in range(k+1, i+1):
                H[i, j] -= H[i, k] * H[j, k] / pivot
                H[j, i] = H[i, j]


def six_columns():
    real, imag = [[line.split() for line in part.strip().splitlines()] for part in SEEDS["6"]]
    B = hermitian(6, [(0, 3, 1, -1), (0, 5, 1, 1), (3, 5, -1, 1)])
    q = witness_bound(input_box(real, imag), B)
    assert iv.mpf("1.0604779") < q and q < iv.mpf("1.0604782")
    print("order-six triple on original decimal box", q)
    V = newton(polar(np.array(real, float) + 1j*np.array(imag, float))[0], basis_for(6))
    W = V[:, [3, 1, 5, 4, 0, 2]]
    W = W * (W[:1, :].conj()/abs(W[:1, :]))
    W = W * (W[:, :1].conj()/abs(W[:, :1]))
    print("symmetry residual", np.max(abs(W - W.T)))
    H = derivatives(V, basis_for(6))[1]
    pairs = list(combinations(range(6), 2))
    maxima = []
    for triple in combinations(range(6), 3):
        indices = []
        for pair in combinations(triple, 2):
            j = 6 + 2*pairs.index(pair)
            indices.extend([j, j+1])
        curvature = np.linalg.eigvalsh(H[np.ix_(indices, indices)])[-1]
        maxima.append((curvature, tuple(k+1 for k in triple)))
    print("positive triples", sum(v > 0 for v, t in maxima), "best", max(maxima))


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=110)
    iv.dps = 50
    exact_pairs()
    exact_saddle()
    six_columns()
    for label, parts in SEEDS.items():
        audit(label, parts)
