#!/usr/bin/env python3
"""Certify exact critical points near the 40 retained U5 representatives.

python3 critical5.py
python3 critical5.py --prepare [--groups 39]

The default reads rational chart data and performs only interval verification.
--prepare proposes those data from the existing catalogue, using point Newton
iteration; proposed data have no certified status until the interval tests pass.
No random search is performed. The input and output JSON files live in data/.
Uniqueness is local modulo row and column phases. No completeness or pairwise
inequivalence assertion is made by this program.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
from mpmath import mp,iv

ROOT=Path(__file__).resolve().parent
N,D=5,16
DENOMINATOR=10**20
RADIUS='1e-12'
BASIS=[np.diag(e).astype(complex) for e in np.eye(N)]
for j,k in combinations(range(N),2):
    for value in (1,1j):
        a=np.zeros((N,N),complex);a[j,k]=value;a[k,j]=complex(value).conjugate()
        BASIS.append(a)
BASIS=np.array(BASIS)

def conjugate(c,z):
    return c.mpc(z.real,-z.imag)

def star(c,a):
    return c.matrix([[conjugate(c,a[j,i]) for j in range(a.rows)] for i in range(a.cols)])

def coords(a):
    return [a[k,k].real for k in range(N)]+[v for j,k in combinations(range(N),2)
                                                   for v in (a[j,k].real,a[j,k].imag)]

def evaluate(c,a,hs,second=True):
    """Value, gradient and ordinary coordinate Hessian of sum |U_ij|."""
    w=(c.eye(N)-1j*a)**-1;u=2*w-c.eye(N)
    du=[2j*w*b*w for b in hs]
    mod=[[abs(u[i,j]) for j in range(N)] for i in range(N)]
    radial=[[[(conjugate(c,u[i,j])*v[i,j]).real for j in range(N)] for i in range(N)] for v in du]
    g=c.matrix([sum(radial[k][i][j]/mod[i][j] for i in range(N) for j in range(N)) for k in range(D)])
    if not second:return u,du,mod,g
    h=c.matrix(D)
    for j in range(D):
        for k in range(j,D):
            dd=-2*w*(hs[j]*w*hs[k]+hs[k]*w*hs[j])*w
            h[j,k]=h[k,j]=sum(
                (conjugate(c,du[j][a,b])*du[k][a,b]+conjugate(c,u[a,b])*dd[a,b]).real/mod[a][b]
                -radial[j][a][b]*radial[k][a][b]/mod[a][b]**3
                for a in range(N) for b in range(N))
    return u,du,mod,g,h

def prepare(j,u):
    from scipy.linalg import qr,polar
    # A rational unit phase moves the Cayley singularity away from the spectrum.
    choices=range(-16,17)
    k=max(choices,key=lambda k:np.linalg.svd(((4+1j*k)/(4-1j*k))*u+np.eye(N),compute_uv=False)[-1])
    alpha=(4+1j*k)/(4-1j*k)
    v=polar(alpha*u)[0]
    a=-1j*(v-np.eye(N))@np.linalg.inv(v+np.eye(N));a=(a+a.conj().T)/2
    w=np.linalg.inv(np.eye(N)-1j*a)
    du=2j*w@BASIS@w
    chart=np.array([coords(-1j*v.conj().T@b) for b in du],float).T
    phase=np.array([coords(b) for b in list(BASIS[:N])+[v.conj().T@e@v for e in BASIS[:N-1]]],float).T
    phase_q=np.linalg.qr(phase)[0]
    projected=chart-phase_q@(phase_q.T@chart)
    indices=sorted(int(i) for i in qr(projected,pivoting=True)[2][:D])
    mp.dps=60
    a=mp.matrix([[mp.mpc(str(z.real),str(z.imag)) for z in row] for row in a])
    hs=[mp.matrix(BASIS[i].tolist()) for i in indices]
    for step in range(8):
        *_,g,h=evaluate(mp,a,hs)
        if max(abs(x) for x in g)<mp.mpf('1e-50'):break
        correction=mp.lu_solve(h,g)
        for x,b in zip(correction,hs):a-=x*b
    else:raise AssertionError(('point Newton did not converge',j))
    integers=[[0]*N for _ in range(N)]
    for i in range(N):integers[i][i]=int(mp.nint(a[i,i].real*DENOMINATOR))
    for i,l in combinations(range(N),2):
        integers[i][l]=int(mp.nint(a[i,l].real*DENOMINATOR))
        integers[l][i]=int(mp.nint(a[i,l].imag*DENOMINATOR))
    positive=sum(x>0 for x in mp.eigsy(h,eigvals_only=True))
    return dict(group=j,phase_parameter=k,denominator=str(DENOMINATOR),
                center=integers,directions=[i+1 for i in indices],radius=RADIUS,positive=int(positive))

def center(c,record):
    a=c.matrix(N);ints=record['center'];den=int(record['denominator'])
    for i in range(N):a[i,i]=c.mpf(ints[i][i])/den
    for i,j in combinations(range(N),2):
        a[i,j]=c.mpc(ints[i][j],ints[j][i])/den;a[j,i]=conjugate(c,a[i,j])
    return a

def midpoint(a):
    return np.array([[float(a[i,j].mid) for j in range(a.cols)] for i in range(a.rows)])

def rational(a):
    return iv.matrix([[format(float(v),'.17g') for v in row] for row in a])

def infinity(a):
    return max(sum(abs(a[i,j]) for j in range(a.cols)).b for i in range(a.rows))

def pair_bounds(u):
    result=[]
    for which,v in [('column',u),('row',u.T)]:
        for j,k in combinations(range(N),2):
            total=iv.mpf(0);weight=iv.mpf(0);phase=iv.mpc(0)
            for i in range(N):
                p=abs(v[i,j]);q=abs(v[i,k]);z=conjugate(iv,v[i,j])*v[i,k]/(p*q)
                d=q*q/p+p*p/q;total+=p+q;weight+=d;phase+=d*z*z
            bound=(weight+abs(phase))/4-total/2
            assert bound.b<iv.mpf('-.22')
            result.append(dict(kind=which,pair=[j+1,k+1],lower=float(bound.a),upper=float(bound.b)))
    return result

def certificate(record,representative):
    iv.dps=50
    radius=iv.mpf(record['radius']);a0=center(iv,record)
    hs=[iv.matrix(BASIS[i-1].tolist()) for i in record['directions']]
    a=a0.copy()
    for h in hs:a+=iv.mpf([-radius.b,radius.b])*h
    u,du,mod,g,h=evaluate(iv,a,hs)
    u0,_,_,g0,h0=evaluate(iv,a0,hs)
    hm=midpoint(h0)
    y=rational(np.linalg.inv(hm))
    contraction=infinity(iv.eye(D)-y*h);eta=infinity(y*g0)
    assert contraction<iv.mpf('.002')
    assert eta<iv.mpf('1e-20') and eta+contraction*radius<radius
    # Floating eigenvectors propose a rational congruence. Its interval error
    # certifies the inertia; floating eigenvalue signs alone prove nothing.
    values,vectors=np.linalg.eigh(hm)
    signs=np.where(values>0,1,-1);positive=int(sum(signs>0))
    assert positive==record['positive'] and 0<positive<D
    p=rational(vectors/np.sqrt(abs(values))[None,:])
    signature=iv.diag([int(v) for v in signs])
    error=infinity(p.T*h*p-signature)
    assert error<iv.mpf('.001')
    tangents=[-1j*star(iv,u)*v for v in du]
    for e in BASIS[:N]:tangents.append(iv.matrix(e.tolist()))
    for e in BASIS[:N-1]:tangents.append(star(iv,u)*iv.matrix(e.tolist())*u)
    t=iv.matrix([coords(b) for b in tangents]).T
    z=rational(np.linalg.inv(midpoint(t)))
    transverse=infinity(iv.eye(N*N)-z*t)
    assert transverse<iv.mpf('1e-6')
    lower=min(v.a for row in mod for v in row)
    norm=sum(v for row in mod for v in row)
    assert lower>iv.mpf('.016') and norm.b<iv.mpf('11.1')
    rounded=format(float(abs(representative).sum()),'.6f')
    assert norm.a>iv.mpf(rounded)-iv.mpf('5e-7') and norm.b<iv.mpf(rounded)+iv.mpf('5e-7')
    k=record['phase_parameter'];alpha=iv.mpc(4,k)/iv.mpc(4,-k)
    target=iv.matrix([[iv.mpc(float(v.real),float(v.imag)) for v in row] for row in representative])
    distance=iv.sqrt(sum(abs(u[i,j]/alpha-target[i,j])**2 for i in range(N) for j in range(N)))
    assert distance.b<iv.mpf('1e-7')
    result=dict(group=record['group'],result='passed',positive=positive,
                inertia=[D-positive,2*N-1,positive],radius=record['radius'],
                contraction_upper=float(contraction),center_correction_upper=float(eta),
                signature_error_upper=float(error),transversality_error_upper=float(transverse),
                min_entry_lower=float(lower),L_interval=[float(norm.a),float(norm.b)],rounded_L=rounded,
                distance_to_phase_rotated_representative_upper=float(distance.b))
    if record['group']==39:
        assert norm.a>iv.mpf('11.08108') and norm.b<iv.mpf('11.08109')
        result['pair_curvature_intervals']=pair_bounds(u)
    return result

def derivative_check(record):
    # A floating finite-difference check is separate from the interval proof.
    mp.dps=30;a0=center(mp,record)
    hs=[mp.matrix(BASIS[i-1].tolist()) for i in record['directions']]
    h=np.array(evaluate(mp,a0,hs)[-1].tolist(),float)
    eps=mp.mpf('1e-5')
    fd=np.column_stack([np.array(((evaluate(mp,a0+eps*b,hs,False)[-1]-
                                 evaluate(mp,a0-eps*b,hs,False)[-1])/(2*eps)).tolist(),float).ravel()
                        for b in hs])
    error=float(np.max(abs(fd-h)))
    assert error<1e-6
    return error

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--groups',help='comma-separated group numbers; default all 40')
    args=parser.parse_args()
    groups=[int(j) for j in args.groups.split(',')] if args.groups else list(range(1,41))
    assert len(set(groups))==len(groups) and all(1<=j<=40 for j in groups)
    matrices=np.load(ROOT/'data/catalogue5.npz')['matrices']
    inputs=ROOT/'data/critical5-input.json';output=ROOT/'data/critical5.json'
    records=json.loads(inputs.read_text())['charts'] if inputs.exists() else []
    records={r['group']:r for r in records}
    result=dict(result='running',scope='Local existence, uniqueness modulo phases and inertia; no completeness or inequivalence claim.',
                precision_decimal_digits=50,certificates=[])
    start=time.monotonic()
    for j in groups:
        if args.prepare:
            records[j]=prepare(j,matrices[j-1])
            inputs.write_text(json.dumps(dict(basis_indexing='1-based: diagonal, then real/imaginary pairs in lexicographic order',
                                             charts=[records[k] for k in sorted(records)]),indent=2)+'\n')
        report=certificate(records[j],matrices[j-1])
        if j==39:report['finite_difference_discrepancy']=derivative_check(records[j])
        result['certificates'].append(report)
        result['seconds']=time.monotonic()-start
        output.write_text(json.dumps(result,indent=2)+'\n')
        print('group',j,'inertia',report['inertia'],'q',format(report['contraction_upper'],'.3g'),
              'eta/r',format(report['center_correction_upper']/float(RADIUS),'.3g'),
              'signature',format(report['signature_error_upper'],'.3g'),flush=True)
    result['result']='passed';result['certified_groups']=groups
    result['distinct_rounded_critical_values']=len({c['rounded_L'] for c in result['certificates']})
    if len(groups)==40:assert result['distinct_rounded_critical_values']==35
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('All',len(groups),'requested critical-point certificates passed in',round(result['seconds'],2),'seconds.',flush=True)

if __name__=='__main__':main()
