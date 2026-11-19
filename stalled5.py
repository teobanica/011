#!/usr/bin/env python3
"""Revisit all 288 retained unresolved U5 endpoints; certify a residual trap.

python3 stalled5.py
Requires NumPy and mpmath. No new random search is performed. The frozen Mac
archive supplies all starts for this follow-up. Results and refined matrices
are written to data/stalled5.json and data/stalled5.npz. Run reproduce.py for
an isolated temporary-copy check that leaves the delivered evidence untouched.

The certificate proves a strict positive local minimum, modulo phases, of
Phi=||R||_F^2/2, R=(S*U-U*S)/(2i). It is not a local maximum of L. Numerical
basin assignments and equivalences are distinguished from this interval proof.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
from collections import Counter
from functools import lru_cache
from itertools import combinations,permutations
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile
import numpy as np
from mpmath import iv
import ahc5_stationary as p

ROOT=Path(__file__).resolve().parent
N,D=5,16
C=np.array([
 [495152000271,190465601450,-262733011052,-124814484784,-323386252325],
 [-377050587335,-448869470372,70859178042,-55846027846,1182451590433],
 [585209765500,-541232190289,464444926073,60765711274,-107072278033],
 [74541284483,-126095931572,461662797924,455696731011,445244851191],
 [48142427836,-53078770252,237226558258,964201483854,-530101168667]],dtype=np.int64)
# Canonical Hermitian basis: five diagonals, then real/imaginary pairs.
# Omit the diagonals and real (1,3), real (2,3), imaginary (2,5), real (4,5),
# with the pair labels here one-based. The remaining sixteen form our chart.
INDICES=[k for k in range(5,25) if k not in (7,13,18,23)]
RADIUS='1e-10'
PERMS=np.array(list(permutations(range(N))))


def rational_center():
    a=np.diag(np.diag(C)).astype(complex)
    for j,k in combinations(range(N),2):
        a[j,k]=complex(int(C[j,k]),int(C[k,j]));a[k,j]=a[j,k].conjugate()
    return a*1e-12


def chart_directions():
    basis=[np.diag(e).astype(complex) for e in np.eye(N)]
    for j,k in combinations(range(N),2):
        for value in (1,1j):
            b=np.zeros((N,N),complex);b[j,k]=value;b[k,j]=complex(value).conjugate()
            basis.append(b)
    return np.array(basis)[INDICES]


def dephase(v):
    v=v*np.conj(v[..., :,0]/abs(v[..., :,0]))[..., :,None]
    return v*np.conj(v[...,0,:]/abs(v[...,0,:]))[...,None,:]


def candidates(u):
    return dephase(u[PERMS[:,None,:,None],PERMS[None,:,None,:]].reshape(-1,N,N))


def match(u,orbit):
    v=dephase(u)
    errors=np.stack([np.linalg.norm(orbit-v,axis=(1,2)),
                     np.linalg.norm(orbit-v.conj(),axis=(1,2))])
    conjugate,k=np.unravel_index(np.argmin(errors),errors.shape)
    return dict(error=float(errors[conjugate,k]),conjugate=bool(conjugate),
                rows=PERMS[k//len(PERMS)].tolist(),columns=PERMS[k%len(PERMS)].tolist())


def certificate():
    """Outward-rounded contraction, positivity and transversality checks."""
    iv.dps=40
    eye=iv.eye(N)
    def conj(z):return iv.mpc(z.real,-z.imag)
    def star(a):return iv.matrix([[conj(a[j,i]) for j in range(a.rows)] for i in range(a.cols)])
    def inner(a,b):return sum((conj(a[i,j])*b[i,j]).real for i in range(N) for j in range(N))
    a0=iv.matrix(N)
    for j in range(N):a0[j,j]=iv.mpf(int(C[j,j]))/10**12
    for j,k in combinations(range(N),2):
        a0[j,k]=iv.mpc(int(C[j,k]),int(C[k,j]))/10**12;a0[k,j]=conj(a0[j,k])
    hs=[iv.matrix(v.tolist()) for v in chart_directions()]
    radius=iv.mpf(RADIUS)

    def evaluate(box,second=True):
        a=a0.copy()
        if box:
            for h in hs:a+=iv.mpf([-radius.b,radius.b])*h
        c=(eye-1j*a)**-1;u=2*c-eye
        du=[2j*c*h*c for h in hs]
        mod=[[abs(u[i,j]) for j in range(N)] for i in range(N)]
        s=iv.matrix([[u[i,j]/mod[i][j] for j in range(N)] for i in range(N)])
        dr=[];ds=[]
        for v in du:
            rho=iv.matrix([[(conj(u[i,j])*v[i,j]).real/mod[i][j]
                            for j in range(N)] for i in range(N)])
            dr.append(rho)
            ds.append(iv.matrix([[v[i,j]/mod[i][j]-u[i,j]*rho[i,j]/mod[i][j]**2
                                  for j in range(N)] for i in range(N)]))
        x=star(s)*u;r=(x-star(x))/(2j)
        dx=[star(ds[k])*u+star(s)*du[k] for k in range(D)]
        rr=[(v-star(v))/(2j) for v in dx]
        gradient=iv.matrix([inner(r,v) for v in rr])
        if not second:return gradient
        hessian=iv.matrix(D)
        for j in range(D):
            for k in range(j,D):
                dd=-2*c*(hs[j]*c*hs[k]+hs[k]*c*hs[j])*c
                dds=iv.matrix(N)
                for aa in range(N):
                    for bb in range(N):
                        v=u[aa,bb];rho=mod[aa][bb];vj=du[j][aa,bb];vk=du[k][aa,bb]
                        rj=dr[j][aa,bb];rk=dr[k][aa,bb]
                        rjk=(conj(vj)*vk+conj(v)*dd[aa,bb]).real/rho-rj*rk/rho
                        dds[aa,bb]=(dd[aa,bb]/rho-(vj*rk+vk*rj+v*rjk)/rho**2
                                    +2*v*rj*rk/rho**3)
                xx=star(dds)*u+star(ds[j])*du[k]+star(ds[k])*du[j]+star(s)*dd
                rjk=(xx-star(xx))/(2j)
                hessian[j,k]=hessian[k,j]=inner(rr[j],rr[k])+inner(r,rjk)
        return u,du,r,gradient,hessian,mod

    def midpoint(a):return np.array([[float(a[i,j].mid) for j in range(a.cols)] for i in range(a.rows)])
    def rational(a):return iv.matrix([[format(v,'.16g') for v in row] for row in a])
    def infinity(a):return max(sum(abs(a[i,j]) for j in range(a.cols)).b for i in range(a.rows))
    u,du,r,g,h,mod=evaluate(True)
    center=evaluate(False);center_g=center[3];center_h=midpoint(center[4])
    hm=midpoint(h)
    # Floating calculations propose rational witnesses; interval inequalities alone
    # validate them. No floating inverse or eigenvalue is used as a certificate.
    y=rational(np.linalg.inv(hm))
    q=infinity(iv.eye(D)-y*h);eta=infinity(y*center_g)
    ch=np.linalg.cholesky(hm);pre=rational(np.linalg.inv(ch.T))
    positive=infinity(star(pre)*h*pre-iv.eye(D))
    bs=[-1j*star(u)*v for v in du]
    for k in range(N):
        e=iv.matrix(N);e[k,k]=1;bs.append(e)
    for k in range(N-1):
        e=iv.matrix(N);e[k,k]=1;bs.append(star(u)*e*u)
    def coords(b):
        return [b[k,k].real for k in range(N)]+[v for j,k in combinations(range(N),2)
                                                  for v in (b[j,k].real,b[j,k].imag)]
    t=iv.matrix([coords(b) for b in bs]).T
    yt=rational(np.linalg.inv(midpoint(t)));transverse=infinity(iv.eye(N*N)-yt*t)
    normr=iv.sqrt(inner(r,r));minentry=min(v.a for row in mod for v in row)
    assert q<iv.mpf('.077') and eta<iv.mpf('7e-13') and eta+q*radius<radius
    assert positive<iv.mpf('.075') and transverse<iv.mpf('.000022')
    assert normr.a>iv.mpf('.01647') and normr.b<iv.mpf('.01649')
    assert minentry>iv.mpf('.1273')

    # Separate floating-point derivative check of the analytic Hessian formula.
    aa=rational_center();hh=chart_directions()
    def numeric_gradient(x):
        cc=np.linalg.inv(np.eye(N)-1j*(aa+np.einsum('k,kij->ij',x,hh)))
        v=2*cc-np.eye(N);dv=2j*cc@hh@cc
        rho=abs(v);ss=v/rho
        dss=dv/rho-v*(np.conj(v)*dv).real/rho**3
        dx=p.adj(dss)@v+p.adj(ss)@dv
        rr=(dx-p.adj(dx))/(2j);res=p.components(v)[2]
        return np.einsum('ij,kij->k',res.conj(),rr).real
    discrepancies=[]
    for eps in (2e-6,5e-7):
        fd=np.column_stack([(numeric_gradient(eps*e)-numeric_gradient(-eps*e))/(2*eps)
                            for e in np.eye(D)])
        discrepancies.append(float(np.max(abs(fd-center_h))))
    assert max(discrepancies)<1e-6
    return dict(result='passed',coordinate_radius=RADIUS,precision_decimal_digits=40,
                contraction_norm_upper='.077',center_correction_upper='7e-13',
                positive_congruence_error_upper='.075',transversality_error_upper='.000022',
                gradient_L_interval=['.01647','.01649'],min_entry_lower='.1273',
                finite_difference_discrepancies=discrepancies,
                conclusion='A strict positive local minimum of Phi modulo row/column phases exists '
                           'inside the rational Cayley chart box. It is not a critical point of L.')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ROOT/'overnight5/stationary-mac.zip') as archive:
            assert all('/' not in n and '\\' not in n for n in archive.namelist())
            archive.extractall(tmp)
        db=sqlite3.connect('file:'+str(Path(tmp)/'results.sqlite')+'?mode=ro',uri=True)
        rows={i:(json.loads(info),np.frombuffer(blob,dtype='<c16').reshape(N,N).copy())
              for i,info,blob in db.execute('select id,info,endpoint from kept order by id')}
        db.close()
    ids=np.array([i for i,(m,_) in rows.items() if m['status']=='unresolved'])
    assert len(ids)==288
    initial=np.array([rows[int(i)][1] for i in ids])
    refined,steps=p.refine(initial,limit=1600,tolerance=1e-12)
    ms=[p.metrics(v) for v in refined]
    critical=[k for k,m in enumerate(ms) if m['gradient']<1e-10]
    traps=[k for k,m in enumerate(ms) if abs(m['gradient']-.016477574426)<1e-8
           and abs(m['L']-10.2696684693)<1e-7]
    boundary=[k for k,m in enumerate(ms) if m['min_modulus']<1e-5 and k not in critical+traps]
    assert (len(critical),len(traps),len(boundary))==(139,134,15)
    assert len(set(critical+traps+boundary))==len(ids)
    assert all(rows[int(ids[k])][0]['solver_steps']==160 for k in critical)
    assert all(rows[int(ids[k])][0]['min_modulus']<9-4*np.sqrt(5) for k in boundary)
    assert max(m['unitarity_error'] for m in ms)<1e-10
    block=refined[traps];r=p.components(block)[2]
    small_kicks=[];small_steps=[]
    for length in (.1,.3):
        trial=p.expstep(block,-length*r/np.linalg.norm(r,axis=(1,2))[:,None,None])
        result,iters=p.refine(trial,limit=800,tolerance=1e-12)
        assert all(abs(p.metrics(v)['gradient']-.016477574426)<1e-8 for v in result)
        small_kicks.append(result);small_steps.append(iters)
    kick=p.expstep(block,-.6*r/np.linalg.norm(r,axis=(1,2))[:,None,None])
    escaped,escape_steps=p.refine(kick,limit=800,tolerance=1e-12)
    em=[p.metrics(v) for v in escaped]
    assert all(m['gradient']<1e-10 and abs(m['L']-10.2641137511)<1e-7 for m in em)

    @lru_cache(maxsize=4)
    def orbit(i):return candidates(rows[i][1])
    levels={10.475603:3990326,10.475787:2499540}
    trap_center=2*np.linalg.inv(np.eye(N)-1j*rational_center())-np.eye(N)
    trap_orbit=candidates(trap_center)
    records=[];witnesses=np.zeros_like(refined);escape_witnesses=np.empty_like(escaped)
    for k,i in enumerate(ids):
        before=rows[int(i)][0]
        entry=dict(id=int(i),original=before,refined=ms[k],extra_steps=int(steps[k]))
        if k in critical:
            spec=p.spectral_data(refined[k]);assert spec['top']>3.67
            reference=levels[round(ms[k]['L'],6)];eq=match(refined[k],orbit(reference))
            assert eq['error']<1e-6
            entry.update(category='iteration_limit',top=spec['top'],positive=spec['positive'],
                         known_reference=reference,equivalence=eq)
            witnesses[k]=spec['witness']
        elif k in traps:
            j=traps.index(k);eq=match(refined[k],trap_orbit);assert eq['error']<1e-6
            small_matches=[match(v[j],trap_orbit) for v in small_kicks]
            assert max(v['error'] for v in small_matches)<1e-6
            ee=match(escaped[j],orbit(912369));assert ee['error']<1e-6
            spec=p.spectral_data(escaped[j]);assert spec['top']>3.50
            entry.update(category='positive_residual_minimum',trap_equivalence=eq,
                         small_kick_equivalences=small_matches,
                         small_kick_steps=[int(v[j]) for v in small_steps],
                         escaped=em[j],escape_steps=int(escape_steps[j]),
                         escaped_top=spec['top'],escaped_positive=spec['positive'],
                         escaped_reference=912369,escaped_equivalence=ee)
            escape_witnesses[j]=spec['witness']
        else:
            entry['category']='near_boundary'
        records.append(entry)
    cert=certificate()
    summary=dict(retained_unresolved=288,counts=dict(Counter(v['category'] for v in records)),
                 refined_levels=dict(Counter(format(ms[k]['L'],'.6f') for k in critical)),
                 restart_critical_max_gradient=max(ms[k]['gradient'] for k in critical),
                 restart_top_min=min(v['top'] for v in records if 'top' in v),
                 trap_max_equivalence_error=max(v['trap_equivalence']['error'] for v in records
                                               if 'trap_equivalence' in v),
                 escaped_count=len(escaped),escaped_level=10.2641137511,
                 smaller_kicks_returning_to_trap={'.1':134,'.3':134},
                 escaped_max_gradient=max(m['gradient'] for m in em),
                 escaped_top_min=min(v['escaped_top'] for v in records if 'escaped_top' in v),
                 boundary_max_original_min_entry=max(rows[int(ids[k])][0]['min_modulus'] for k in boundary),
                 boundary_max_refined_min_entry=max(ms[k]['min_modulus'] for k in boundary),
                 certificate=cert,result='passed',
                 scope='All 288 retained unresolved points, not all 404581 unresolved runs. '
                       'Refinement and equivalence results are numerical; only the stated '
                       'rational-chart existence/positivity result is interval-certified.')
    output=ROOT/'data';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'stalled5.npz',ids=ids,refined=refined,witnesses=witnesses,
                        escaped_ids=ids[traps],escaped=escaped,escape_witnesses=escape_witnesses,
                        small_kick_lengths=[.1,.3],small_kick_endpoints=np.array(small_kicks))
    (output/'stalled5.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
