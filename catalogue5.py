#!/usr/bin/env python3
"""Catalogue and interval-certify every saved nonflat stationary Mac endpoint.

python3 catalogue5.py
Requires NumPy and mpmath. The 452 input matrices come from stationary-mac.zip.
Classes are numerical phase/permutation/conjugation matches, not an exhaustive
classification. Every individual rounding box receives an interval uphill test.
Transposition comparisons and the equal rounded norms of groups 15/16 are
numerical checks; they do not assert exact equivalence or equality of norms.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
from itertools import permutations
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile
import numpy as np
from mpmath import iv
import ahc5_stationary as p

ROOT=Path(__file__).resolve().parent
PERMS=np.array(list(permutations(range(5))))

def dephase(u):
    u=u*np.conj(u[..., :,0]/abs(u[..., :,0]))[..., :,None]
    return u*np.conj(u[...,0,:]/abs(u[...,0,:]))[...,None,:]

def orbit(u):
    return dephase(u[PERMS[:,None,:,None],PERMS[None,:,None,:]].reshape(-1,5,5))

def match(u,candidates):
    u=dephase(u)
    errors=np.stack([np.linalg.norm(candidates-u,axis=(1,2)),np.linalg.norm(candidates-u.conj(),axis=(1,2))])
    a,b=np.unravel_index(np.argmin(errors),errors.shape)
    return dict(error=float(errors[a,b]),conjugate=bool(a),rows=PERMS[b//120].tolist(),columns=PERMS[b%120].tolist())

def certify(u,b):
    iv.dps=35
    def conj(z):return iv.mpc(z.real,-z.imag)
    def star(m):return iv.matrix([[conj(m[j,i]) for j in range(m.rows)] for i in range(m.cols)])
    def box(v):return iv.mpf(format(float(v),'.17g'))+iv.mpf(['-1e-8','1e-8'])
    v=iv.matrix([[iv.mpc(box(u[i,j].real),box(u[i,j].imag)) for j in range(5)] for i in range(5)])
    h=iv.matrix(5)
    for i in range(5):
        h[i,i]=iv.mpf(format(float(b[i,i].real),'.12f'))
        for j in range(i+1,5):
            h[i,j]=iv.mpc(format(float(b[i,j].real),'.12f'),format(float(b[i,j].imag),'.12f'));h[j,i]=conj(h[i,j])
    mod=[[abs(v[i,j]) for j in range(5)] for i in range(5)]
    s=iv.matrix([[v[i,j]/mod[i][j] for j in range(5)] for i in range(5)])
    vh=v*h;x=star(s)*v;xh=x*h*h
    q=sum((vh[i,j]*conj(s[i,j])).real**2/mod[i][j] for i in range(5) for j in range(5))
    q-=sum(xh[i,i].real for i in range(5))
    assert q.a>iv.mpf('.20634')
    return dict(lower=float(q.a),upper=float(q.b),certified_positive=True)

def transpose_comparisons(classes,orbits):
    pairs=[];parent=list(range(len(classes)))
    def root(j):
        while parent[j]!=j:j=parent[j]
        return j
    for j,c in enumerate(classes):
        for k in range(j+1,len(classes)):
            # L is 5-Lipschitz for the Frobenius norm on 5 by 5 matrices.
            if abs(c['L']-classes[k]['L'])>5e-6:continue
            if j not in orbits:orbits[j]=orbit(c['U'])
            eq=match(classes[k]['U'].T,orbits[j])
            if eq['error']<1e-6:
                pairs.append(dict(groups=[j+1,k+1],equivalence=eq))
                parent[root(k)]=root(j)
    assert [v['groups'] for v in pairs]==[[4,5],[19,20],[22,23],[30,31]]
    assert max(v['equivalence']['error'] for v in pairs)<6e-11
    count=len({root(j) for j in range(len(classes))})
    assert count==36
    u,v=classes[14]['U'],classes[15]['U']
    norms=[float(np.sum(abs(w))) for w in (u,v)]
    minima=[float(np.min(abs(w))) for w in (u,v)]
    assert all(format(x,'.12f')=='10.290855136357' for x in norms)
    assert .109049<minima[0]<.109050 and .211227<minima[1]<.211228
    return dict(pairs=pairs,observed_groups_allowing_transposition=count,
                coincident_rounded_norms=dict(groups=[15,16],norms=norms,
                    rounded_to_12_decimals='10.290855136357',minimum_moduli=minima),
                scope='Numerical comparisons only; no exact equality of norms or '
                      'pairwise inequivalence of the certified critical points is asserted.')

def main():
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ROOT/'overnight5/stationary-mac.zip') as z:
            assert all('/' not in x and '\\' not in x for x in z.namelist());z.extractall(tmp)
        db=sqlite3.connect('file:'+str(Path(tmp)/'results.sqlite')+'?mode=ro',uri=True)
        rows=[(i,json.loads(s),np.frombuffer(u,dtype='<c16').reshape(5,5).copy(),
               np.frombuffer(b,dtype='<c16').reshape(5,5).copy())
              for i,s,u,b in db.execute('select id,info,endpoint,witness from kept order by id')
              if json.loads(s)['status']=='nonflat_uphill']
        db.close()
    assert len(rows)==452
    rows.sort(key=lambda v:(v[1]['L'],v[0]))
    classes=[];records=[];orbits={}
    for idx,m,u,b in rows:
        certificate=certify(u,b)
        found=None
        for j,c in enumerate(classes):
            if abs(m['L']-c['L'])>1e-7:continue
            if j not in orbits:orbits[j]=orbit(c['U'])
            eq=match(u,orbits[j])
            if eq['error']<1e-6:found=(j,eq);break
        if found is None:
            j=len(classes);spec=p.spectral_data(u)
            classes.append(dict(U=u,id=idx,L=m['L'],min_entry=m['min_modulus'],
                                top=m['top'],positive=m['positive'],zero=spec['zero'],
                                smallest_absolute_horizontal=float(min(abs(spec['horizontal']))),
                                retained_count=0,certificate_lower=certificate['lower']))
            eq=dict(error=0.,conjugate=False,rows=list(range(5)),columns=list(range(5)))
        else:j,eq=found
        classes[j]['retained_count']+=1
        classes[j]['certificate_lower']=min(classes[j]['certificate_lower'],certificate['lower'])
        records.append(dict(id=idx,class_index=j,equivalence=eq,certificate=certificate))
    transposes=transpose_comparisons(classes,orbits)
    data=ROOT/'data'
    np.savez_compressed(data/'catalogue5.npz',matrices=np.array([v['U'] for v in classes]),
                        representative_ids=[v['id'] for v in classes],top=[v['top'] for v in classes])
    for c in classes:del c['U']
    result=dict(result='passed',count=452,observed_classes=len(classes),box_radius='1e-8',
                equivalence_tolerance='1e-6',minimum_certified_curvature=min(v['certificate']['lower'] for v in records),
                classes=classes,records=records,transposition_comparisons=transposes,
                scope='Numerical equivalence catalogue of retained endpoints; interval exclusion in every '
                      'specified coordinate box, conditional on unitarity. No critical-root existence or '
                      'completeness claim. Class counts count retained matrices only.')
    (data/'catalogue5.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('classes','records')},indent=2))

if __name__=='__main__':main()
