#!/usr/bin/env python3
"""Read-only audit of a daytime U5 output folder or share.zip. NumPy only.

Usage: python3 audit_day5.py PATH
Checks all counters and tracked unresolved IDs, generating source hashes,
every retained start/endpoint, all saved spectra and curvature witnesses.
Does not repeat searches or reconstruct unretained matrices.
"""
from functools import lru_cache
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def eigenbasis_start_error(p, seed, idx, family, u):
    """Check a saved start up to signs/rotations in repeated eigenspaces.

    This checks numerical compatibility with the stated start protocol,
    not identical regeneration on a different NumPy/LAPACK build. The original
    campaign source and the actual saved start are never changed.
    """
    assert family in ('root_soft','ascent_weak')
    bid,k=divmod(idx,p.BATCH);cache=p.catalogue_cache()
    rng=np.random.default_rng(np.random.SeedSequence([seed,bid]))
    indices=(bid//len(p.DAY_POLICIES)*p.BATCH+np.arange(p.BATCH))%len(p.CATALOGUE)
    if family=='ascent_weak':
        indices=cache['weak'][rng.integers(0,len(cache['weak']),p.BATCH)]
    j=int(indices[k]);base=p.CATALOGUE[j]
    z,v=np.linalg.eig(base.conj().T@u)
    b=v@np.diag(np.angle(z))@np.linalg.inv(v);b=(b+b.conj().T)/2
    errors=[float(np.linalg.norm(u.conj().T@u-np.eye(p.N))),
            float(np.max(abs(p.expstep(base,b)-u)))]
    values=cache['spectra'][j];directions=cache['directions'][j]
    if family=='root_soft':
        coeff=rng.normal(size=(p.BATCH,3))[k]
        scale=(10**rng.uniform(-4,0,p.BATCH))[k]
        a=b/scale;coord=np.einsum('ij,kji->k',a,directions).real
        order=np.argsort(abs(values))[:3]
        errors.extend([abs(np.linalg.norm(b)/scale-1),
                       np.linalg.norm(a-np.einsum('k,kij->ij',coord,directions))])
        unseen=set(range(len(values)))
        while unseen:
            first=min(unseen)
            group=np.flatnonzero(abs(values-values[first])<1e-7)
            unseen.difference_update(group.tolist())
            expected=np.sum(coeff[np.isin(order,group)]**2)/np.sum(coeff**2)
            errors.append(abs(np.sum(coord[group]**2)-expected))
    else:
        rng.choice([-1,1],size=p.BATCH)  # Its sign is part of the unknown eigenvector sign.
        noise=np.einsum('bk,kij->bij',rng.normal(size=(p.BATCH,p.DIM)),p.BASIS)
        noise/=np.linalg.norm(noise,axis=(1,2))[:,None,None]
        scale=(10**rng.uniform(-4,-.3,p.BATCH))[k];a=b/scale
        top=directions[abs(values-values[-1])<1e-7]
        def project(x):
            return np.einsum('k,kij->ij',np.einsum('ij,kji->k',x,top).real,top)
        ap=a-project(a);np_=noise[k]-project(noise[k])
        assert np.linalg.norm(ap)>1e-10
        alpha=.1*np.vdot(ap,np_).real/np.linalg.norm(ap)**2
        assert .89<alpha<1.11
        w=alpha*a-.1*noise[k]
        errors.extend([abs(np.linalg.norm(b)/scale-1),np.linalg.norm(w-project(w)),
                       abs(np.linalg.norm(w)-1)])
    return float(max(errors))


def audit(folder):
    db=sqlite3.connect('file:'+str((folder/'results.sqlite').resolve())+'?mode=ro',uri=True)
    assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    cfg=json.loads(db.execute("select value from meta where key='config'").fetchone()[0])
    mode=cfg['mode'];assert mode in ('day-replay','day-search')
    path=ROOT/('ahc5_'+mode.replace('-','_')+'.py')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==cfg['sha256']
    assert any(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p in folder.glob('*.py'))
    spec=importlib.util.spec_from_file_location('audited_day',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    nxt=p.get_meta(db,'next_batch');total=nxt*p.BATCH
    aggregates=[(f,s,json.loads(d)) for f,s,d in db.execute('select * from aggregates')]
    baseline=[(f,s,json.loads(d)) for f,s,d in db.execute('select * from baseline_aggregates')]
    assert sum(d['n'] for _,_,d in aggregates)==sum(d['n'] for _,_,d in baseline)==total
    before=sum(d['n'] for _,s,d in baseline if s=='unresolved')
    after=sum(d['n'] for _,s,d in aggregates if s.startswith('unresolved'))
    assert db.execute("select count(*) from followups where baseline_status='unresolved'").fetchone()[0]==before
    assert db.execute("select count(*) from followups where status like 'unresolved%'").fetchone()[0]==after
    assert db.execute('select count(*) from followups where id<0 or id>=?',(total,)).fetchone()[0]==0
    stationary=sum(d['n'] for _,s,d in aggregates if s in ('candidate','nonflat_uphill'))
    assert db.execute('select coalesce(sum(n),0) from signature_counts').fetchone()[0]==stationary
    flags=sum(d['n'] for _,s,d in aggregates if s in ('candidate','hessian_uncertain'))
    assert db.execute("select count(*) from slots where pool='flags'").fetchone()[0]==flags
    assert db.execute('select count(*) from slots s left join kept k on k.id=s.run_id where k.id is null').fetchone()[0]==0
    @lru_cache(maxsize=32)
    def starts(bid):
        return p.starts_for(cfg['seed'],bid)[1] if mode=='day-replay' else p.day_starts(cfg['seed'],bid)[1]
    count=spectra=exact_starts=basis_starts=0
    worst_metric=worst_spec=worst_q=worst_direct=worst_basis=0.
    basis_families=Counter()
    products=p.BASIS[:,None]@p.BASIS[None,:]
    for idx,txt,initial,raw,endpoint,full,hor,witness in db.execute('select * from kept order by id'):
        m=json.loads(txt);count+=1
        u0=np.frombuffer(initial,dtype='<c16').reshape(5,5)
        if np.max(abs(u0-starts(idx//16)[idx%16]))<1e-11:
            exact_starts+=1
        else:
            assert mode=='day-search'
            err=eigenbasis_start_error(p,cfg['seed'],idx,m['family'],u0)
            assert err<1e-7,(idx,m['family'],err)
            basis_starts+=1;basis_families[m['family']]+=1;worst_basis=max(worst_basis,err)
        if endpoint is None:
            assert m['status']=='numeric_error';continue
        u=np.frombuffer(endpoint,dtype='<c16').reshape(5,5)
        for key,value in p.metrics(u).items():
            if m.get(key) is not None:
                err=abs(value-m[key]);worst_metric=max(worst_metric,err);assert err<1e-9*max(1,abs(value))
        if raw is not None and 'raw' in m:
            v=np.frombuffer(raw,dtype='<c16').reshape(5,5)
            for key,value in p.metrics(v).items():assert abs(value-m['raw'][key])<1e-9*max(1,abs(value))
        if hor is not None:
            sp=p.spectral_data(u);spectra+=1
            # Direct polarization of the geodesic second derivative, without
            # using the campaign's Hessian assembly routine.
            phase=u/abs(u)
            radial=((u@p.BASIS)*phase.conj()).real.reshape(p.DIM,-1)/np.sqrt(abs(u)).ravel()
            direct=radial@radial.T-np.sum((u@products)*phase.conj(),axis=(-2,-1)).real
            direct=(direct+direct.T)/2
            err=float(np.max(abs(np.linalg.eigvalsh(direct)-np.frombuffer(full,dtype='<f8'))))
            worst_direct=max(worst_direct,err);assert err<1e-8
            for key,blob in [('full',full),('horizontal',hor)]:
                saved=np.frombuffer(blob,dtype='<f8');err=np.max(abs(saved-sp[key]));worst_spec=max(worst_spec,float(err))
                assert err<1e-8*max(1,np.max(abs(saved)))
            b=np.frombuffer(witness,dtype='<c16').reshape(5,5)
            assert np.linalg.norm(b-b.conj().T)<1e-10 and abs(np.linalg.norm(b)-1)<1e-9
            s,x,_=p.components(u);q=np.sum(((u@b*s.conj()).real**2)/abs(u))-np.trace(x@b@b).real
            err=abs(q-m['top']);worst_q=max(worst_q,float(err));assert err<1e-8*max(1,abs(q))
            if m['status'] in ('candidate','nonflat_uphill','hessian_uncertain'):
                assert p.classify(p.metrics(u),sp)==m['status']
            if m['status']=='unresolved_uphill':assert m['gradient']>p.GRAD_TOL and sp['top']>p.CURV_TOL
            if m['status'] in ('candidate','nonflat_uphill'):
                sig,error=p.signature_match(u,sp)
                assert sig==m['known_signature'] and abs(error-m['signature_error'])<1e-8
        if m['status']=='small_entry_excluded':assert m['min_modulus']<p.BOUNDARY_CUTOFF
        if m['status']=='near_flat':assert m['flat_error']<=p.FLAT_TOL
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['starts']==total and summary['retained_endpoints']==count
    counts=Counter()
    for _,status,data in aggregates:counts[status]+=data['n']
    assert dict(counts)==summary['counts']
    transitions=[dict(before=a,after=b,count=n) for a,b,n in db.execute(
        'select baseline_status,status,count(*) from followups group by baseline_status,status order by 1,2')]
    assert transitions==summary['followup_transitions']
    assert db.execute("select count(*) from followups where status='nonflat_uphill' and (gradient>? or top<=?)",
                      (p.GRAD_TOL,p.CURV_TOL)).fetchone()[0]==0
    assert db.execute("select count(*) from followups where status='small_entry_excluded' and min_modulus>=?",
                      (p.BOUNDARY_CUTOFF,)).fetchone()[0]==0
    result=dict(mode=mode,counted_starts=total,retained_endpoints=count,spectrum_pairs=spectra,
                final_counts=dict(counts),followup_transitions=transitions,
                starts_regenerated_within_1e_11=exact_starts,
                starts_checked_up_to_eigenbasis=basis_starts,eigenbasis_families=dict(basis_families),
                max_eigenbasis_compatibility_error=worst_basis,
                start_check_scope='Exact regeneration where possible; otherwise numerical eigenspace compatibility, not identical regeneration.',
                tracked_initial_unresolved=before,tracked_final_unresolved=after,
                all_candidate_flags_retained=flags,max_scalar_discrepancy=worst_metric,
                max_spectrum_discrepancy=worst_spec,max_witness_discrepancy=worst_q,
                max_direct_hessian_discrepancy=worst_direct,result='passed')
    db.close();return result

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit(__doc__)
    path=Path(sys.argv[1])
    if path.is_dir():result=audit(path)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                assert all('/' not in n and '\\' not in n and n not in ('.','..') for n in z.namelist());z.extractall(tmp)
            result=audit(Path(tmp))
    print(json.dumps(result,indent=2))
