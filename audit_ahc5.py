#!/usr/bin/env python3
"""Read-only audit of a U5 campaign folder or its share.zip. NumPy only.

Usage: python3 audit_ahc5.py path/to/ahc5-stationary-out/share.zip
Checks counters, preserved source hash, every retained start and endpoint,
all stored spectra, and the saved positive-curvature witness. The two original
ahc5 launch scripts must be beside this file. No optimization is performed.
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

def audit(folder):
    db=sqlite3.connect('file:'+str((folder/'results.sqlite').resolve())+'?mode=ro',uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        meta={k:json.loads(v) for k,v in db.execute('select key,value from meta')}
        cfg=meta['config'];mode=cfg['mode'];assert mode in ('ascent','stationary')
        path=ROOT/('ahc5_'+mode+'.py')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==cfg['sha256'], 'Use the matching original script version for this archive.'
        assert hashlib.sha256((folder/path.name).read_bytes()).hexdigest()==digest
        spec=importlib.util.spec_from_file_location('ahc5_audit_module',path)
        p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
        groups=[(family,status,json.loads(data)) for family,status,data in db.execute('select * from aggregates')]
        assert sum(d['n'] for _,_,d in groups)==meta['next_batch']*p.BATCH
        flags=sum(d['n'] for _,status,d in groups if status in ('candidate','hessian_uncertain'))
        assert db.execute("select count(*) from slots where pool='flags'").fetchone()[0]==flags
        assert db.execute('select count(*) from slots s left join kept k on k.id=s.run_id where k.id is null').fetchone()[0]==0
        @lru_cache(maxsize=32)
        def starts(bid):
            return p.starts_for(cfg['seed'],bid)[1]
        worst_metric=worst_spectrum=worst_witness=0.;count=spectra=0
        retained=Counter(); unresolved_curvatures=[]
        for idx,txt,initial,raw,endpoint,full,hor,witness in db.execute('select * from kept order by id'):
            m=json.loads(txt);count+=1;retained[m['status']]+=1
            u0=np.frombuffer(initial,dtype='<c16').reshape(5,5)
            assert np.max(abs(u0-starts(idx//p.BATCH)[idx%p.BATCH]))<1e-11
            if endpoint is None:
                assert m['status']=='numeric_error';continue
            u=np.frombuffer(endpoint,dtype='<c16').reshape(5,5)
            current=p.metrics(u)
            for key,val in current.items():
                if m.get(key) is not None:
                    error=abs(val-m[key]);worst_metric=max(worst_metric,error)
                    assert error<1e-10*max(1,abs(val)),(idx,key,error)
            if raw is not None and 'raw' in m:
                rv=np.frombuffer(raw,dtype='<c16').reshape(5,5)
                for key,val in p.metrics(rv).items():
                    assert abs(val-m['raw'][key])<1e-10*max(1,abs(val))
            if hor is not None:
                spec=p.spectral_data(u);spectra+=1
                for key,blob in [('full',full),('horizontal',hor)]:
                    values=np.frombuffer(blob,dtype='<f8');error=np.max(abs(values-spec[key]))
                    worst_spectrum=max(worst_spectrum,float(error))
                    assert error<1e-8*max(1,np.max(abs(values)))
                b=np.frombuffer(witness,dtype='<c16').reshape(5,5)
                assert np.linalg.norm(b-b.conj().T)<1e-11 and abs(np.linalg.norm(b)-1)<1e-10
                s,x,_=p.components(u)
                q=np.sum(((u@b*s.conj()).real**2)/abs(u))-np.trace(x@b@b).real
                error=abs(q-m['top']);worst_witness=max(worst_witness,float(error))
                assert error<1e-8*max(1,abs(q))
                assert p.classify(current,spec)==m['status']
            else:
                assert m['status'] not in ('nonflat_uphill','candidate')
            if m['status']=='unresolved':
                # At a noncritical point this is a geodesic second derivative,
                # not a stationary-point inertia or a statement about its limit.
                unresolved_curvatures.append(p.spectral_data(u)['top'])
        counts=Counter()
        extrema={}
        for _,status,data in groups:
            counts[status]+=data['n']
            target=extrema.setdefault(status,{})
            for key,value in data.items():
                if isinstance(value,dict) and 'min' in value and 'max' in value:
                    previous=target.setdefault(key,dict(min=value['min'],max=value['max']))
                    previous['min']=min(previous['min'],value['min'])
                    previous['max']=max(previous['max'],value['max'])
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['counts']==dict(counts)
        assert summary['starts']==meta['next_batch']*p.BATCH
        assert summary['retained_endpoints']==count
        return {'mode':mode,'counted_starts':meta['next_batch']*p.BATCH,'retained_endpoints':count,
                'counts':dict(counts),'retained_statuses':dict(retained),
                'aggregate_extrema':extrema,
                'retained_unresolved_geodesic_curvature':
                    dict(count=len(unresolved_curvatures),minimum=min(unresolved_curvatures),
                         maximum=max(unresolved_curvatures)) if unresolved_curvatures else None,
                'spectrum_pairs_checked':spectra,'all_flags_retained':flags,
                'max_scalar_discrepancy':worst_metric,'max_spectrum_discrepancy':worst_spectrum,
                'max_witness_discrepancy':worst_witness,'source_sha256':digest,'result':'passed'}
    finally:
        db.close()

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit(__doc__)
    source=Path(sys.argv[1])
    if source.is_dir():
        report=audit(source)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(source) as z:
                assert all('/' not in name and '\\' not in name and name not in ('.','..') for name in z.namelist())
                z.extractall(tmp)
            report=audit(Path(tmp))
    print(json.dumps(report,indent=2))
