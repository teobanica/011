#!/usr/bin/env python3
"""Regression and runtime tests for the two standalone daytime launchers.

python3 test_day5.py
Uses temporary output folders. Tests the original replay path, complete failure
identifier retention, atomic rollback, standalone launches, pause/resume, locks,
configuration guards, source preservation, and independent output audits.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import numpy as np
import ahc5_stationary as old
import audit_day5

ROOT=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);return p

def start_basis_checks(p):
    """Accept eigenbasis freedom but reject a changed perturbation length."""
    cache=p.catalogue_cache()
    original=cache['directions'].copy();witness=cache['witness'].copy()
    changed=original.copy()
    for j,values in enumerate(cache['spectra']):
        unseen=set(range(len(values)))
        while unseen:
            first=min(unseen)
            group=np.flatnonzero(abs(values-values[first])<1e-7)
            unseen.difference_update(group.tolist())
            changed[j,group[0]]*=-1
            if len(group)>1:
                a,b=group[:2];x=changed[j,a].copy();y=changed[j,b].copy()
                changed[j,a]=.6*x+.8*y;changed[j,b]=-.8*x+.6*y
    samples=[]
    try:
        cache['directions']=changed;cache['witness']=changed[:,-1].copy()
        for bid in (2,5):
            family,starts,indices=p.day_starts(521819,bid)
            samples.append((bid,family,starts,indices))
    finally:
        cache['directions']=original;cache['witness']=witness
    accepted=rejected=0
    for bid,family,starts,indices in samples:
        for k,u in enumerate(starts):
            idx=bid*p.BATCH+k
            err=audit_day5.eigenbasis_start_error(p,521819,idx,family,u)
            assert err<1e-7,(family,k,err)
            accepted+=1
            base=p.CATALOGUE[indices[k]]
            z,v=np.linalg.eig(base.conj().T@u)
            b=v@np.diag(np.angle(z))@np.linalg.inv(v);b=(b+b.conj().T)/2
            bad=p.expstep(base,1.1*b)
            try:
                err=audit_day5.eigenbasis_start_error(p,521819,idx,family,bad)
            except AssertionError:
                rejected+=1
            else:
                assert err>.01,(family,k,err)
                rejected+=1
    return dict(allowed_basis_changes_accepted=accepted,wrong_lengths_rejected=rejected)

def main():
    replay=module('ahc5_day_replay');search=module('ahc5_day_search')
    replay.self_check();search.self_check()
    basis_checks=start_basis_checks(search)
    bids=[0,1,432489,1407432]
    baseline_unresolved=0
    for bid in bids:
        a=old.run_batch(521812,bid);b=replay.run_batch(521812,bid)
        for original,new in zip(a,b):
            assert np.array_equal(original['start'],new['start'])
            assert np.max(abs(original['endpoint']-new['raw']))<1e-12
            assert original['info']['status']==new['info']['baseline']['status']
            baseline_unresolved+=original['info']['status']=='unresolved'
    assert baseline_unresolved>0
    with tempfile.TemporaryDirectory() as tmp:
        # Exercise journal counting on synthetic copies of an actual difficult batch.
        db=sqlite3.connect(Path(tmp)/'transaction.sqlite');db.executescript(replay.DDL)
        rows=replay.run_batch(521812,1407432)
        for k,row in enumerate(rows):row['info'].update(id=k,batch_id=0,offset=k)
        retention=replay.Retention(db,521812)
        replay.commit_batch(db,retention,0,rows,1.)
        expected=sum(row['info']['baseline']['status']=='unresolved' or row['info']['status'].startswith('unresolved') for row in rows)
        assert db.execute('select count(*) from followups').fetchone()[0]==expected
        assert replay.get_meta(db,'next_batch')==1
        # A late failure rolls back counts, cursor, identifiers and saved matrices.
        before=list(db.execute('select family,status,data from aggregates order by 1,2'))
        import copy
        bad=copy.deepcopy(rows)
        for k,row in enumerate(bad):row['info'].update(id=16+k,batch_id=1)
        bad[-1]['info'].pop('status')
        try:replay.commit_batch(db,retention,1,bad,2.)
        except KeyError:pass
        else:raise AssertionError('missing diagnostic did not stop transaction')
        assert replay.get_meta(db,'next_batch')==1
        assert list(db.execute('select family,status,data from aggregates order by 1,2'))==before
        assert db.execute('select count(*) from followups').fetchone()[0]==expected
        db.close()
    runtime=[]
    for mode in ('replay','search'):
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/('ahc5_day_'+mode+'.py');shutil.copy2(ROOT/script.name,script)
            out=Path(tmp)/'out'
            cmd=[sys.executable,str(script),'--out',str(out),'--workers','2']
            proc=subprocess.Popen(cmd+['--hours','.05'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            deadline=time.monotonic()+30
            while True:
                line=proc.stdout.readline()
                if 'Starting/resuming' in line:break
                if not line or time.monotonic()>deadline:raise RuntimeError('standalone launch failed: '+line)
            duplicate=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            assert duplicate.returncode==2 and 'another process' in duplicate.stderr
            time.sleep(.2);proc.send_signal(signal.SIGINT)
            output,_=proc.communicate(timeout=60);assert proc.returncode==0,output
            first=json.loads((out/'summary.json').read_text());assert first['state']=='Paused safely' and first['starts']>0
            nxt=first['next_batch']
            done=subprocess.run(cmd+['--batches',str(nxt+8)],capture_output=True,text=True,timeout=60)
            assert done.returncode==0,done.stdout+done.stderr
            final=json.loads((out/'summary.json').read_text());assert final['starts']==(nxt+8)*16
            assert final['observed_process_peak_sum_mib']<1024
            audit=audit_day5.audit(out)
            if mode=='search':
                changed=subprocess.run(cmd+['--seed','1'],capture_output=True,text=True,timeout=30)
                assert changed.returncode!=0 and 'differs' in changed.stderr
            runtime.append(dict(mode=mode,interrupt_saved=True,concurrent_lock_checked=True,
                                resumed_exactly=128,peak_process_sum_mib=final['observed_process_peak_sum_mib'],audit=audit))
    result=dict(result='passed',baseline_batches_compared=bids,initial_unresolved_replayed=baseline_unresolved,
                atomic_rollback_checked=True,start_basis_checks=basis_checks,standalone_runtime=runtime)
    (ROOT/'data/day5-tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
