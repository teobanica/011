#!/usr/bin/env python3
"""Local mathematical and storage checks for the two U5 overnight programs.

Run python3 test_ahc5.py. Requires NumPy only. Temporary databases are removed.
The programs' --self-test additionally runs on every normal launch.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def run():
    reports={}
    for mode in ['ascent','stationary']:
        p=module('ahc5_'+mode);reports[mode]=p.self_check()
        rng=np.random.default_rng(71)
        u=p.haar(rng,6)
        for polar in [False,True]:
            v,_=p.ascent(u,use_polar=polar,limit=300)
            assert np.all(abs(v).sum((1,2)) >= abs(u).sum((1,2))-1e-12)
            assert max(p.metrics(w)['unitarity_error'] for w in v)<1e-11
        saddle=p.calibration()
        near=p.perturb(rng,np.tile(saddle,(6,1,1)),-4,-3)
        final,_=p.refine(near,limit=160)
        assert max(p.metrics(v)['gradient'] for v in final)<1e-10
        assert max(abs(abs(v).sum()-abs(saddle).sum()) for v in final)<1e-9
        for v in final:
            assert p.classify(p.metrics(v),p.spectral_data(v))=='nonflat_uphill'
        # The positive eigenvector really increases L on both sides at the saddle.
        b=p.spectral_data(saddle)['witness']
        for sign in [-1,1]:
            moved=p.expstep(saddle[None],sign*1e-3*b[None])[0]
            assert abs(moved).sum()>abs(saddle).sum()+1e-7
        # Every start family is unitary, reproducible, and full support after perturbation.
        policies=p.ASCENT_POLICIES if mode=='ascent' else p.ROOT_POLICIES
        for bid in range(len(policies)):
            fam,starts,cap=p.starts_for(p.DEFAULT_SEED,bid)
            assert np.array_equal(starts,p.starts_for(p.DEFAULT_SEED,bid)[1])
            assert max(p.metrics(v)['unitarity_error'] for v in starts)<1e-12
            assert abs(starts).min()>1e-10
        # Counts survive reopening, and bounded pools delete only unreferenced matrices.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.sqlite'
            db=sqlite3.connect(path);db.executescript(p.DDL)
            retention=p.Retention(db,10)
            for bid in range(4):
                rows=[]
                for k in range(p.BATCH):
                    idx=bid*p.BATCH+k;m=p.metrics(saddle)
                    info=dict(m,id=idx,batch_id=bid,offset=k,family='synthetic',status='near_flat',raw=m)
                    rows.append(dict(info=info,start=saddle,raw=saddle,endpoint=saddle,
                                     full=None,horizontal=None,witness=None))
                p.commit_batch(db,retention,bid,rows,bid+1.)
            assert db.execute('SELECT count(*) FROM kept').fetchone()[0]==16
            assert json.loads(db.execute('SELECT data FROM aggregates').fetchone()[0])['n']==64
            assert p.get_meta(db,'next_batch')==4
            db.close();db=sqlite3.connect(path)
            again=p.Retention(db,10)
            assert len(again.pools['sample:synthetic:near_flat'])==16
            # Review flags are retained individually even after the ordinary sample fills.
            spec=p.spectral_data(saddle)
            for bid in range(4,7):
                rows=[]
                for k in range(p.BATCH):
                    idx=bid*p.BATCH+k;m=p.metrics(saddle)
                    info=dict(m,id=idx,batch_id=bid,offset=k,family='synthetic_flags',status='candidate',
                              raw=m,top=spec['top'],positive=spec['positive'])
                    rows.append(dict(info=info,start=saddle,raw=saddle,endpoint=saddle,full=spec['full'],
                                     horizontal=spec['horizontal'],witness=spec['witness']))
                p.commit_batch(db,again,bid,rows,bid+1.)
            assert len(again.pools['flags'])==48
            assert sum(json.loads(r[0])['status']=='candidate' for r in db.execute('select info from kept'))==48
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            db.close()
        reports[mode]['source_sha256']=hashlib.sha256((ROOT/('ahc5_'+mode+'.py')).read_bytes()).hexdigest()
    report={'checks':reports,'result':'passed',
            'scope':'mathematical calibration, ascent, root recovery, all start families, bounded retention and reopening'}
    path=ROOT/'data'/'ahc5-tests.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

def runtime():
    import subprocess,signal,time,sys
    script=ROOT/'ahc5_ascent.py'
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'out'
        first=subprocess.Popen([sys.executable,str(script),'--out',str(out),'--workers','2'],
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        while True:
            line=first.stdout.readline()
            if 'Starting/resuming' in line:break
            if not line:raise RuntimeError('child exited before starting')
        second=subprocess.run([sys.executable,str(script),'--out',str(out)],capture_output=True,text=True)
        assert second.returncode==2 and 'another process' in second.stderr
        time.sleep(.3);first.send_signal(signal.SIGINT)
        first.communicate(timeout=30);assert first.returncode==0
        data=json.loads((out/'summary.json').read_text());start=data['next_batch']
        assert start>0 and data['state']=='Paused safely'
        resumed=subprocess.run([sys.executable,str(script),'--out',str(out),'--workers','2',
                                '--batches',str(start+6)],capture_output=True,text=True,timeout=30)
        assert resumed.returncode==0,resumed.stdout+resumed.stderr
        final=json.loads((out/'summary.json').read_text());assert final['starts']==(start+6)*16
        mismatch=subprocess.run([sys.executable,str(script),'--out',str(out),'--seed','123'],
                                capture_output=True,text=True)
        assert mismatch.returncode!=0 and 'differs' in mismatch.stderr
        with sqlite3.connect(out/'results.sqlite') as db:
            assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
        report={'lock_refused_concurrent_run':True,'interrupt_saved':True,'resume_added_exact_starts':96,
                'seed_mismatch_refused':True,'share_archive_created':(out/'share.zip').exists(),'result':'passed'}
        (ROOT/'data'/'ahc5-runtime-tests.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    run()
    if '--runtime' in sys.argv:
        runtime()
