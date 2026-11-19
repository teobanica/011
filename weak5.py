#!/usr/bin/env python3
"""Certify a positive curvature on the retained weaker U5 saddle's rounding box.

Run python3 weak5.py. Requires mpmath. This is a rigorous exclusion of local
maxima in the box, conditional only on the matrix being unitary. Criticality
and the quoted full inertia of the numerical center are not certified here.
The two overnight programs themselves require only NumPy.
"""
import json
from pathlib import Path
from mpmath import mp,iv

ROOT=Path(__file__).resolve().parent

def check():
    mp.dps=70;iv.dps=60
    data=json.loads((ROOT/'data'/'ahc5-weak.json').read_text())
    def box(value):
        mid=mp.mpf(str(value));eps=mp.mpf('1e-12')
        return iv.mpf([str(mid-eps),str(mid+eps)])
    u=iv.matrix([[iv.mpc(box(data['U_real'][i][j]),box(data['U_imag'][i][j]))
                  for j in range(5)] for i in range(5)])
    b=iv.matrix(5)
    for i in range(5):
        b[i,i]=iv.mpf(format(data['B_real'][i][i],'.12f'))
        for j in range(i+1,5):
            b[i,j]=iv.mpc(format(data['B_real'][i][j],'.12f'),format(data['B_imag'][i][j],'.12f'))
            b[j,i]=iv.mpc(b[i,j].real,-b[i,j].imag)
    r=iv.matrix([[abs(u[i,j]) for j in range(5)] for i in range(5)])
    phase=iv.matrix([[u[i,j]/r[i,j] for j in range(5)] for i in range(5)])
    star=iv.matrix([[iv.mpc(phase[j,i].real,-phase[j,i].imag) for j in range(5)] for i in range(5)])
    x=star*u;ub=u*b;xb2=x*b*b
    q=sum((ub[i,j]*iv.mpc(phase[i,j].real,-phase[i,j].imag)).real**2/r[i,j] for i in range(5) for j in range(5))
    q-=sum(xb2[i,i].real for i in range(5))
    assert q.a>iv.mpf('.20635') and q.b<iv.mpf('.20636')
    result={'run_id':1001,'coordinate_radius':'1e-12','B_rounding_decimal_places':12,
            'certified_curvature_bounds':['0.20635','0.20636'],
            'meaning':'Every unitary in this entrywise real/imaginary box has a positive second variation; '
                      'the numerical criticality and inertia are separate floating-point statements.',
            'result':'passed'}
    (ROOT/'data'/'ahc5-weak-certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    check()
