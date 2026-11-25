"""Recompute all reported tensors with the final independently checked jets."""
import argparse,json
from pathlib import Path
import numpy as np
from second_fundamental import FlexChart,uncj,cj,second_form,real_second_form


def sci(x):
    if x==0:return '0'
    a,b=f'{x:.2e}'.split('e')
    return a+r'\times10^{'+str(int(b))+'}'


def main(out):
    rows=[];rich=[];geos=[]
    for number,(label,tau,s) in enumerate([('A',.23+1.08j,5+2j),('B',.07+.94j,-4+3j),('C',.31+.83j,-2+7j)],1):
        chart=FlexChart(tau,s);D,DD=chart.jets();D2,DD2=chart.jets(.7,40)
        for N,h in [(32,.001),(64,.0005),(96,.00025)]:
            p=Path(f'results/generic{number}/common-N{N}-h{h:g}.json')
            raw=json.loads(p.read_text());g=uncj(raw['g']);C=uncj(raw['connection']);LC=np.array(raw['real_connection'])
            def tensor(C,LC):
                a=second_form(g,C,D,DD);E=uncj(a['orthonormal_parameter_frame']);normal=uncj(a['unit_normal'])
                a.update(real_second_form(g,LC,D,DD,E,normal))
                alt=second_form(g,C,D2,DD2)
                a['geometry_B_check']=float(np.linalg.norm(uncj(a['B_orthonormal'])-uncj(alt['B_orthonormal'])))
                return a
            a=tensor(C,LC);a.update(sample=label,N=N,h=h,source=str(p),metric_config=raw['metric_config'])
            rows.append(a)
            if number==1:
                hh=.001 if N<96 else .0005
                lo=.0005 if N<96 else .00025
                large=json.loads(Path(f'results/generic1/common-N{N}-h{hh:g}.json').read_text())
                small=json.loads(Path(f'results/generic1/common-N{N}-h{lo:g}.json').read_text())
                CR=(4*uncj(small['connection'])-uncj(large['connection']))/3
                LR=(4*np.array(small['real_connection'])-np.array(large['real_connection']))/3
                b=tensor(CR,LR);b.update(N=N,h_small=lo,h_large=hh);rich.append(b)
    for name in ['geodesic-generic2','geodesic-generic2-N64']:
        source=json.loads((Path('results')/name/'geodesics.json').read_text())
        for r in source['results']:
            geos.append(dict(N=source['metric_config']['N'],**r))
    summary=dict(full_tensors=rows,richardson_sample_A=rich,geodesics=geos)
    Path('results/summary.json').write_text(json.dumps(summary,indent=2))

    lines=[r'\begin{center}',r'\begin{tabular}{ccccc}',r'\toprule',
        r'Sample & $N$ & $h$ & $\eps_\II$ & $\tfrac12\|\II_{\mathbb R}\|$\\',r'\midrule']
    for a in rows:
        lines.append(f"{a['sample']} & {a['N']} & ${sci(a['h'])}$ & ${sci(a['epsilon_II'])}$ & ${sci(a['epsilon_II_real_half'])}$\\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}',
        r'Here $N$ controls the bulk mesh. The three levels use cusp radii',
        r'$10^{-4},3\cdot10^{-5},10^{-5}$, respectively; angular counts',
        r'$64,80,96$; radial ratios $1.20,1.15,1.12$; and $4^2,5^2,6^2$',
        r'interior quadrature points per triangle. Thus several error sources',
        r'are refined together. The $N=96$ meshes use a translated flat fundamental',
        r'domain; the underlying moduli coordinates remain unchanged.',
        r'\medskip',r'\noindent At the finest level the three absolute complex coefficients',
        r'of the symmetric part are:',r'\begin{center}',r'\begin{tabular}{cccc}',
        r'\toprule Sample & $|B_{11}|$ & $|B_{12}|$ & $|B_{22}|$\\\midrule']
    for a in rows:
        if a['N']!=96:continue
        B=uncj(a['B_orthonormal']);B=(B+B.T)/2
        lines.append(a['sample']+' & '+' & '.join('$'+sci(abs(v))+'$' for v in [B[0,0],B[0,1],B[1,1]])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'Sample A has the most clustered punctures and the largest derivative',
        r'error. We additionally combine stencils on the same mesh by',
        r'$\Gamma_R=(4\Gamma_{h/2}-\Gamma_h)/3$, and likewise for the real',
        r'connection, cancelling the leading centered-difference error:',
        r'\begin{center}\begin{tabular}{cccc}',
        r'\toprule $N$ & smaller $h$ & $\eps_{\II,R}$ & $\tfrac12\|\II_{\mathbb R,R}\|$\\\midrule']
    for a in rich:
        lines.append(f"{a['N']} & ${sci(a['h_small'])}$ & ${sci(a['epsilon_II'])}$ & ${sci(a['epsilon_II_real_half'])}$\\\\")
    lines += [r'\bottomrule\end{tabular}\end{center}',
        'The geometry-only change of the normalized tensor under independent Cauchy',
        'jet extraction is at most $'+sci(max(r['geometry_B_check'] for r in rows+rich))+'$.',
        r'The remaining values should be read as discretization residuals:',
        r'they decrease under refinement, and no stable nonzero obstruction has',
        r'been detected. They do not certify exact vanishing.']
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    (out/'results.tex').write_text('\n'.join(lines)+'\n')

    lines=[r'\begin{center}\begin{tabular}{ccccc}',r'\toprule',
           r'Metric $N$ & $t$ & Steps & $|r(t)|$ & $|\widetilde r(t)|$\\\midrule']
    for r in geos:
        lines.append(f"{r['N']} & {r['time']:.2f} & {r['steps']} & ${sci(abs(uncj(r['normal_residual'])))}$ & ${sci(abs(uncj(r['control_normal_residual'])))}$\\\\")
    lines += [r'\bottomrule\end{tabular}\end{center}',
        r'The smaller-time control coefficients approach $-0.0375-0.035i$,',
        r'as predicted. The control departure is several orders of magnitude',
        r'larger than the flex residual. Refining the time step and then the',
        r'metric gives a separate check; this still shares the same Liouville',
        r'uniformization method and is not a second independent metric solver.',
        'The largest speed drift in these shots is $'+sci(max(abs(r['speed']-1) for r in geos))+'$.',
        'The endpoint-coordinate matching error is below $'+sci(max(max(r['matching_error'],r['control_matching_error']) for r in geos))+'$.']
    (out/'geodesics.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({"fine_epsilon":[a['epsilon_II'] for a in rows if a['N']==96],
                       "richardson":[a['epsilon_II'] for a in rich],"geodesic_fine":geos[-1]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
