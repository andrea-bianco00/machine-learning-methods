"""Separate exploratory sensitivity study for ENSAI Q5, not assignment baseline."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.q5 import data, direction, objective, zero_gradient


def run(z, y, *, method, distribution, scale, budget, seed, h=0.01, checkpoints=20):
    """Compare zero-order methods with eta_k=scale/k; scale=1 is the baseline."""
    if method not in ('zgd', 'zsgd') or distribution not in ('gaussian', 'rademacher', 'sphere'):
        raise ValueError('Unsupported algorithm or direction')
    if not np.isfinite(scale) or scale <= 0 or budget < 0 or not np.isfinite(h) or h <= 0:
        raise ValueError('Invalid scale, budget or perturbation')
    n, d = z.shape
    cost = 2*n if method == 'zgd' else 2
    steps = budget // cost
    theta = np.zeros(d)
    rng = np.random.default_rng(seed)
    history = [(0, objective(z, y, theta))]
    for k in range(1, steps + 1):
        u = direction(rng, d, distribution)
        index = int(rng.integers(n)) if method == 'zsgd' else None
        grad = zero_gradient(z, y, theta, u, h, index)
        theta -= (scale/k)*grad
        if k == steps or k % max(1, steps//checkpoints) == 0:
            history.append((k*cost, objective(z, y, theta)))
    return history


def study(budget=30000, repeats=5, data_seed=42, run_seed=100, h=0.01):
    if repeats < 2 or budget < 6000:
        raise ValueError('Require repeats>=2 and budget>=6000')
    z, y = data(seed=data_seed)
    optimum = objective(z, y, np.linalg.lstsq(z, y, rcond=None)[0])
    configurations = [(d, s) for d in ('gaussian','rademacher','sphere') for s in (1., .1, .01)]
    rows, paths = [], {}
    for method in ('zgd','zsgd'):
        for dist, scale in configurations:
            label = f'{method}:{dist}:scale={scale:g}'
            trajectories=[]
            for j in range(repeats):
                history=run(z,y,method=method,distribution=dist,scale=scale,budget=budget,seed=run_seed+j,h=h)
                trajectories.append(history)
                for calls, value in history:
                    rows.append({'method':method,'direction':dist,'scale':scale,'run':j,
                                 'component_value_calls':calls,'objective':value,
                                 'objective_gap':max(0.,value-optimum)})
            paths[label]=trajectories
    return rows, {'ols_minimum':optimum,'budget':budget,'repeats':repeats,
                  'data_seed':data_seed,'run_seed_start':run_seed,'h':h,
                  'scales':[1.,.1,.01], 'directions':['gaussian','rademacher','sphere'],
                  'note':'Exploratory sensitivity only; objective monitoring costs excluded; identical component-function-value budgets.'}


def save(rows, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (output/'q5_sensitivity.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    final=[r for r in rows if r['component_value_calls']==metadata['budget']]
    summaries=[]
    for method in ('zgd','zsgd'):
        for dist in metadata['directions']:
            for scale in metadata['scales']:
                vals=[r['objective'] for r in final if r['method']==method and r['direction']==dist and r['scale']==scale]
                if vals:
                    summaries.append({'method':method,'direction':dist,'scale':scale,'median_final':float(np.median(vals)),
                                      'min_final':float(np.min(vals)),'max_final':float(np.max(vals)),
                                      'runs':len(vals)})
    metadata['final_summaries']=summaries
    (output/'q5_sensitivity.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    fig, axes=plt.subplots(2,3,figsize=(15,9),sharex=True)
    for i,method in enumerate(('zgd','zsgd')):
        for j,dist in enumerate(metadata['directions']):
            ax=axes[i,j]
            for scale in metadata['scales']:
                runs=[]
                for k in range(metadata['repeats']):
                    points=[r for r in rows if r['method']==method and r['direction']==dist and r['scale']==scale and r['run']==k]
                    runs.append(points)
                    # Individual stochastic trajectories remain visible but subdued.
                    ax.plot([r['component_value_calls'] for r in points],
                            [max(r['objective_gap'],1e-14) for r in points],
                            color={1.:'tab:blue',.1:'tab:orange',.01:'tab:green'}[scale],
                            linewidth=.7,alpha=.18)
                # Highlight the median trajectory across seeds at matching checkpoints.
                # No smoothing or clipping of extreme outcomes is performed.
                budgets=sorted(set(p['component_value_calls'] for run in runs for p in run))
                median_gap=[]
                for calls in budgets:
                    vals=[p['objective_gap'] for run in runs for p in run
                          if p['component_value_calls']==calls]
                    median_gap.append(max(float(np.median(vals)),1e-14))
                ax.plot(budgets,median_gap,linewidth=2.1,
                        color={1.:'tab:blue',.1:'tab:orange',.01:'tab:green'}[scale],
                        label=f'c={scale:g} median')
            ax.set_yscale('log');ax.set_title(f'{method.upper()} — {dist}')
            ax.set_xlabel('Component-function-value calls');ax.set_ylabel('Objective gap to OLS (log scale)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Q5 exploratory extension: direction distribution and step-size prefactor')
    fig.tight_layout();fig.savefig(output/'q5_sensitivity.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget',type=int,default=30000)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=Path('results'))
    args=parser.parse_args()
    rows,meta=study(budget=args.budget,repeats=args.repeats)
    save(rows,meta,args.output)
    print(json.dumps(meta['final_summaries'],indent=2))
