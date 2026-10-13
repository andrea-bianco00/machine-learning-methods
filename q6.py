"""ENSAI Q6: derivative-free linear and one-ReLU regression.

Each ZSGD update uses 2 component-function calls; ZGD uses 2*n.
Full training diagnostics and held-out evaluations are not optimizer calls.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def generate(n=3000, p=8, noise_sd=.5, seed=42):
    if n < 2 or p < 2 or not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError('invalid data parameters')
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,p))
    y=x[:,0]+np.maximum(x[:,1],0.)+noise_sd*rng.normal(size=n)
    return x,y


def predict(x, theta, model):
    p=x.shape[-1]
    if model=='linear' and theta.shape==(p,):
        return x@theta
    if model=='flexible' and theta.shape==(2*p,):
        return x@theta[:p]+np.maximum(x@theta[p:],0.)
    raise ValueError('invalid model or parameter dimensions')


def loss(x,y,theta,model):
    return float(np.mean((predict(x,theta,model)-y)**2))


def initial_theta(p,model):
    if model=='linear':
        return np.zeros(p)
    if model=='flexible':
        theta=np.zeros(2*p)
        theta[p+1]=.1  # avoid ReLU kink at gamma=0
        return theta
    raise ValueError('invalid model')


def direction(rng,d):
    # E[U U^T]=I; ||U||^2=d
    return rng.choice(np.array([-1.,1.]),size=d)


def estimate(x,y,theta,u,model,h=.01):
    if not np.isfinite(h) or h<=0:
        raise ValueError('invalid h')
    return (loss(x,y,theta+h*u,model)-loss(x,y,theta-h*u,model))/(2*h)*u


def optimize(x,y,model,method,budget,seed=100,step_scale=.03,h=.01,checkpoints=40):
    if method not in ('zgd','zsgd') or model not in ('linear','flexible'):
        raise ValueError('invalid model/method')
    if not isinstance(budget,(int,np.integer)) or budget<0:
        raise ValueError('invalid budget')
    if not np.isfinite(step_scale) or step_scale<=0 or not np.isfinite(h) or h<=0:
        raise ValueError('invalid step or h')
    if checkpoints<=0:
        raise ValueError('invalid checkpoints')
    n,p=x.shape
    theta=initial_theta(p,model)
    cost=2*n if method=='zgd' else 2
    steps=budget//cost
    rng=np.random.default_rng(seed)
    history=[{'calls':0,'train_mse':loss(x,y,theta,model)}]
    for k in range(1,steps+1):
        u=direction(rng,len(theta))
        if method=='zgd':
            g=estimate(x,y,theta,u,model,h)
        else:
            i=int(rng.integers(n))
            g=estimate(x[i:i+1],y[i:i+1],theta,u,model,h)
        theta-=step_scale/np.sqrt(k)*g
        if k==steps or k%max(1,steps//checkpoints)==0:
            history.append({'calls':k*cost,'train_mse':loss(x,y,theta,model)})
    return theta,history,{'updates':steps,'oracle_calls':steps*cost,
                          'unused_budget':budget-steps*cost,'cost_per_update':cost}


def experiment(budget=120000,repeats=5,train_seed=42,test_seed=123,step_scale=.03,h=.01):
    if budget<6000 or repeats<2 or train_seed==test_seed:
        raise ValueError('budget>=6000, repeats>=2, distinct seeds required')
    x,y=generate(seed=train_seed)
    xt,yt=generate(n=1000,seed=test_seed)
    oracle_theta=np.zeros(2*x.shape[1])
    oracle_theta[0]=1
    oracle_theta[x.shape[1]+1]=1
    oracle_risk=loss(xt,yt,oracle_theta,'flexible')
    ols=np.linalg.lstsq(x,y,rcond=None)[0]
    records=[];traces=[]
    for model,method in [('flexible','zgd'),('flexible','zsgd'),('linear','zsgd')]:
        for run in range(repeats):
            theta,hist,info=optimize(x,y,model,method,budget,seed=100+run,
                                      step_scale=step_scale,h=h)
            for point in hist:
                traces.append({'model':model,'method':method,'run':run,**point})
            records.append({'model':model,'method':method,'run':run,
                            'train_mse':loss(x,y,theta,model),
                            'test_mse':loss(xt,yt,theta,model),**info})
    meta={'budget_per_run':budget,'repeats':repeats,'n_train':len(y),'n_test':len(yt),
          'p':x.shape[1],'noise_sd':.5,'train_seed':train_seed,'test_seed':test_seed,
          'h':h,'step_size':'step_scale/sqrt(k)','step_scale':step_scale,
          'direction':'Rademacher (E[UU^T]=I)',
          'true_model_test_mse':oracle_risk,'noise_variance':.25,
          'linear_ols_train_mse':loss(x,y,ols,'linear'),
          'linear_ols_test_mse':loss(xt,yt,ols,'linear'),
          'diagnostics':'Full training losses and test evaluations excluded from optimization oracle budget',
          'caution':'Test data used only for final evaluation, not for tuning.',
          'summaries':[]}
    for model,method in [('flexible','zgd'),('flexible','zsgd'),('linear','zsgd')]:
        subset=[r for r in records if r['model']==model and r['method']==method]
        meta['summaries'].append({'model':model,'method':method,
             'median_train_mse':float(np.median([r['train_mse'] for r in subset])),
             'median_test_mse':float(np.median([r['test_mse'] for r in subset])),
             'min_test_mse':float(min(r['test_mse'] for r in subset)),
             'max_test_mse':float(max(r['test_mse'] for r in subset)),
             'updates_per_run':subset[0]['updates'],
             'calls_per_run':subset[0]['oracle_calls']})
    return traces,records,meta


def save(traces,records,meta,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'q6.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    for filename,rows in [('q6_runs.csv',records),('q6_trajectories.csv',traces)]:
        with (output/filename).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=[('flexible','zgd'),('flexible','zsgd'),('linear','zsgd')]
    for model,method in labels:
        subset=[r for r in traces if r['model']==model and r['method']==method and r['run']==0]
        axes[0].plot([r['calls'] for r in subset],[r['train_mse'] for r in subset],label=f'{model} {method}')
    axes[0].set(xlabel='Component-function evaluations',ylabel='Train MSE',title='Optimization (first repeat)')
    axes[0].set_yscale('log');axes[0].legend();axes[0].grid(alpha=.2)
    names=[f'{m}\n{a.upper()}' for m,a in labels]
    vals=[[r['test_mse'] for r in records if r['model']==m and r['method']==a] for m,a in labels]
    axes[1].boxplot(vals,tick_labels=names)
    axes[1].axhline(meta['true_model_test_mse'],ls='--',color='black',label='True DGP parameters')
    axes[1].axhline(meta['linear_ols_test_mse'],ls=':',color='gray',label='Linear OLS')
    axes[1].set(ylabel='Independent test MSE (1000 samples)',title='Generalization across seeds')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'q6_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget',type=int,default=120000)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--step-scale',type=float,default=.03)
    parser.add_argument('--output',type=Path,default=Path('results'))
    args=parser.parse_args()
    traces,records,meta=experiment(budget=args.budget,repeats=args.repeats,step_scale=args.step_scale)
    save(traces,records,meta,args.output)
    print(json.dumps(meta,indent=2))
