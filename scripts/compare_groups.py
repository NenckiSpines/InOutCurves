"""Reusable dataset comparison and CLI used by the three runnable examples.

The paired design aligns identifiers before any statistic is calculated.
Independent comparisons can have unequal sizes and any number of groups >=2.
For >=3 groups the omnibus statistic is the sum of pairwise squared L2
integrals, followed by the same two-group tests and Holm adjustment.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import math
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
import scipy
from .subject_results import SubjectResults
from .permutation_tests import permutation_test
from .posthoc_tests import pairwise_posthoc
from .reference_tests import anova_group_effect,paired_condition_effect
from .distances import l2_squared
from .cache import read_cache,save_cache,write_csv,write_json,sha256_file,project_code_hash
from .paths import ROOT,INPUT_DATA,DUMPS,FIGURES,resolve_path
from .plotting import COLORS,configure_plots

PRESETS={'test':{'permutations':199,'dpi':160},'final':{'permutations':9999,'dpi':300}}


def compare_dataset(data: SubjectResults,groups: Sequence[str],*,design: str='independent',
                    permutations: int=9999,seed: int=20260919,alpha: float=0.05,
                    exact: str='auto',exact_limit: int=50000,require_omnibus: bool=True) -> dict:
    if seed<0 or not 0<alpha<1:
        raise ValueError('Seed must be nonnegative and alpha must be between 0 and 1.')
    arrays,ids=data.analysis_arrays(groups,design)
    result=permutation_test(arrays,data.currents,design=design,permutations=permutations,
                            rng=np.random.default_rng(np.random.SeedSequence([seed,300])),
                            exact=exact,exact_limit=exact_limit,return_distribution=True)
    if design=='paired':
        f,pa=paired_condition_effect(*arrays)
        label='Repeated-measures Condition main effect (paired mean-response t squared)'
    else:
        f,pa=anova_group_effect(*arrays)
        label='Mixed-design ANOVA Group main effect'
    pairs=[]
    for i,j in combinations(range(len(groups)),2):
        pairs.append({'group_1':groups[i],'group_2':groups[j],
                      'l2_squared':l2_squared(arrays[i].mean(0),arrays[j].mean(0),data.currents)})
    posthoc=(pairwise_posthoc(arrays,data.currents,labels=groups,permutations=permutations,seed=seed,
                             alpha=alpha,omnibus_pvalue=result.pvalue,require_omnibus=require_omnibus,
                             exact=exact,exact_limit=exact_limit) if len(groups)>=3 else [])
    return {'groups':list(groups),'design':design,'subject_ids':[list(s) for s in ids],
            'subject_counts':[len(a) for a in arrays],'recording_counts':[sum(r.group_id==g for r in data.results) for g in groups],
            'statistic_definition':('sum over i<j of integral (mean_i-mean_j)^2 dx' if len(groups)>=3 else
                                    'integral (mean_1-mean_2)^2 dx'),
            'omnibus':result.as_dict(include_distribution=True),'alpha':alpha,
            'omnibus_reject':bool(result.pvalue<=alpha),'posthoc':posthoc,'pairwise_distances':pairs,
            'require_omnibus':require_omnibus,
            'anova_comparator':{'name':label,'f_value':f if math.isfinite(f) else 'Infinity','pvalue':pa},
            'assumption':('Within-subject exchangeability of the two condition labels; independent subjects.'
                          if design=='paired' else 'Exchangeability of complete subject curves across groups under the null.'),
            'units':'Response units squared multiplied by stimulus units; no response-unit assumption made.'}


def export_tables(data: SubjectResults,report: dict,folder: Path) -> None:
    groups=report['groups'];arrays,ids=data.analysis_arrays(groups,report['design'])
    summary=[];curves=[]
    for group,arr,block in zip(groups,arrays,ids):
        sem=arr.std(0,ddof=1)/np.sqrt(len(arr))
        for x,mean,se in zip(data.currents,arr.mean(0),sem):
            summary.append({'group':group,'stimulation':float(x),'mean':float(mean),'sem':float(se),'subjects':len(arr)})
        for subject,row in zip(block,arr):
            for x,y in zip(data.currents,row):
                curves.append({'group':group,'subject':subject,'stimulation':float(x),'response':float(y)})
    write_csv(folder/'group_summary.csv',summary)
    write_csv(folder/'subject_curves.csv',curves)
    write_csv(folder/'null_distribution.csv',[{'allocation':i+1,'l2_statistic':v} for i,v in enumerate(report['omnibus']['null_distribution'])])
    write_csv(folder/'pairwise_distances.csv',report['pairwise_distances'])
    if report['posthoc']:write_csv(folder/'posthoc_tests.csv',report['posthoc'])
    small={**report,'omnibus':{k:v for k,v in report['omnibus'].items() if k!='null_distribution'}}
    write_json(folder/'results.json',small)


def plot_comparison(data: SubjectResults,report: dict,args,stem: str) -> str:
    """Lettered panels and captions in the established poster style."""
    import matplotlib.pyplot as plt
    configure_plots()
    arrays,ids=data.analysis_arrays(report['groups'],report['design'])
    groups=report['groups'];x=data.currents;test=report['omnibus'];multi=len(groups)>=3
    fig,axes=plt.subplots(2,2,figsize=(13.4,9.1)) if multi else plt.subplots(1,3,figsize=(16.5,5.8))
    if multi:fig.subplots_adjust(left=0.09,right=0.98,bottom=0.17,top=0.81,hspace=0.5,wspace=0.30)
    else:fig.subplots_adjust(left=0.07,right=0.985,bottom=0.22,top=0.76,wspace=0.28)
    axs=np.ravel(axes)
    title=('Three independent groups: omnibus and post-hoc inference' if multi else
           'Paired measurements from the same subjects' if report['design']=='paired' else
           'Two independent subject groups')
    fig.suptitle(title,fontsize=19,fontweight='bold',y=0.975)
    fig.text(0.5,0.905,f"Source: {data.path.name if data.path else 'in-memory data'} | "+
             ' | '.join(f'{g}: n={len(a)}' for g,a in zip(groups,arrays)),ha='center',fontsize=10.5)
    status=('TEST / EXECUTION CHECK: small requested permutation count' if args.mode=='test' else
            'Complete subject curves; finite-sample permutation inference conditional on exchangeability')
    fig.text(0.5,0.865,status,ha='center',fontsize=10,fontweight='bold')
    for ax in axs:
        ax.grid(axis='y',alpha=0.13);ax.set_axisbelow(True)
    # Panel A always shows subject-level group means, never slice pseudoreplication.
    for i,(g,a) in enumerate(zip(groups,arrays)):
        mean=a.mean(0);sem=a.std(0,ddof=1)/np.sqrt(len(a))
        axs[0].fill_between(x,mean-sem,mean+sem,color=COLORS[i],alpha=0.15,lw=0)
        axs[0].plot(x,mean,color=COLORS[i],lw=2.5,label=g)
    axs[0].set_title('A  Group means and SEM',loc='left',fontweight='bold',pad=12)
    axs[0].set_xlabel(args.stimulation_label);axs[0].set_ylabel(args.response_label);axs[0].legend()
    h=axs[1] if multi else axs[2]
    h.hist(test['null_distribution'],bins=35,color=COLORS[0],alpha=0.75,label=f"{test['method']}: {test['evaluated_permutations']:,} allocations")
    h.axvline(test['statistic'],color=COLORS[1],lw=2,label='Observed statistic')
    h.set_title(('B  Grand-distance null distribution' if multi else 'C  Randomization distribution'),loc='left',fontweight='bold',pad=12)
    h.set_xlabel(r'$L^2_{\mathrm{Grand}}=\sum_{i<j} L^2_{ij}$' if multi else r'Squared distance, $L^2$')
    h.set_ylabel('Allocations (count)');h.legend(fontsize=8)
    if multi:
        pairs=report['pairwise_distances'];names=[f"{r['group_1']} / {r['group_2']}" for r in pairs]
        axs[2].bar(names,[r['l2_squared'] for r in pairs],color=COLORS[:len(pairs)],alpha=0.8)
        axs[2].set_title('C  Contributions to the grand distance',loc='left',fontweight='bold',pad=12)
        axs[2].set_ylabel(r'Pairwise squared distance, $L^2_{ij}$');axs[2].set_xlabel('Group pair')
        post=report['posthoc'];ys=np.arange(len(post))
        raw=np.asarray([r['p_raw'] for r in post]);adj=np.asarray([r['p_holm'] for r in post])
        axs[3].scatter(raw,ys-0.08,color=COLORS[0],s=45,label='Unadjusted')
        axs[3].scatter(adj,ys+0.08,color=COLORS[1],marker='s',s=45,label='Holm-adjusted')
        axs[3].axvline(args.alpha,color='0.2',ls='--',lw=1,label=f'alpha={args.alpha:g}')
        axs[3].set(xscale='log',xlim=(min(raw.min(),args.alpha)/2,1.2),yticks=ys,yticklabels=names)
        axs[3].set_title('D  Pairwise follow-up tests',loc='left',fontweight='bold',pad=12)
        axs[3].set_xlabel('Permutation p-value');axs[3].legend(fontsize=8,loc='best')
        gate=('Omnibus gate passed.' if report['omnibus_reject'] else 'Omnibus gate not passed: no gated pairwise rejection.') if args.require_omnibus else 'Ungated Holm-adjusted pairwise decisions.'
        note=f"Omnibus p = {test['pvalue']:.5g}. Holm correction covers all {len(post)} pairs. {gate}"
        caption=('(A) Group-mean subject curves with SEM. (B) Null distribution of the sum of all pairwise '
                 'squared distances; complete subjects are reassigned while preserving group sizes. '
                 '(C) Observed pairwise contributions (summed, not squared again). '
                 '(D) Pair-specific permutation p-values and Holm-adjusted p-values. '
                 'All pairs are included in the correction. '+gate)
    elif report['design']=='paired':
        d=arrays[1]-arrays[0]
        axs[1].plot(x,d.T,color=COLORS[2],alpha=0.2,lw=0.75)
        sem=d.std(0,ddof=1)/np.sqrt(len(d));mean=d.mean(0)
        axs[1].fill_between(x,mean-sem,mean+sem,color=COLORS[2],alpha=0.15)
        axs[1].plot(x,mean,color=COLORS[2],lw=2.5,label='Mean within-subject difference')
        axs[1].axhline(0,color='0.4',ls=':',lw=1)
        axs[1].set_title('B  Matched subject differences',loc='left',fontweight='bold',pad=12)
        axs[1].set_xlabel(args.stimulation_label);axs[1].set_ylabel(f'{groups[1]} - {groups[0]} (source units)');axs[1].legend(fontsize=8)
        note=f"Paired permutation p = {test['pvalue']:.5g}. Labels swap only within the same subject; stimulus points remain together."
        caption=('(A) Condition means and between-subject SEM. (B) Individual within-subject difference curves and their '
                 'mean plus/minus SEM. (C) Null distribution generated by independently swapping the two complete '
                 'condition curves within each ID-matched subject. This is not a between-subject permutation. '
                 'Missing pairs cause an explicit error; no subjects are silently dropped.')
    else:
        for i,(g,a) in enumerate(zip(groups,arrays)):
            for j,row in enumerate(a):axs[1].plot(x,row,color=COLORS[i],alpha=0.4,lw=0.8,label=g if j==0 else None)
        axs[1].set_title('B  Individual subject curves',loc='left',fontweight='bold',pad=12)
        axs[1].set_xlabel(args.stimulation_label);axs[1].set_ylabel(args.response_label);axs[1].legend()
        note=f"Permutation p = {test['pvalue']:.5g}. Complete subjects are exchanged while retaining the two group sizes."
        caption=('(A) Group means plus/minus SEM calculated across independent subjects. (B) Individual subject-level '
                 'curves, after averaging repeated recordings within subject and condition. (C) Null distribution '
                 'from reallocating complete curves between groups, retaining the original group sizes.')
    fig.text(0.5,0.067,note,ha='center',fontsize=10)
    fig.text(0.5,0.027,'Small-count test runs check implementation; statistical conclusions require a suitable design and adequate precision.',ha='center',fontsize=9)
    args.out.mkdir(parents=True,exist_ok=True)
    for fmt in args.formats:fig.savefig(args.out/f'{stem}.{fmt}',dpi=args.dpi,bbox_inches='tight',pad_inches=0.16)
    plt.close(fig)
    caption+=f" Observed statistic={test['statistic']:.9g}; p={test['pvalue']:.9g}; {test['method']}; "
    caption+=f"{test['evaluated_permutations']:,} allocations evaluated of {test['total_allocations']:,} possible. "
    caption+='Monte Carlo p-values include the +1 correction; exhaustive p-values include the observed allocation and ties. '
    caption+=report['assumption']
    (args.out/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    return caption


def main(argv=None,*,default_data=None,default_groups=None,default_design='independent',
         example_name='custom_comparison') -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=default_data)
    parser.add_argument('--groups',nargs='+',default=default_groups)
    parser.add_argument('--design',choices=['paired','independent'],default=default_design)
    parser.add_argument('--mode',choices=PRESETS,default='test')
    parser.add_argument('--permutations',type=int)
    parser.add_argument('--seed',type=int,default=20260919)
    parser.add_argument('--alpha',type=float,default=0.05)
    parser.add_argument('--exact',choices=['auto','always','never'],default='auto')
    parser.add_argument('--exact-limit',type=int,default=50000)
    parser.add_argument('--no-omnibus-gate',dest='require_omnibus',action='store_false')
    parser.set_defaults(require_omnibus=True)
    parser.add_argument('--regenerate',action='store_true')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg'])
    parser.add_argument('--dpi',type=int)
    parser.add_argument('--response-label',default='Response (source units)')
    parser.add_argument('--stimulation-label',default='Stimulation current (\u00b5A)')
    parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args(argv)
    if args.data is None:parser.error('--data is required.')
    args.data=resolve_path(args.data,INPUT_DATA/'stress_response_ca1.csv')
    args.permutations=args.permutations if args.permutations is not None else PRESETS[args.mode]['permutations']
    args.dpi=args.dpi if args.dpi is not None else PRESETS[args.mode]['dpi']
    if args.permutations<1 or args.exact_limit<0 or args.seed<0 or args.dpi<72 or not 0<args.alpha<1:
        parser.error('Invalid permutation count, exact ceiling, seed, dpi, or alpha.')
    data=SubjectResults.from_csv(args.data)
    args.groups=args.groups or data.get_group_ids()
    # Always validate subject identities, including when reading an existing cache.
    data.analysis_arrays(args.groups,args.design)
    args.cache_dir=resolve_path(args.cache_dir,DUMPS/'examples'/example_name/args.mode)
    args.out=resolve_path(args.out,FIGURES/'examples'/example_name/args.mode)
    if args.cache_dir==args.data.parent:parser.error('Caches need a dedicated output subfolder.')
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    spec={'workflow':'subject_comparison','input_sha256':sha256_file(args.data),'shared_code_sha256':project_code_hash(),
          'numpy':np.__version__,'scipy':scipy.__version__,
          **{k:getattr(args,k) for k in ('groups','design','permutations','seed','alpha','exact','exact_limit','require_omnibus')}}
    path=args.cache_dir/'comparison.pkl';cached=read_cache(path,spec,args.regenerate)
    if cached:
        report=cached['data'];print(f'[cache] {path}')
    else:
        report=compare_dataset(data,args.groups,**{k:getattr(args,k) for k in ('design','permutations','seed','alpha','exact','exact_limit','require_omnibus')})
        save_cache(path,spec,report);print(f'[new] {path}')
    export_tables(data,report,args.cache_dir)
    if not args.skip_figures:plot_comparison(data,report,args,example_name)
    write_json(args.cache_dir/'run_manifest.json',{'spec':spec,'data_path':str(args.data),
                                                  'figure_directory':str(args.out),'mode':args.mode})
    test=report['omnibus']
    print(f"Groups: {', '.join(report['groups'])}; design={args.design}; subjects={report['subject_counts']}")
    print(f"L2{'_Grand' if len(args.groups)>2 else ''}={test['statistic']:.9g}; p={test['pvalue']:.9g}; "
          f"{test['method']}, {test['evaluated_permutations']:,} allocations")
    for row in report['posthoc']:
        print(f"  {row['group_1']} vs {row['group_2']}: p={row['p_raw']:.6g}; Holm={row['p_holm']:.6g}; "
              f"gated rejection={row['reject_after_gate']}")
    print(f'Results: {args.cache_dir}\nFigures: {args.out}')


if __name__=='__main__':main()
