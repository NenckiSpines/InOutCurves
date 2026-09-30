"""Export the specified population calibration and descriptive diagnostics.

python -m scripts.calibrate_variability --model within-group
This replaces the executable scratch calibration code. It does not claim a
best-fitting distribution from the missing original distribution-selection tool.
"""
import argparse
from pathlib import Path
import numpy as np
from .paths import ROOT,INPUT_DATA,DUMPS,FIGURES,resolve_path
from .simulation import load_source,calibrate
from .cache import write_json,sha256_file
from .plotting import configure_plots,COLORS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=INPUT_DATA/'ltp_paired.csv')
    p.add_argument('--groups',nargs=2,default=['PRE','POST'])
    p.add_argument('--model',choices=['within-group','archive'],default='within-group')
    p.add_argument('--out',type=Path)
    args=p.parse_args(argv)
    data=load_source(resolve_path(args.data,INPUT_DATA/'ltp_paired.csv'),*args.groups)
    model=calibrate(data,args.model,ROOT)
    folder=DUMPS/'calibration'/args.model;folder.mkdir(parents=True,exist_ok=True)
    output=resolve_path(args.out,FIGURES/'calibration'/args.model);output.mkdir(parents=True,exist_ok=True)
    write_json(folder/'calibration.json',{'model':args.model,'description':model.description,
        'gain_location':model.gain_location,'gain_scale':model.gain_scale,'noise_sd':model.noise_sd,
        'input_sha256':sha256_file(data.path),'recordings_per_subject':model.recordings_per_subject,
        'note':'Fixed selected model, not an automated best-distribution claim.'})
    np.savez_compressed(folder/'calibration_arrays.npz',gains=model.calibration_gains,
                       residuals=model.calibration_residuals,x=model.x,templates=model.templates)
    configure_plots()
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12.8,5.1));fig.subplots_adjust(bottom=0.23,top=0.78,wspace=0.28)
    fig.suptitle('Empirical gain and residual-variability diagnostics',fontsize=18,fontweight='bold')
    ax[0].hist(model.calibration_gains,bins=15,density=True,color=COLORS[0],alpha=0.6,label='Within-condition empirical gains')
    grid=np.linspace(model.gain_location,max(model.calibration_gains.max(),model.gain_location+3*model.gain_scale),200)
    if model.gain_scale>0:
        density=np.exp(-(grid-model.gain_location)/model.gain_scale)/model.gain_scale
        ax[0].plot(grid,density,color=COLORS[1],label=f'{args.model} gain density')
    ax[0].set(title='A  Multiplicative gain',xlabel='Gain factor',ylabel='Density');ax[0].legend(fontsize=8)
    ax[1].plot(model.x,model.noise_sd,color=COLORS[0],marker='o')
    ax[1].set(title='B  Additive residual variability',xlabel='Stimulation current (\u00b5A)',ylabel='Residual standard deviation (source units)')
    for a in ax:a.grid(axis='y',alpha=0.13)
    fig.text(0.5,0.07,'The archive mode retains historical pooled-template parameters; within-group mode removes condition effects first.',ha='center',fontsize=9)
    for fmt in ('png','pdf','svg'):fig.savefig(output/f'calibration_diagnostics.{fmt}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    (output/'caption.txt').write_text('Descriptive calibration checks. The gain histogram is based on deviations from condition-specific means. '
        'The overlay and residual standard deviations show the selected generation model. In archive mode these parameters '
        'are historical pooled-template estimates, not a new fit to the displayed within-condition gains. No model-selection '
        'ranking or goodness-of-fit p-value is claimed.\n',encoding='utf-8')
    print(f'Calibration: {folder}\nFigure: {output}')


if __name__=='__main__':main()
