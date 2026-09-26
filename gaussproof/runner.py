"""Reproducible end-to-end experiment orchestration."""
from pathlib import Path
import itertools
import platform
import time
import hashlib
import numpy as np
import torch
from . import __version__
from .data import load_mnist, make_splits, digest
from .models import initialize, projection_indices, train, predict, conservative_epsilon
from .attacks import output_scores, features, LogisticAttack, rmia, trajectory_scores
from .metrics import evaluate, auc
from .io import write_csv, write_json
from .plots import roc_plot, summary_plots
from .validation import permutation_control, synthetic_controls


def validate_config(c):
    for k in ['train_size','eval_size','calibration_size','shadow_pool_size','population_size',
              'steps','references','projection_dim','sparse_iterations','bootstrap','threads','permutation_repeats']:
        if type(c[k]) is not int or c[k] <= 0: raise ValueError(f'{k} must be a positive integer')
    if c['calibration_size'] < 8: raise ValueError('Need at least 8 calibration samples per class')
    if not 0 < c['delta'] < 1 or c['lr'] <= 0: raise ValueError('Invalid delta or learning rate')
    if not c['seeds'] or not c['sigmas'] or not c['clips'] or not c['batch_sizes']: raise ValueError('Empty sweep')
    if min(c['sigmas']) < 0 or min(c['clips']) <= 0: raise ValueError('Invalid noise/clipping')
    if max(c['batch_sizes']) > c['train_size'] or min(c['batch_sizes']) < 1: raise ValueError('Invalid batch size')
    if len(set(c['seeds'])) != len(c['seeds']): raise ValueError('Seeds must be unique')
    if not 0 <= c['rmia_a'] <= 1 or c['rmia_gamma'] <= 0 or c['sparse_penalty'] < 0:
        raise ValueError('Invalid attack hyperparameters')
    if type(c['control_repeats']) is not int or c['control_repeats'] <= 0: raise ValueError('Invalid control repeats')


def source_digest():
    h=hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob('*.py')):
        h.update(file.name.encode());h.update(file.read_bytes())
    return h.hexdigest()


def run(cfg, data_path, output, log=print):
    validate_config(cfg)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run: {output}')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    start=time.time()
    x,y=load_mnist(data_path)
    write_json(output/'config.json',cfg)
    write_json(output/'provenance.json',dict(version=__version__,source_sha256=source_digest(),python=platform.python_version(),
               torch=torch.__version__,numpy=np.__version__,dataset_sha256=digest(data_path),
               dataset_rows=len(y),device='cpu',trajectory='actual noisy SGD updates',
               adjacency='replace-one',accountant='conservative zCDP; no amplification',
               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    rows, validation, utilities=[],[],[]
    for seed in cfg['seeds']:
        split=make_splits(len(y),cfg,seed)
        write_json(output/f'splits_seed{seed}.json',{k:v.tolist() for k,v in split.items()})
        # Shuffle candidate order independently of membership and use fixed IDs for every attack.
        ids=np.concatenate([split[k] for k in ['eval_member','eval_nonmember','cal_member','cal_nonmember']])
        ids=np.random.default_rng(seed+10).permutation(ids)
        membership=np.isin(ids,split['train']).astype(int)
        eval_mask=np.isin(ids,np.r_[split['eval_member'],split['eval_nonmember']])
        cal_mask=~eval_mask
        # Two disjoint calibration halves: hybrid fitting and threshold selection.
        fit_mask=np.zeros(len(ids),bool)
        for group in ['cal_member','cal_nonmember']:
            fit_mask |= np.isin(ids,split[group][:len(split[group])//2])
        threshold_mask=cal_mask & ~fit_mask
        assert not (fit_mask & eval_mask).any() and not (threshold_mask & fit_mask).any()
        for sigma,clip,batch in itertools.product(cfg['sigmas'],cfg['clips'],cfg['batch_sizes']):
            name=f'seed{seed}_sigma{sigma:g}_clip{clip:g}_B{batch}'
            folder=output/name; folder.mkdir()
            log(f'[{name}] target CNN',flush=True)
            model=initialize(seed)
            coordinates=projection_indices(model,cfg['projection_dim'],seed+20)
            kwargs=dict(steps=cfg['steps'],batch_size=batch,clip=clip,sigma=sigma,lr=cfg['lr'])
            trace=train(model,x,y,split['train'],**kwargs,seed=seed+100,
                        candidate_ids=ids,coordinates=coordinates,
                        progress=lambda msg:log(f'  target {msg}',flush=True))
            torch.save(model.state_dict(),folder/'target.pt')
            np.savez_compressed(folder/'trajectory.npz',**trace,candidate_ids=ids,coordinates=coordinates)
            p=predict(model,x,ids); pop=predict(model,x,split['population'])
            truth=y[ids].numpy(); pop_truth=y[split['population']].numpy()
            ref_candidates,ref_pop,shadow_x,shadow_y=[],[],[],[]
            for ref in range(cfg['references']):
                rseed=seed+1000*(ref+1)
                perm=np.random.default_rng(rseed).permutation(split['shadow_pool'])
                rtrain=perm[:cfg['train_size']]; rout=perm[cfg['train_size']:2*cfg['train_size']]
                # Same initialization distribution and training algorithm, independent random stream.
                shadow=initialize(rseed)
                log(f'[{name}] shadow/reference {ref+1}/{cfg["references"]}',flush=True)
                train(shadow,x,y,rtrain,**kwargs,seed=rseed+100)
                train_p=predict(shadow,x,rtrain); out_p=predict(shadow,x,rout)
                shadow_x.extend([features(train_p,y[rtrain].numpy()),features(out_p,y[rout].numpy())])
                shadow_y.extend([np.ones(len(rtrain)),np.zeros(len(rout))])
                rp=predict(shadow,x,ids); rz=predict(shadow,x,split['population'])
                ref_candidates.append(rp[np.arange(len(ids)),truth])
                ref_pop.append(rz[np.arange(len(pop)),pop_truth])
                torch.save(shadow.state_dict(),folder/f'reference_{ref}.pt')
                write_json(folder/f'reference_{ref}_split.json',dict(train=rtrain.tolist(),nonmember=rout.tolist()))
            shokri=LogisticAttack().fit(np.vstack(shadow_x),np.concatenate(shadow_y))
            scores=output_scores(p,truth)
            scores['shokri']=shokri.score(features(p,truth))
            scores['rmia']=rmia(p[np.arange(len(ids)),truth],np.asarray(ref_candidates),
                                pop[np.arange(len(pop)),pop_truth],np.asarray(ref_pop),cfg['rmia_a'],cfg['rmia_gamma'])
            log(f'[{name}] trajectory decoding',flush=True)
            penalty=cfg['sparse_penalty']*clip*clip
            scores.update(trajectory_scores(trace['dictionary'],trace['releases'],penalty,cfg['sparse_iterations']))
            hybrid_x=np.column_stack([scores['rmia'],scores['gaussproof']])
            hybrid=LogisticAttack().fit(hybrid_x[fit_mask],membership[fit_mask])
            scores['hybrid']=hybrid.score(hybrid_x)
            write_json(folder/'attack_fits.json',dict(shokri=shokri.state(),hybrid=hybrid.state(),
                        hybrid_fit_ids=ids[fit_mask].tolist(),threshold_ids=ids[threshold_mask].tolist()))
            np.savez_compressed(folder/'observations.npz',candidate_ids=ids,probabilities=p,
                                reference_true_probabilities=ref_candidates,population_probabilities=pop,
                                population_reference_true_probabilities=ref_pop)
            context=dict(seed=seed,sigma=sigma,clip=clip,batch_size=batch)
            score_rows=[]
            for i,ident in enumerate(ids):
                role='evaluation' if eval_mask[i] else ('hybrid_fit' if fit_mask[i] else 'threshold_calibration')
                score_rows.append(dict(record_id=int(ident),membership=int(membership[i]),role=role,
                                       **{k:float(v[i]) for k,v in scores.items()}))
            write_csv(folder/'scores.csv',score_rows)
            for attack,score in scores.items():
                threat='black_box' if attack not in ['rero','gaussproof','hybrid'] else ('hybrid' if attack=='hybrid' else 'gradient_trajectory')
                m=evaluate(score[eval_mask],membership[eval_mask],score[threshold_mask],membership[threshold_mask],cfg['bootstrap'],seed)
                rows.append(dict(**context,attack=attack,threat_model=threat,**m))
                control=permutation_control(score[eval_mask],membership[eval_mask],cfg['permutation_repeats'],seed+40)
                validation.append(dict(**context,attack=attack,control='permuted_membership',**control))
            # Dictionaries are from the real training run; noise-only observations may still
            # interact with membership-dependent gradient norms. Report, do not force chance.
            rng=np.random.default_rng(seed+50)
            pure=rng.normal(size=trace['releases'].shape)*max(sigma,1)*clip
            controls=trajectory_scores(trace['dictionary'],pure,penalty,cfg['sparse_iterations'])
            for attack,score in controls.items():
                validation.append(dict(**context,attack=attack,control='real_dictionary_pure_noise',
                                       mean_auc=auc(score[eval_mask],membership[eval_mask]),std_auc=0.0,repeats=1))
            trainp=predict(model,x,split['train']); testp=predict(model,x,split['eval_nonmember'])
            utilities.append(dict(**context,train_accuracy=float((trainp.argmax(1)==y[split['train']].numpy()).mean()),
                 heldout_accuracy=float((testp.argmax(1)==y[split['eval_nonmember']].numpy()).mean()),
                 conservative_epsilon=conservative_epsilon(cfg['steps'],sigma,cfg['delta']),delta=cfg['delta'],
                 steps=cfg['steps'],elapsed_seconds=time.time()-start))
            roc_plot({k:v[eval_mask] for k,v in scores.items()},membership[eval_mask],folder/'figures'/'roc',name)
            write_csv(output/'metrics.csv',rows);write_csv(output/'validation.csv',validation);write_csv(output/'utility.csv',utilities)
            log(f'[{name}] done; held-out accuracy {utilities[-1]["heldout_accuracy"]:.3f}',flush=True)
    write_csv(output/'synthetic_controls.csv',synthetic_controls(repeats=cfg['control_repeats']))
    summary_plots(rows,output/'figures')
    write_json(output/'completion.json',dict(status='complete',conditions=len(utilities),metric_rows=len(rows),elapsed_seconds=time.time()-start))
    log(f'Completed {len(utilities)} conditions in {time.time()-start:.1f}s: {output}',flush=True)
    return rows
