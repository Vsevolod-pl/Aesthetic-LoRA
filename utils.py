import json
import torch
import xxhash
import asyncio
import aiofiles
import numpy as np
import multiprocessing
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, mutual_info_score

def load_json(fname, *args, **kwargs):
    with open(fname) as f:
        return json.load(f, *args, **kwargs)


def save_json(jd, fname, *args, indent=4, **kwargs):
    with open(fname, 'w') as f:
        json.dump(jd, f, *args, indent=indent, **kwargs)


def to_logits(x):
    return np.log(x) - np.log(1-x)


def from_logits(x):
    return 1/(1+np.exp(-x))


def acc_auc(pred, gt, N_bins=100000, range=(0, 1)):
    try:
        cnts, vals = np.histogram(np.abs(pred - gt), bins=N_bins, range=range)
    except:
        return float('nan')
    # vals, cnts = np.unique(np.abs(pred - gt), return_counts=True)
    cntcumsum = cnts.cumsum()
    return cntcumsum.sum() / (cntcumsum.max()*len(cntcumsum))


def acc_scores(data_test, pred, gt):
    pass

def top_n_accuracy(data_test, pred, gt, thr=0.5):
    scene_votes = dict()
    scene_votes_pred = dict()
    name = data_test.name.to_numpy()
    left = data_test.left.to_numpy()
    right = data_test.right.to_numpy()
    for i in range(len(data_test)):
        scene_votes[name[i]] = scene_votes.get(name[i], dict())
        scene_votes[name[i]][left[i]] = scene_votes[name[i]].get(left[i], 0) + (gt[i] > thr)
        scene_votes[name[i]][right[i]] = scene_votes[name[i]].get(right[i], 0) + (gt[i] < thr)
    
        scene_votes_pred[name[i]] = scene_votes_pred.get(name[i], dict())
        scene_votes_pred[name[i]][left[i]] = scene_votes_pred[name[i]].get(left[i], 0) + (pred[i] > thr)
        scene_votes_pred[name[i]][right[i]] = scene_votes_pred[name[i]].get(right[i], 0) + (pred[i] < thr)
    
    ordered_mos = []
    ordered_pred = []
    for scene in scene_votes:
        ordered_mos.append(sorted(scene_votes[scene], key=lambda name: -scene_votes[scene][name]))
        ordered_pred.append(sorted(scene_votes[scene], key=lambda name: -scene_votes_pred[scene][name]))

    accs = []
    for n in range(len(ordered_mos[0])+1):
        accs.append(np.mean([ordered_mos[i][:n] == ordered_pred[i][:n] for i in range(len(ordered_mos))]))
    return accs


def bt_scores(gt, pred, data_test, eps=1e-5):
    scene_votes = dict()
    scene_votes_pred = dict()
    name = data_test.name.to_numpy()
    left = data_test.left.to_numpy()
    right = data_test.right.to_numpy()
    styls = np.unique(left)
    st2ind = {name:i for i, name in enumerate(styls)}
    sz = len(styls)
    transform = np.zeros((sz*sz, sz))
    n = 0
    for i in range(sz):
        for j in range(sz):
            transform[n, i] += 1
            transform[n, j] += -1
            n += 1
    
    for i in range(len(data_test)):
        scene_votes[name[i]] = scene_votes.get(name[i], np.zeros((sz, sz))+0.5)
        scene_votes[name[i]][st2ind[left[i]], st2ind[right[i]]] = min(max(gt[i], eps), 1-eps)
        scene_votes[name[i]][st2ind[right[i]], st2ind[left[i]]] = min(max(1-gt[i], eps), 1-eps)
        
        scene_votes_pred[name[i]] = scene_votes_pred.get(name[i], np.zeros((sz, sz))+0.5)
        scene_votes_pred[name[i]][st2ind[left[i]], st2ind[right[i]]] = min(max(pred[i], eps), 1-eps)
        scene_votes_pred[name[i]][st2ind[right[i]], st2ind[left[i]]] = min(max(1-pred[i], eps), 1-eps)
    for sc in scene_votes:
        scene_votes[sc] = np.log(scene_votes[sc]/(1-scene_votes[sc]))
        scene_votes_pred[sc] = np.log(scene_votes_pred[sc]/(1-scene_votes_pred[sc]))
    scene_bt = dict()
    pred_scene_bt = dict()
    for sc in scene_votes:
        BT, _, _, _ = np.linalg.lstsq(transform, scene_votes[sc].flatten())
        BT -= BT.min()
        scene_bt[sc] = {styls[i]:BT[i] for i in range(sz)}
        
        BT, _, _, _ = np.linalg.lstsq(transform, scene_votes_pred[sc].flatten())
        BT -= BT.min()
        pred_scene_bt[sc] = {styls[i]:BT[i] for i in range(sz)}        
    return scene_bt, pred_scene_bt

def eval_metrics(pred, gt, data_test=None, thr=0.5, thr_left=0.35, thr_right=0.65):
    res = {
        f'Accuracy({thr:.3})': float(((pred > thr) == (gt > thr)).mean()),
        'Triple Accuracy': np.mean(((pred < thr_left) & (gt < thr_left)) | ((thr_left <= pred) & (pred <= thr_right) & (thr_left <= gt) & (gt <= thr_right)) | ((thr_right < pred) & (thr_right < gt))),
        'Spearmanr': float(stats.spearmanr(pred, gt).statistic),
        'Kendall': float(stats.kendalltau(pred, gt).statistic),
        f'ROC AUC({thr:.3})': float(roc_auc_score((gt > thr), pred)),
        'Accuracy AUC': acc_auc(pred, gt),
        f'mAP({thr:.3})': float(average_precision_score((gt > thr), pred)),
        'Crossentropy': (-np.log(pred+1e-20) * gt).mean(),
        'BCE': torch.nn.functional.binary_cross_entropy(torch.from_numpy(pred).to(torch.float32), torch.from_numpy(gt).to(torch.float32)),
        # 'KL mutual score': mutual_info_score(gt, pred)
    }
    if data_test is not None:
        rank_accs = top_n_accuracy(data_test, pred, gt)
        for rank in range(1, len(rank_accs)):
            res[f'Top {rank} accuracy'] = rank_accs[rank]

        gt_scene_bt, pred_scene_bt = bt_scores(gt, pred, data_test)
        gt_scores = []
        pred_scores = []
        mets_avg_scene = {
            'BT Pearson': [],
            'BT Spearman': [],
            'BT Kendall': [],
            'BT Top-1 Accuracy': [],
        }
        for sc in gt_scene_bt:
            for st in gt_scene_bt[sc]:
                gt_scores.append(gt_scene_bt[sc][st])
                pred_scores.append(pred_scene_bt[sc][st])
            gt_per_scene = [gt_scene_bt[sc][st] for st in gt_scene_bt[sc]]
            preds_per_scene = [pred_scene_bt[sc][st] for st in gt_scene_bt[sc]]
            mets_avg_scene['BT Pearson'].append(stats.pearsonr(gt_per_scene, preds_per_scene).statistic)
            mets_avg_scene['BT Spearman'].append(stats.spearmanr(gt_per_scene, preds_per_scene).statistic)
            mets_avg_scene['BT Kendall'].append(stats.kendalltau(gt_per_scene, preds_per_scene).statistic)
            mets_avg_scene['BT Top-1 Accuracy'].append(np.argmax(gt_per_scene)==np.argmax(preds_per_scene))
        res['BT Spearman'] = float(stats.spearmanr(pred_scores, gt_scores).statistic)
        res['BT Pearson'] = float(stats.pearsonr(pred_scores, gt_scores).statistic)
        res['BT Kendall'] = float(stats.kendalltau(pred_scores, gt_scores).statistic)
        res['BT Top-1 Accuracy'] = float(np.nanmean(mets_avg_scene['BT Top-1 Accuracy']))
        for m in mets_avg_scene:
            res[m+' per scene'] = np.nanmean(mets_avg_scene[m])
    return res


def store_results(name, metrics, comment=None, path='./best_results.json'):
    try:
        res = load_json(path)
    except:
        res = dict()
    res[name] = metrics
    if comment:
        res[name]['comment'] = comment
    save_json(res, path)


def format_results():
    meta = load_json('best_results.json')
    names = list(meta)
    maxlen = max([len(s) for s in names])
    for c in names:
        scorr, kcorr, acc = meta[c].values()
        print(f'{c:{maxlen}} - Spearman: {scorr: .3f}, Kendall: {kcorr: .3f}, Accuracy: {acc: .3f}')

def preprocess_frame(frame, bad_params, drop_original=True, diff=False, dropna=True):
    frame = frame.copy()
    if drop_original:
        not_original_mask = np.logical_and(frame.left != 'original', frame.right != 'original')
        frame = frame[not_original_mask]
    
    if diff:
        for col in frame.columns:
            if '_left' in col:
                col_name = col[:-5]
                if col not in bad_params:
                    frame[col_name] = frame[col_name+'_left'] - frame[col_name+'_right']
                frame.drop(columns=[col_name+'_left', col_name+'_right'], inplace=True)
            elif '_right' not in col:
                col_name = col
                if col_name in bad_params:
                    frame.drop(columns=[col_name], inplace=True)
    else:
        to_drop = [name+'_left' if name+'_left' in frame.columns else name for name in bad_params]
        to_drop += [name+'_right' for name in bad_params if name+'_right' in frame.columns]
        frame.drop(columns=to_drop, inplace=True)
        
    if dropna:
        frame.dropna(axis=1, inplace=True)
    
    return frame

def load_pairs(frame, init=None, both_counter=None):
    if init is None:
        init = dict()
    if both_counter is None:
        both_counter = dict()
    res = init
    golden_null = frame['GOLDEN:result'].isnull()
    input_null = frame['INPUT:image_left'].isnull()
    for i in range(len(frame)):
        if golden_null[i] and not input_null[i]:
            style_left = frame['INPUT:image_left'][i].split('/')[-2]
            img_left = frame['INPUT:image_left'][i].split('/')[-1].rstrip()
            style_right = frame['INPUT:image_right'][i].split('/')[-2]
            img_right = frame['INPUT:image_right'][i].split('/')[-1].rstrip()
        
            assert img_right == img_left

            name = img_left
            ans = frame['OUTPUT:result'][i]
            left_right = (style_left, style_right)
            
            if name not in res:
                res[name] = dict()

            if name not in both_counter:
                both_counter[name] = dict()

            if left_right not in res[name]:
                res[name][left_right] = 0

            if left_right not in both_counter[name]:
                both_counter[name][left_right] = 0

            if ans == 'LEFT' or ans == 'RIGHT':
                res[name][left_right] = res[name].get(left_right, 0) + (ans=='LEFT') - (ans=='RIGHT')
            elif ans == 'BOTH':
                both_counter[name][left_right] = both_counter[name].get(left_right, 0) + 1
                
    return res, both_counter

def load_votes(frame, answer=None):
    if answer is None:
        answer = dict()
    golden_null = frame['GOLDEN:result'].isnull()
    input_null = frame['INPUT:image_left'].isnull()
    for i in range(len(frame)):
        if golden_null[i] and not input_null[i]:
            style_left = frame['INPUT:image_left'][i].split('/')[-2]
            img_left = frame['INPUT:image_left'][i].split('/')[-1].rstrip()
            style_right = frame['INPUT:image_right'][i].split('/')[-2]
            img_right = frame['INPUT:image_right'][i].split('/')[-1].rstrip()
        
            assert img_right == img_left

            name = img_left
            ans = frame['OUTPUT:result'][i]
            left_right = (style_left, style_right)
            
            if name not in answer:
                answer[name] = dict()

            if left_right not in answer[name]:
                answer[name][left_right] = dict()

            answer[name][left_right][ans] = answer[name][left_right].get(ans, 0) + 1
                
    return answer


async def process_file(file_path, func):
    async with aiofiles.open(file_path, 'rb') as f:
        return func(await f.read(), file_path)


async def gather_procs(batch, func):
    return await asyncio.gather(*[process_file(path, func) for path in batch])


def process_batch(batch, func, return_list) -> None:
    return_list += asyncio.run(gather_procs(batch, func))


def process_files(files, func, max_concurrent=50, num_procs=25):
    with multiprocessing.Manager() as manager:
        return_list = manager.list()
        batches = [(files[i:i+max_concurrent], func, return_list) for i in range(0, len(files), max_concurrent)]
        with multiprocessing.Pool(processes=num_procs) as pool:
            pool.starmap(process_batch, batches)
        return list(return_list)
