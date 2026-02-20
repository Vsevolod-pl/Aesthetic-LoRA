import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import eval_metrics


class Tester:
    def __init__(self, test_set, bs=50, device='cpu', thr=0.5, tqdm=lambda a:a, preload=True, dataloader_kwargs={}):
        self.device = device
        self.thr = thr
        self.preload = preload
        self.test_set = test_set

        if preload:
            test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
            self.test_batches = []
            for things in tqdm(test_loader):
                self.test_batches.append(things)
        else:
            self.test_batches = DataLoader(test_set, batch_size=bs, shuffle=False, **dataloader_kwargs)

    def predict(self, model, tqdm=lambda a:a):
        test_predict = []
        test_lbls = []
        model.eval()
        with torch.no_grad():
            for things_cpu in tqdm(self.test_batches):
                inputs = [t.to(self.device) for t in things_cpu[:-1]]
                test_predict.append(model(*inputs).squeeze().detach().cpu().numpy())
                test_lbls.append(things_cpu[-1].detach().cpu().numpy())
        test_predict = [np.atleast_1d(x) for x in test_predict]
        test_lbls = [np.atleast_1d(x) for x in test_lbls]
        return np.concatenate(test_predict), np.concatenate(test_lbls)

    def eval(self, model, tqdm=lambda a:a):
        test_predict, test_lbls = self.predict(model, tqdm)
        return eval_metrics(test_predict, test_lbls, data_test=self.test_set.markup, thr=self.thr)
