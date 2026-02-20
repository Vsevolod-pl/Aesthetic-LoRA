import torch
from pathlib import Path

from utils import eval_metrics


def fix_name(start, allowed=None, symbol_to_fix='_'):
    if allowed is None:
        allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. :/')
    res = start
    for banned in set(start) - allowed:
        res = res.replace(banned, symbol_to_fix)
    return res


def log_metrics(metrics, epoch, logging_function):
    for metric, value in metrics.items():
        logging_function(key=fix_name(metric), value=value, step=epoch)


def train_step(train_loader, optimizer, model, criterion, num_model_params=None, device='cpu'):
    if num_model_params is None:
        num_model_params = sum(param.numel() for param in model.parameters())
    running_loss = 0.0

    model.train()
    stats = {'cnt':0}
    for things_cpu in train_loader:
        things = [t.to(device) for t in things_cpu]
        labels = things[-1].float()
        inputs = things[:-1]

        optimizer.zero_grad()
        probability = model(*inputs).squeeze()
        loss = criterion(probability, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_metrics = eval_metrics(probability.detach().cpu().numpy(), labels.detach().cpu().numpy())
        for k, value in batch_metrics.items():
            stats[k] = stats.get(k, 0) + value*len(labels)
        stats['cnt'] += len(labels)

    stats = {'train '+k:v/stats['cnt'] for k,v in stats.items() if k!='cnt'}

    max_abs_grad = 0
    mean_abs_grad = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                abs_ps = param.grad.data.abs()
                max_abs_grad = max(max_abs_grad, abs_ps.max().item())
                mean_abs_grad += abs_ps.sum().item()
        mean_abs_grad /= num_model_params

    stats['avg_epoch_loss'] = running_loss/len(train_loader)
    stats['mean_abs_grad'] = mean_abs_grad
    stats['max_abs_grad'] = max_abs_grad

    return stats


def save_checkpoint(exp_name, model, opt, sched, metrics, epoch, path2save='./weights/'):
    Path(f'{path2save}/{exp_name}').mkdir(parents=True, exist_ok=True)
    all_state_dict = {
        'epoch': epoch,
        'metrics': metrics,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': sched.state_dict(),
    }
    torch.save(all_state_dict, f'{path2save}/{exp_name}/{exp_name}_{epoch}.pth')


def train_loop(exp_name,
               train_loader,
               tester,
               criterion,
               model,
               optimizer,
               sched,
               device,
               num_epochs,
               save_freq,
               path2save='./weights/',
               logging_func=lambda key, value, step: None,
               trg=range,
               start_epoch=0):
    hist = []
    num_model_params = sum(param.numel() for param in model.parameters())
    for epoch in trg(start_epoch, num_epochs):
        sched.step()
        params2log = train_step(train_loader, optimizer, model, criterion, num_model_params, device)

        if epoch % save_freq == save_freq-1:
            for metric, value in tester.eval(model).items():
                params2log[metric] = value

            save_checkpoint(exp_name, model, optimizer, sched, params2log, epoch, path2save=path2save)
        hist.append(params2log)
        log_metrics(params2log, epoch, logging_func)
    return hist
