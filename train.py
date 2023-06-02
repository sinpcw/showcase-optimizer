import os
import cv2
import numpy as np
import pandas as pd
import datetime
import timm
import albumentations as A
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score
from timm.scheduler import CosineLRScheduler
from typing import List

try:
    import dadaptation
except ModuleNotFoundError:
    print('Module Not Found: dadaptation')
try:
    from lion_pytorch import Lion
except ModuleNotFoundError:
    print('Module Not Found: lion_pytorch')

class CFG:
    device = 'cuda'
    amp_use = True
    amp_dtype = torch.float16
    epoch = 50
    # epoch = 200
    num_classes = 120
    batch_size = 64
    image_size = 224
    image_dir = 'data/dog-breed-identification/train'
    num_workers = 4
    persistent_workers = (True if os.name == 'nt' else False)
    train_csv = 'sample_train.csv'
    valid_csv = 'sample_valid.csv'
    # Select Optimizer:
    # optimizer = 'Adam'
    # optimizer = 'SAM_Adam'
    # optimizer = 'AdamW'
    # optimizer = 'SAM_AdamW'
    # optimizer = 'DAdaptAdam'
    # optimizer = 'SAM_DAdaptAdam'
    # optimizer = 'Lion'
    # optimizer = 'SAM_Lion'
    optimizer = 'SophiaG'
    # Select Model:
    model_name = 'resnet50'
    # model_name = 'tf_efficientnet_b0'
    # model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k'

def get_optimizer_and_scheduler(model):
    optim = None
    if CFG.optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        base_lr = 1e-4
    if CFG.optimizer == 'SAM_Adam':
        optim = SAM(model.parameters(), torch.optim.Adam, lr=1e-4, rho=0.05)
        base_lr = 1e-4
    if CFG.optimizer == 'AdamW':
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        base_lr = 1e-4
    if CFG.optimizer == 'SAM_AdamW':
        optim = SAM(model.parameters(), torch.optim.AdamW, lr=1e-4, weight_decay=1e-5, rho=0.05)
        base_lr = 1e-4
    if CFG.optimizer == 'DAdaptAdam':
        optim = dadaptation.DAdaptAdam(model.parameters(), lr=1.0)
        base_lr = 1.0
    if CFG.optimizer == 'SAM_DAdaptAdam':
        optim = SAM(model.parameters(), dadaptation.DAdaptAdam, lr=1.0, rho=0.05)
        base_lr = 1.0
    if CFG.optimizer == 'Lion':
        optim = Lion(model.parameters(), lr=1e-5, weight_decay=1e-3)
        base_lr = 1e-5
    if CFG.optimizer == 'SAM_Lion':
        optim = SAM(model.parameters(), Lion, lr=1e-5, weight_decay=1e-3, rho=0.05)
        base_lr = 1e-5
    if CFG.optimizer == 'SophiaG':
        optim = SophiaG(model.parameters(), lr=1e-4, weight_decay=1e-5)
        base_lr = 1e-4
    if optim is None:
        raise NameError('{} is not defined.'.format(CFG.optimizer))
    scheduler = CosineLRScheduler(optim, t_initial=CFG.epoch, lr_min=base_lr * 1e-3, warmup_t=3, warmup_lr_init=base_lr * 1e-2, warmup_prefix=True)
    return optim, scheduler

class SAM(torch.optim.Optimizer):
    """
    SAM - https://arxiv.org/abs/2010.01412
    https://github.com/davda54/sam/blob/main/sam.py
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class SophiaG(torch.optim.Optimizer):
    """
    SophiaG - https://arxiv.org/pdf/2305.14342.pdf
    https://github.com/Liuhong99/Sophia/blob/main/sophia.py
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho = 0.04,
         weight_decay=1e-1, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho, 
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])
                
                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  bs=bs,
                  beta1=beta1,
                  beta2=beta2,
                  rho=group['rho'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def sophiag(params: List[torch.Tensor],
          grads: List[torch.Tensor],
          exp_avgs: List[torch.Tensor],
          hessian: List[torch.Tensor],
          state_steps: List[torch.Tensor],
          capturable: bool = False,
          *,
          bs: int,
          beta1: float,
          beta2: float,
          rho: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_sophiag(params: List[torch.Tensor],
                         grads: List[torch.Tensor],
                         exp_avgs: List[torch.Tensor],
                         hessian: List[torch.Tensor],
                         state_steps: List[torch.Tensor],
                         *,
                         bs: int,
                         beta1: float,
                         beta2: float,
                         rho: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        if capturable:
            step = step_t
            step_size = lr 
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr 
            
            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)

def train_epoch(model, loader, loss_fn, scaler, optim):
    value = 0
    count = 0
    model.train()
    for itr in loader:
        optim.zero_grad()
        x, t = itr
        bs = x.shape[0]
        x = x.to(CFG.device)
        t = t.to(CFG.device)
        if isinstance(optim, SAM):
            optim.step = optim.first_step
        with autocast(enabled=scaler._enabled, dtype=CFG.amp_dtype):
            y = model(x)
            loss = loss_fn(y, t)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        if isinstance(optim, SAM):
            optim.step = optim.second_step
            with autocast(enabled=scaler._enabled, dtype=CFG.amp_dtype):
                y = model(x)
                l = loss_fn(y, t)
            scaler.scale(l).backward()
            scaler.step(optim)
            scaler.update()
        value += bs * loss.item()
        count += bs
    value = value / count if count > 0 else 0
    return value

def valid_epoch(model, loader, loss_fn, scaler):
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.int32)
    value = 0
    count = 0
    model.eval()
    with torch.no_grad():
        actfn = torch.nn.Softmax(dim=1)
        for _, itr in enumerate(loader):
            x, t = itr
            bs = x.shape[0]
            x = x.to(CFG.device)
            t = t.to(CFG.device)
            with autocast(enabled=scaler._enabled, dtype=CFG.amp_dtype):
                y = model(x)
                loss = loss_fn(y, t)
            value += bs * loss.item()
            count += bs
            y = actfn(y)
            y = y.detach().cpu().numpy()
            y = np.argmax(y, axis=1)
            t = t.detach().cpu().numpy()
            y_pred = np.append(y_pred, y)
            y_true = np.append(y_true, t)
    value = value / count if count > 0 else 0
    score = accuracy_score(y_true, y_pred)
    return value, score

def get_train_loader(dataframe: pd.DataFrame) -> torch.utils.data.DataLoader:
    train_aug = A.Compose([
        A.Resize(CFG.image_size, CFG.image_size, p=1.00),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
    loader = torch.utils.data.DataLoader(
        ImageDataset(dataframe, train_aug),
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=CFG.num_workers,
        persistent_workers=CFG.persistent_workers,
    )
    return loader

def get_valid_loader(dataframe: pd.DataFrame) -> torch.utils.data.DataLoader:
    valid_aug = A.Compose([
        A.Resize(CFG.image_size, CFG.image_size, p=1.00),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
    loader = torch.utils.data.DataLoader(
        ImageDataset(dataframe, valid_aug),
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers,
        persistent_workers=CFG.persistent_workers,
    )
    return loader

class ImageDataset(Dataset):
    def __init__(self, dataframe, augment):
        self.dataframe = dataframe
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        col_img = 0
        col_lbl = 2
        imf = os.path.join(CFG.image_dir, self.dataframe.iat[idx, col_img]) + '.jpg'
        img = self.__getimage__(imf)
        img = self.augment(image=img)['image'].astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        lbl = self.dataframe.iat[idx, col_lbl]
        return img, lbl

    def __getimage__(self, name):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def get_timestamp():
    t = datetime.datetime.now()
    return t.strftime('%Y/%m/%d %H:%M:%S')

def scheduler_step(scheduler, epoch):
    if scheduler is None:
        return
    if isinstance(scheduler, CosineLRScheduler):
        scheduler.step(epoch)
    else:
        scheduler.step()
    
def main():
    train_loader = get_train_loader(pd.read_csv(CFG.train_csv))
    valid_loader = get_valid_loader(pd.read_csv(CFG.valid_csv))
    model = timm.create_model(CFG.model_name, num_classes=CFG.num_classes, pretrained=True).to(CFG.device)
    if CFG.device != 'cpu' and CFG.amp_use:
        model.forward = autocast()(model.forward)
    optim, scheduler = get_optimizer_and_scheduler(model)
    scaler = GradScaler(enabled=CFG.amp_use)
    loss_fn = torch.nn.CrossEntropyLoss()
    # torch.autograd.set_detect_anomaly(True)
    observe = {
        'best_score' : 0,
        'best_epoch' : 0,
        'set_time' : datetime.datetime.now(),
        'end_time' : None
    }
    for e in range(1, 1+CFG.epoch):
        train_loss = train_epoch(model, train_loader, loss_fn, scaler, optim)
        valid_loss, valid_score = valid_epoch(model, valid_loader, loss_fn, scaler)
        scheduler_step(scheduler, e)
        print('[{}] epoch: {:2}, train_loss: {:.4e}, valid_loss: {:.4e}, accuracy: {:.3f}'.format(get_timestamp(), e, train_loss, valid_loss, valid_score))
        if valid_score > observe['best_score']:
            observe['best_score'] = valid_score
            observe['best_epoch'] = e
    observe['end_time'] = datetime.datetime.now()
    # memo:
    print('[{}] best accuracy: {:.3f} ({}epoch)'.format(get_timestamp(), observe['best_score'], observe['best_epoch']))
    print('[{}] training time: {} sec.'.format(get_timestamp(), (observe['end_time'] - observe['set_time']).seconds))
    print('[{}]    model name: {}'.format(get_timestamp(), CFG.model_name))
    print('[{}]     optimizer: {}'.format(get_timestamp(), CFG.optimizer))

""" エントリポイント """
if __name__ == "__main__":
    main()
