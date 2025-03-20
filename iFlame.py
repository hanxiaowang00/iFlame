import glob
import os
# os.chdir('/home/wangh0t/treegen')
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
import math
import trimesh  # 确保已安装 trimesh 库
import numpy as np
from collections import OrderedDict
import pickle
import numpy as np
# from scipy.special import comb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import random
# from models_ae1 import *
from flash_attn.modules.mha import MHA
from torch.cuda.amp import autocast, GradScaler
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

# from tqdm import tqdm
import wandb
from datasetmy import TrianglesDataset
import re
import json
from typing import List, Optional, Tuple
import subprocess
import time
import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import math
from transformer import *
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from flash_attn.modules.mlp import  GatedMlp
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
import sys
block_size=8
offset_size=16
dic=129
fnnn=2816
# desired_faceso=262
pad_symbol=dic
start_symbol = dic+1  # 开始符号索引
end_symbol = dic+2    # 结束符号索引
from pathlib import Path
# allmaxlen=100
maaa=-100
lengthprompt=50*9
allmaxlen=769
name='shapenet'
argvname=name#sys.argv[1]
# 0.001 muon  
argvlr='0.001'#sys.argv[2]
optname='muon'#sys.argv[3]
class Args:
    def __init__(self):
        self.weight_decay = 0.1  # Default weight decay
        self.lr = float(argvlr)      # Absolute learning rate (to be computed)
        self.blr = 1e-4          # Base learning rate
        self.layer_decay = 0.75   # Layer-wise learning rate decay
        self.min_lr = 1.28e-6        # Minimum learning rate
        self.warmup_epochs =10    # Number of warmup epochs
        self.epochs =10      # Total number of training epochs
        self.batch_size = 48#40#57   # Batch size
        # self.world_size = 4       # Number of dist5ributed processes (adjust if using DDP)

args = Args()


from torch.optim.lr_scheduler import LambdaLR    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=10000, lr_start=1e-4, lr_end=1e-5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = lr_end + (lr_start - lr_end) * cosine_decay
        return lr / lr_start  # 归一化
    return LambdaLR(optimizer, lr_lambda)
def accuracy_(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=-1)

    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer


def get_nvidia_smi():

    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return f"Error running nvidia-smi: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Exception occurred while running nvidia-smi: {str(e)}"


def discretizeit(vertices, num_tokens=256):

    vertices_scaled = np.clip(vertices + 0.5, 0, 1) * num_tokens
    vertices_quantized = np.round(vertices_scaled).astype(int)
    
    return vertices_quantized


def inverse_discretize(vertices_quantized, num_tokens=256):

    vertices_normalized = vertices_quantized.astype(float) / num_tokens - 0.5
    
    return vertices_normalized

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss




def get_optimizer( model,optimizer_name= "muon", lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "output_proj" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "output_proj" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"
def set_random_seed(rank, base_seed=3407):

    seed = base_seed + rank  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_mesh_as_obj(vertices_faces, file_path):
    """
    Save mesh data with shape (n, 9) to a .obj file.
    Each row represents a face with 3 vertices (3 coordinates per vertex).

    :param vertices_faces: The mesh data with shape (n, 9) where each row contains 9 values (3 vertices, each with 3 coordinates)
    :param file_path: The path where the .obj file will be saved
    """
    with open(file_path, 'w') as f:
        # Write vertices (each vertex consists of 3 values)
        vertex_count = 0  # To keep track of the vertex index
        # vertices_faces=vertices_faces.reshape(-1,9)
        vertices_faces=vertices_faces[:vertices_faces.shape[0]-vertices_faces.shape[0]%9].reshape(-1,9)
        
        for row in vertices_faces:
            for i in range(0, 9, 3):
                # Extract each vertex's x, y, z coordinates
                x, y, z = row[i], row[i+1], row[i+2]
                f.write(f"v {x} {y} {z}\n")
                vertex_count += 1
        
        # Write faces (each face is made up of 3 vertices, 1-based index)
        for i in range(vertex_count // 3):
            # Each face is made by three consecutive vertices
            v1 = 3 * i + 1
            v2 = 3 * i + 2
            v3 = 3 * i + 3
            f.write(f"f {v1} {v2} {v3}\n")

def perform_inference(model,epoch, optimizer, criterion, dataloader, device, scaler, epochs=200,rank=0, data_iter_step=0,startepoc=0,dataloader1=None):

    set_random_seed(rank)
    world_size=4    
    max_count = 5000  
    if rank == 0:

        output_path_epoch = Path(argvname) / str(epoch)
        (output_path_epoch / "gt").mkdir(parents=True, exist_ok=True)
        (output_path_epoch / "pred").mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()
    glist=[]
    gilist=[]

    with torch.no_grad():

        global_count = rank

        for face_seq, sampled_points in tqdm(dataloader1, disable=(rank != 0)):

            face_seq = face_seq.reshape(face_seq.shape[0], -1).to(device)
            sampled_points = sampled_points.to(device).half()

            generated_seq = model.module.generate_sequence(
                initial_input=face_seq[:,:lengthprompt],
                pc=sampled_points,
                max_seq_len=1000*9,
                device=device,
                top_k=50,
                top_p=0.95,
                temperature=1
            )
            glist.append(generated_seq)
            gilist.append(face_seq[:,:].cpu().numpy())

            batch_size = generated_seq.shape[0]
            for i in range(batch_size):

                if global_count >= max_count:
                    break
                m= generated_seq[0]
                global_count += world_size

            if global_count >= max_count:
                break
            if global_count >= 10:
                break
    if dist.is_initialized():

        glist_np = np.concatenate([t for t in glist]) if glist else np.array([])


        gather_list = [np.empty_like(glist_np) for _ in range(world_size)]
        dist.all_gather_object(gather_list, glist_np)
        
        gilist_np = np.concatenate([t for t in gilist]) if gilist else np.array([])


        gatheri_list = [np.empty_like(gilist_np) for _ in range(world_size)]
        dist.all_gather_object(gatheri_list, gilist_np)


        if rank == 0:

            combined_glist = np.concatenate(gather_list, axis=0)
            combined_gilist = np.concatenate(gatheri_list, axis=0)
            

            np_save_path =output_path_epoch / "glist.npy"
            np_save_path.parent.mkdir(parents=True, exist_ok=True)  

            for i in range(combined_glist.shape[0]):

                    end_indices = np.where(combined_glist[i] == end_symbol)[0]
                    save_mesh_as_obj(combined_gilist[i][:lengthprompt], output_path_epoch / (str(i) + 'oripart.obj'))
                    save_mesh_as_obj(combined_gilist[i][:], output_path_epoch / (str(i) + 'ori.obj'))
                    
                    print(end_indices)
                    
                    if end_indices.size > 0: 
                        d = end_indices[0] 
                        save_mesh_as_obj(combined_glist[i][:d], output_path_epoch / (str(i) + '.obj'))

                    else:
                        save_mesh_as_obj(combined_glist[i][:], output_path_epoch / (str(i) + '.obj'))

            np.save(np_save_path, combined_glist)


    else:

        if rank == 0:
            combined_glist = np.concatenate([t for t in glist], axis=0) if glist else np.array([])
            np_save_path = output_path_epoch / "glist.npy"
            np_save_path.parent.mkdir(parents=True, exist_ok=True) 
            np.save(np_save_path, combined_glist)




def main(rank, world_size):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.seed=3407
    random.seed(args.seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12336'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')


    num_categories = dic+3  # 0-129
    if name=='shapenet':
        model =     iFlame(num_categories=num_categories, embed_dim=512, num_heads=16,  depth=2, length=allmaxlen)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")
    model.to(device)
 

    model=torch.compile(model)

    model = DDP(model, device_ids=[rank],find_unused_parameters=False,gradient_as_bucket_view=True)
    data_iter_step=0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_symbol)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    optimizer = get_optimizer (model,optimizer_name= optname, lr=args.lr)
    scaler = GradScaler()

    dataset = TrianglesDataset(
        # dataset_path=dataset_path,
        split='train',
        scale_augment=True,
        shift_augment=True,
        overfit=False,
        num_tokens=dic-1,
        # category_prefix='03211117',
         allmaxlen=allmaxlen
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,drop_last=True)#,collate_fn=dataset.collate_fn)#, num_workers=os.cpu_count()
    dataset1 = TrianglesDataset(
        # dataset_path=dataset_path,
        split='val',
        scale_augment=False,
        shift_augment=False,
        overfit=False,
        num_tokens=dic-1,
        #  category_prefix='03211117',
         allmaxlen=allmaxlen
    )
    
    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1)#,collate_fn=dataset1.collate_fn)#, num_workers=os.cpu_count()
    
    
    checkpoint_path=None
    startepoc=0
    if checkpoint_path :
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        startepoc = checkpoint['epoch'] 
        data_iter_step=checkpoint['data_iter_step']
        print(f"Process {rank}: Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs * len(dataloader)//10, num_training_steps=args.epochs * len(dataloader), lr_start=args.lr, lr_end=args.min_lr)


    train_model(model, scheduler,optimizer, criterion, dataloader, device, scaler, epochs=args.epochs,rank=rank,data_iter_step=data_iter_step,startepoc=startepoc,dataloader1=dataloader1)


    dist.destroy_process_group()

def train_model(model,scheduler,  optimizer, criterion, dataloader, device, scaler, epochs=200,rank=0, data_iter_step=0,startepoc=0,dataloader1=None):
    model.train()
    if rank == 0:
        wandb.init(
                project="meshplayground",  # Replace with your project name
                name= argvname+argvlr,
                config={
                    "learning_rate": 0.0001,
                    "epochs": epochs,
                    "batch_size": dataloader.batch_size,
                    "model": "AutoEncoder",
                    "dataset": "CustomOBJDataset",
                }
            )

    iter_idx=0


    accumulation_steps= 1
  
    for epoch in range(1 + startepoc, epochs + 1):
        total_loss = 0
        data_iter_step += 1

        dataloader.sampler.set_epoch(epoch)
        with tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", disable=(rank != 0)) as pbar:
         for  batch_idx, (face_seq, sampled_points) in enumerate( pbar):
            # Move data to device
            iter_idx += 1
            face_seq = face_seq.reshape(face_seq.shape[0], -1).to(device)
            # face_seq=face_seq[:,:90]
            sampled_points = sampled_points.to(device)
            # adjust_learning_rate(optimizer, data_iter_step / len(dataloader) + epoch, args)
            
            # Forward pass
            optimizer.zero_grad()
            with autocast():
            # if 1:
                # m = face_seq.shape[0]
                outputs = model(face_seq, sampled_points)
                
                # Prepare targets for loss calculation
                targets = face_seq[:, 9:].contiguous().view(-1)  # Target: remove the first element (start symbol)
                outputs = outputs[:, 8:-1, :].contiguous().view(-1, dic + 3)  # Output: remove the last element

                    # Calculate loss
                loss = criterion(outputs, targets.long())
                acc = accuracy_(outputs.detach(), targets, ignore_label=pad_symbol, device=device)
            loss_value = loss.item() 
            total_loss += loss_value
            if torch.isnan(loss):
                return 0

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            pbar.set_postfix({'train_loss':loss.item(),'train_acc':acc.item()})
                        

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0 and iter_idx % 10 == 0:
                    wandb.log({
                        "epoch": epoch,
                        "iteration": iter_idx,
                        "loss": loss_value
                    })

         if (batch_idx + 1) % accumulation_steps != 0 and accumulation_steps>1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            
     
        # Optionally, log average loss per epoch
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})

        if  rank == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
            torch.cuda.synchronize()
            wandb.log({"epoch": epoch + 1, "train_loss": total_loss / len(dataloader)})
        # Print loss and accuracy every 10 epochs or on the first epoch
        # if epoch % 2 == 0 or epoch == 1:
        #         perform_inference(model, startepoc,optimizer, criterion, dataloader, device, scaler, epochs=args.epochs,rank=rank,data_iter_step=data_iter_step,startepoc=startepoc,dataloader1=dataloader1)

        if epoch % 1 == 0 or epoch == 1 : #or epoch == 1
            # print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
            correct = 0.0
            total = 0.0

            # Calculate accuracy
            log_perplexity_sum = 0.0  # Initialize log perplexity sum
            log_perplexity_count = 0  # Count of non-padded tokens

            with torch.no_grad():
                for face_seq, sampled_points in tqdm(dataloader1, desc=f"Epoch {epoch}/{epochs}", disable=(rank != 0)):
                    face_seq, sampled_points = face_seq.reshape(face_seq.shape[0], -1).to(device), sampled_points.to(device)
                    # face_seq=face_seq[:,:90]
                    with autocast():
                    # if 1:
                        outputs = model(face_seq, sampled_points)
                    
                    # Predictions and targets
                    predictions = outputs[:, 8:-1, :].argmax(dim=-1)
                    targets_shifted = face_seq[:, 9:].contiguous()
                    
                    # Mask to exclude padding
                    mask = targets_shifted != pad_symbol
                    
                    # Calculate accuracy
                    correct += (predictions[mask] == targets_shifted[mask]).sum().item()
                    total += targets_shifted[mask].numel()
                    targets = face_seq[:, 9:].contiguous().view(-1)  # Target: remove the first element (start symbol)
                    outputs = outputs[:, 8:-1, :].contiguous().view(-1,dic+3)  # Output: remove the last element

                    # Calculate log perplexity
                    log_probs = F.cross_entropy(
                        outputs,  # Transpose for cross_entropy format
                        targets,
                        reduction='none'  # Keep individual losses
                    )
                    
                    # Apply mask and sum up log perplexity terms
                    log_perplexity_sum += log_probs[mask.view(-1)].float() .sum().item()
                    log_perplexity_count += mask.sum().item()
                
                # Calculate average log perplexity across all batches
                avg_log_perplexity = log_perplexity_sum / log_perplexity_count if log_perplexity_count > 0 else float('inf')

                # Summing across distributed devices
                correct_tensor = torch.tensor(correct).to(device)
                total_tensor = torch.tensor(total).to(device)
                log_perplexity_tensor = torch.tensor(log_perplexity_sum).to(device)
                perplexity_count_tensor = torch.tensor(log_perplexity_count).to(device)

                dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(log_perplexity_tensor, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(perplexity_count_tensor, dst=0, op=dist.ReduceOp.SUM)

                if rank == 0:
                    accuracy = correct_tensor.item() / total_tensor.item()
                    avg_log_perplexity = log_perplexity_tensor.item() / perplexity_count_tensor.item() if perplexity_count_tensor.item() > 0 else float('inf')
                    print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}, Avg Log Perplexity: {avg_log_perplexity:.4f}")
                # if rank ==0:

                #     accuracy = correct_tensor.item() / total_tensor.item()
                #     print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')
                    wandb.log({"epoch": epoch +1, "val_accuracy": accuracy,"Avg Log Perplexity": avg_log_perplexity})
                # if rank == 0 and epoch >=10 and (epoch%10==0 or epoch==args.epochs):    
                if rank == 0 and  epoch==args.epochs:    
                    nvidia_smi_output = get_nvidia_smi()
                    wandb.log({
                        
                            'nvidia_smi': wandb.Html(f"<pre>{nvidia_smi_output}</pre>")
                        }) 
                    checkpoint = {
                        'epoch': epoch,  #
                        'model_state_dict': model.module.state_dict(),  # 
                        'optimizer_state_dict': optimizer.state_dict(),  # 
                        'scaler_state_dict': scaler.state_dict(),  #
                        'loss': accuracy,  # 
                        'data_iter_step':data_iter_step
                    }
                    model_save_path = name+str(epoch)+argvname+argvlr+optname+"small.pth"
                    torch.save(checkpoint, model_save_path)
                    print(f"Checkpoint saved to {model_save_path}")
                    wandb.log({"epoch": epoch +1, "val_accuracy": accuracy})


if __name__ == "__main__":
    world_size = int(sys.argv[1])#torch.cuda.device_count()
    # torch.set_float32_matmul_precision('high')
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
