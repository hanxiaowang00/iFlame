import glob
import os
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

import wandb
import re
import json
from typing import List, Optional, Tuple
import subprocess
import time
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import math
from transformer import *
from pathlib import Path
# from flash_attn.modules.mlp import  GatedMlp
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from datasetmy import TrianglesDataset
import sys
block_size=8
offset_size=16
dic=129
fnnn=2816

pad_symbol=dic
start_symbol = dic+1 # 
end_symbol = dic+2    # 
lengthprompt=9*1
lengthoftree=800*9
# lengthoftree=100
maaa=-100
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
exp_name='shapenet'

name='shapenet'
class Args:
    def __init__(self):
        self.weight_decay = 0.05  # Default weight decay
        self.lr = 5e-3          # Absolute learning rate (to be computed)
        self.blr = 1e-4           # Base learning rate
        self.layer_decay = 0.75   # Layer-wise learning rate decay
        self.min_lr = 1.28e-6        # Minimum learning rate
        self.warmup_epochs =10    # Number of warmup epochs
        self.epochs = 20       # Total number of training epochs
        self.batch_size = 64#40#57   # Batch size
        self.world_size = 4       # Number of distributed processes (adjust if using DDP)

args = Args()




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


def main(rank, world_size):

    # os.makedirs("/home/wangh0t/treegen/data10", exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.seed=3407
    random.seed(args.seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12333'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')


    num_categories = dic+3  # 0-129
    if exp_name=='shapenet':
        model = iFlame(num_categories=num_categories, embed_dim=512, num_heads=16,  depth=2, length=lengthoftree)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")
    model.to(device)

    model = DDP(model, device_ids=[rank],find_unused_parameters=False)
    data_iter_step=0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_symbol)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()


    dataset = TrianglesDataset(
        # dataset_path=dataset_path,
        split='train',
        scale_augment=True,
        shift_augment=True,
        overfit=False,
        num_tokens=dic-1,
        #   category_prefix='02828884'
        # category_prefix='02828884'#'03001627'
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,drop_last=True)#, num_workers=os.cpu_count()
    dataset1 = TrianglesDataset(

        split='val',
        scale_augment=False,
        shift_augment=False,
        overfit=False,
        num_tokens=dic-1,


    )
    
    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1)#, num_workers=os.cpu_count()
    
    checkpoint_path=sys.argv[1]#'/home/wangh0t/MeshGPT/padded_datashapenet.npy20248s.pth'#/home/wangh0t/treegenlast/autoencoder_checkpoint_epoch.pth'
    startepoc=0
    if  checkpoint_path :
            # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            state_dict = torch.load(checkpoint_path)
            checkpoint = {}
            for k, v in state_dict['model_state_dict'].items():
                new_key = k.replace('_orig_mod.', '')  # 去掉前缀
                checkpoint[new_key] = v
            # model.load_state_dict(new_state_dict)
            model.module.load_state_dict(checkpoint, strict=True)
            # optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            scaler.load_state_dict(state_dict['scaler_state_dict'])
            startepoc = state_dict['epoch'] 
            data_iter_step=state_dict['data_iter_step']
            print(f"Process {rank}: Loaded checkpoint '{checkpoint_path}' (epoch {state_dict['epoch']})")


    train_model(model, optimizer, criterion, dataloader, device, scaler, epochs=args.epochs,rank=rank,data_iter_step=data_iter_step,startepoc=startepoc,dataloader1=dataloader1)

    dist.destroy_process_group()

def set_random_seed(rank, base_seed=3407):
   
    seed = base_seed + rank 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def perform_inference(model,epoch, optimizer, criterion, dataloader, device, scaler, epochs=200,rank=0, data_iter_step=0,startepoc=0,dataloader1=None):

    set_random_seed(rank)
    world_size=4    
    max_count = 5000 
    if rank == 0:
        # rnn/<epoch>/gt 和 rnn/<epoch>/pred
        output_path_epoch = Path(exp_name) / str(epoch)
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
                max_seq_len=lengthoftree,
                device=device,
                top_k=50,
                top_p=0.95,
                temperature=1.0
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
  

def train_model(model, optimizer, criterion, dataloader, device, scaler, epochs=200,rank=0, data_iter_step=0,startepoc=0,dataloader1=None):

    iter_idx=0
    if 1 : #or epoch == 1
            # print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
            correct = 0.0
            total = 0.0

            # Calculate accuracy
            log_perplexity_sum = 0.0  # Initialize log perplexity sum
            log_perplexity_count = 0  # Count of non-padded tokens

            with torch.no_grad():
                for face_seq, sampled_points in dataloader1:
                    face_seq, sampled_points = face_seq.reshape(face_seq.shape[0], -1).to(device), sampled_points.to(device)
                    
                    with autocast():
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
                    print(f"Epoch0 - Accuracy: {accuracy:.4f}, Avg Log Perplexity: {avg_log_perplexity:.4f}")

        
        
    perform_inference(model, startepoc,optimizer, criterion, dataloader, device, scaler, epochs=args.epochs,rank=rank,data_iter_step=data_iter_step,startepoc=startepoc,dataloader1=dataloader1)

if __name__ == "__main__":
    world_size = 1#torch.cuda.device_count()
    # torch.set_float32_matmul_precision('high')
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
