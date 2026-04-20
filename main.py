import numpy as np
import torch 
import os
import warnings
import time
from logger import create_logger
import datetime
from functools import reduce
import operator
import random

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from utils.util import (
    AverageMeter,
    ensure_dir,
    get_rank,
    load_checkpoint,
    loss_with_exist,
    loss_with_aux,
    print_model_params,
    save_checkpoint,
    setup_distributed,
    synchronize,
)
from utils.validation import validate, validate_plantseg

import config
from args import get_parser
from model.refsegformer import RefSegFormer
from dataset.ReferDataset import ReferDataset
from dataset.plantseg import PlantSegDataset
from dataset.transform import get_transform


warnings.filterwarnings("ignore")


def apply_locked_defaults(args):
    if args.dataset != 'plantseg':
        return args

    if args.cfg_file == "configs/swin_base_patch4_window12_480.yaml":
        args.cfg_file = "configs/swin_base_patch4_window12_plantseg_480.yaml"
    args.use_mask = True
    args.use_pixel_decoder = True
    args.use_exist = False
    if args.batch_size == 1:
        args.batch_size = 6
    return args


def build_dataset(args, split, eval_mode, logger):
    if args.dataset == 'plantseg':
        return PlantSegDataset(
            args,
            split=split,
            image_transforms=get_transform(args, resize_target=not eval_mode),
            max_tokens=args.num_max_tokens,
            eval_mode=eval_mode,
            logger=logger,
        )

    return ReferDataset(
        args,
        split=split,
        image_transforms=get_transform(args),
        max_tokens=args.num_max_tokens,
        eval_mode=eval_mode,
        logger=logger,
    )


def build_sampler(dataset, distributed, shuffle):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return RandomSampler(dataset)
    return SequentialSampler(dataset)


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, epoch, local_rank, args):
    local_rank = get_rank()
    num_steps=len(train_loader)
    model.train()
    optimizer.zero_grad()

    batch_time=AverageMeter()
    loss_meter=AverageMeter()

    start=time.time()
    end=time.time()

    for idx, batch in enumerate(train_loader):
        if args.dataset == 'plantseg':
            img, target, emb, att_mask = batch
            exist = None
        else:
            img, target, emb, att_mask, exist = batch
            exist = exist.cuda(local_rank, non_blocking=True)

        emb = emb.squeeze(1)
        att_mask = att_mask.squeeze(1)
        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        emb = emb.cuda(local_rank, non_blocking=True)
        att_mask = att_mask.cuda(local_rank, non_blocking=True)

        output, exist_pred = model(img, emb, att_mask)
        
        if (args.dataset == 'rrefcoco' or args.dataset == 'rrefcoco+' or args.dataset == 'rrefcocog') and (args.use_exist):
            loss = loss_with_exist(args, output, exist_pred, target, exist)
        else:
            loss = loss_with_aux(args, output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # measure time
        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time() - end)
        end=time.time()

        if idx % args.print_freq==0 and local_rank==0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # remaining time
            etas=batch_time.avg*(num_steps-idx)
            logger.info(
                f'Train:[{epoch}/{args.epoch}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.avg:.4f}\t'
                f'loss {loss_meter.avg:.4f}\t'
                f'mem {memory_used:.0f}MB')
            batch_time.reset()
            loss_meter.reset()
    epoch_time=time.time()-start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate_all(args, model):
    if args.dataset == 'plantseg':
        eval_split = args.type
        eval_dataset = build_dataset(args, eval_split, True, logger)
        eval_sampler = build_sampler(eval_dataset, args.distributed, shuffle=False)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            sampler=eval_sampler,
        )
        metrics = validate_plantseg(
            args,
            logger,
            eval_loader,
            model,
            local_rank,
            split=eval_split,
            save_masks=args.save_masks or eval_split == 'test',
        )
        logger.info(f"Successfully evaluated plantseg split {eval_split}")
        return metrics

    eval_splits = []
    if args.dataset in ['refcoco', 'refcoco+']:
        eval_splits = ['val', 'testA', 'testB']
    elif args.dataset == 'refcocog' and args.splitBy == 'umd':
        eval_splits = ['val', 'test']
    elif args.dataset == 'refcocog' and args.splitBy == 'google':
        eval_splits = ['val']
    elif args.dataset in ['rrefcoco', 'rrefcoco+', 'rrefcocog']:
        eval_splits = ['val']
    
    for eval_split in eval_splits:
        eval_dataset = build_dataset(args, eval_split, True, logger)
        eval_sampler = build_sampler(eval_dataset, args.distributed, shuffle=False)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size = 1,
            num_workers = 8,
            pin_memory = torch.cuda.is_available(),
            sampler = eval_sampler
        )
        validate(args, logger, eval_loader, model, local_rank, eval_mode=False)
        logger.info(f'Successfully evaluated {args.dataset}({args.splitBy}), split {eval_split}')
    logger.info('Evaluating ended')

if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()
    args = apply_locked_defaults(args)

    local_rank, args.distributed = setup_distributed()
    
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = torch.cuda.is_available()
    cudnn.deterministic = True

    # only print in rank 0
    cfg=config.get_config(args)
    logger_path = os.path.join(args.output, args.exp)
    if args.eval:
        logger_path += '/eval'
    if local_rank != 0:
        synchronize()
    ensure_dir(logger_path)
    if local_rank == 0:
        synchronize()
    logger = create_logger(output_dir=logger_path, dist_rank=local_rank, name=f"{cfg.MODEL.NAME}")
    logger.info(args)
    
    # build model
    model = RefSegFormer(cfg, args, logger)
    model = model.cuda(local_rank)
    if args.distributed and torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if local_rank != 0:
        synchronize()
    if local_rank == 0:
        print_model_params(model, logger, details=False)
        synchronize()

    # build dataset (train and eval)
    train_dataset = build_dataset(args, 'train', False, logger)
    train_sampler = build_sampler(train_dataset, args.distributed, shuffle=True)
    train_loader = DataLoader(train_dataset,
                            batch_size = args.batch_size,
                            num_workers = 8,
                            pin_memory = torch.cuda.is_available(),
                            sampler = train_sampler,)

    eval_split = args.type
    if args.dataset == 'plantseg' and not args.eval:
        eval_split = 'val'
    eval_dataset = build_dataset(args, eval_split, True, logger)
    eval_sampler = build_sampler(eval_dataset, args.distributed, shuffle=False)
    eval_loader = DataLoader(eval_dataset,
                            batch_size = 1,
                            num_workers = 8,
                            pin_memory = torch.cuda.is_available(),
                            sampler = eval_sampler,)

    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in model_without_ddp.image_encoder.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)
    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in model_without_ddp.segmentation.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                        [[p for p in model_without_ddp.text_encoder.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]

    # build optimizer and lr scheduler
    optimizer = AdamW(params=params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 - step / (len(train_loader) * args.epoch)) ** 0.9)

    if args.eval:
        load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger, args.ckpt_epoch, best=True)
        validate_all(args, model)
        exit(0)
    if args.resume:
        load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger, args.ckpt_epoch, best=True)
        
    # training
    logger.info("Start training")
    start_time = time.time()
    best_metric = -1
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, optimizer, scheduler, epoch, local_rank, args)

        if args.dataset == 'plantseg':
            metrics = validate_plantseg(args, logger, eval_loader, model, local_rank, split='val', save_masks=False)
            better_epoch = (best_metric < metrics["IoU"])
            metric = metrics["IoU"]
        else:
            metrics = validate(args, logger, eval_loader, model, local_rank, eval_mode=False)
            oIoU = metrics["oIoU"]
            rIoU = metrics["rIoU"]
            if args.dataset == 'rrefcoco' or args.dataset == 'rrefcoco+' or args.dataset == 'rrefcocog':
                better_epoch = (best_metric < rIoU)
                metric = rIoU
            else:
                better_epoch = (best_metric < oIoU)
                metric = oIoU

        if local_rank != 0:
            synchronize()
        if local_rank == 0:
            if better_epoch:
                logger.info('Better epoch {}'.format(epoch))
                best_epoch = epoch
                best_metric = metric
                save_checkpoint(epoch, model_without_ddp, optimizer, scheduler, logger, args, best=True)
            synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # evaluate after training
    load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger, args.ckpt_epoch, best=True)
    if args.dataset == 'plantseg':
        test_dataset = build_dataset(args, 'test', True, logger)
        test_sampler = build_sampler(test_dataset, args.distributed, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            sampler=test_sampler,
        )
        logger.info('Evaluating best checkpoint on test split')
        validate_plantseg(args, logger, test_loader, model, local_rank, split='test', save_masks=True)
    else:
        args.eval_mode = 'all'
        logger.info('Evaluating after training')
        validate_all(args, model)
        args.eval_mode = 'cat'
        logger.info("Evaluating using text concat prompt")
        validate_all(args, model)
        logger.info('Evaluating ended')
