import os
import shutil
import argparse
import pickle
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from antibody_bfn.datasets import get_dataset
from antibody_bfn.models import get_model
from antibody_bfn.utils.misc import *
from antibody_bfn.utils.data import *
from antibody_bfn.utils.train import *


if __name__ == '__main__':
    # Fix for macOS shared memory issue
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--accum_steps', type=int, default=1, help='Number of gradient accumulation steps')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(f"Loss weights: {config.train.loss_weights}")
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    
    # Custom Sampler for CDR-type consistency
    if config.dataset.train.type == 'lmdb_preprocessed':
        db_dir = os.path.dirname(config.dataset.train.db_path)
        indices_path = os.path.join(db_dir, 'meta_indices.pkl')
        with open(indices_path, 'rb') as f:
            indices_by_type = pickle.load(f)
        batch_sampler = CDRBatchSampler(indices_by_type, config.train.batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=PaddingCollate(), 
            num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size, 
            collate_fn=PaddingCollate(), 
            shuffle=True,
            num_workers=args.num_workers
        )
        
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.device == 'cuda'))
    optimizer.zero_grad()
    it_first = 1
    min_val_loss = float('inf')

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        it_first = ckpt['iteration'] + 1
        min_val_loss = ckpt.get('min_val_loss', float('inf'))
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        # Force update learning rate from config
        new_lr = config.train.optimizer.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f'Overwriting learning rate to {new_lr} from config')
        
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])

    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()
        
        accum_loss_dict = {}
        valid_batch_count = 0

        # Gradient Accumulation
        for i in range(args.accum_steps):
            # Prepare data
            try:
                batch = recursive_to(next(train_iterator), args.device)
            except StopIteration:
                break

            if 'fixed_t' in config.train:
                batch['fixed_t'] = config.train.fixed_t

            # Forward
            # if args.debug: torch.set_anomaly_enabled(True)
            device_type = 'cuda' if args.device == 'cuda' else 'cpu'
            if args.device == 'mps': device_type = 'mps'
            
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(args.device != 'cpu')):
                loss_dict = model(batch)
                avg_t = loss_dict.pop('avg_t', 0.0)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                loss_dict['avg_t'] = avg_t # Put it back for logging
            
            # NaN Check and Skip
            if not torch.isfinite(loss):
                logger.warning(f'NaN or Inf detected in loss at iter {it}, micro-batch {i}. Skipping micro-batch.')
                continue

            # Normalize loss by accumulation steps
            loss = loss / args.accum_steps

            # Backward
            scaler.scale(loss).backward()
            
            # Accumulate logging stats
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + val
            valid_batch_count += 1

        time_forward_end = current_milli_time()

        if valid_batch_count == 0:
            logger.warning(f'All micro-batches failed at iter {it}. Skipping step.')
            optimizer.zero_grad()
            return

        # Average logging stats
        for k in accum_loss_dict:
            accum_loss_dict[k] /= valid_batch_count

        # Logging
        # Extract average t for logging
        avg_t = accum_loss_dict.pop('avg_t', 0.0)
        
        # Convert losses to tensors for log_losses compatibility
        for k in accum_loss_dict:
            accum_loss_dict[k] = torch.tensor(accum_loss_dict[k])

        # Optimizer Step
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        
        # Linear Warmup
        warmup_steps = config.train.get('warmup_steps', 1000)
        if it <= warmup_steps:
            warmup_factor = float(it) / float(warmup_steps)
            for param_group in optimizer.param_groups:
                # Assuming the initial LR in config is the target LR
                # We need to store the base LR somewhere to be cleaner, 
                # but for now we assume config.train.optimizer.lr is the target.
                param_group['lr'] = config.train.optimizer.lr * warmup_factor

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        log_freq = config.train.get('log_freq', 1)
        if it % log_freq == 0:
            log_losses(accum_loss_dict, it, 'train', logger, writer, others={
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                't': avg_t,
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                # Forward
                device_type = 'cuda' if args.device == 'cuda' else 'cpu'
                if args.device == 'mps': device_type = 'mps'
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(args.device != 'cpu')):
                    loss_dict = model(batch)
                    avg_t = loss_dict.pop('avg_t', 0.0)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    # We don't necessarily need to log t in val, but let's keep it consistent
                    loss_dict['avg_t'] = avg_t

                loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_state = {
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                        'min_val_loss': min_val_loss,
                    }
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save(ckpt_state, ckpt_path)
                    
                    if avg_val_loss < min_val_loss:
                        min_val_loss = avg_val_loss
                        ckpt_state['min_val_loss'] = min_val_loss
                        best_path = os.path.join(ckpt_dir, 'best.pt')
                        torch.save(ckpt_state, best_path)
                        logger.info(f'New best model saved to {best_path} with loss {min_val_loss:.4f}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
    finally:
        if not args.debug:
            writer.close()
        # Kill dataloader workers
        if 'train_iterator' in locals():
            del train_iterator
        logger.info('Done.')
