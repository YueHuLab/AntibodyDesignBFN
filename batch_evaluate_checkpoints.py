#!/usr/bin/env python
"""
Batch Evaluation of Checkpoints on Test Set.

Loops through checkpoints from start_ckpt to end_ckpt and evaluates each.
Only prints checkpoint name and summary table.
Saves each result to a separate folder.

Usage:
    python batch_evaluate_checkpoints.py \
        --config configs/test/bfn_testset.yml \
        --test_set data/2025_testset_43.csv \
        --chothia_dir data/2025_pdbs \
        --device mps \
        --start_ckpt 23200 \
        --end_ckpt 25000 \
        --step 100
"""
import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import tempfile
import copy
import warnings

# Suppress tqdm and other verbose output
warnings.filterwarnings('ignore')

from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
from diffab.utils.train import recursive_to
from diffab.utils.misc import load_config, seed_all
from diffab.utils.data import PaddingCollate, DEFAULT_PAD_VALUES
from diffab.utils.transforms import get_transform, Compose
from diffab.utils.transforms.mask import MaskSingleCDR
from diffab.utils.transforms.merge import MergeChains
from diffab.utils.transforms.patch import PatchAroundAnchor
from diffab.tools.renumber import renumber as renumber_antibody

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
CDR_TYPES = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']


def evaluate_single_cdr(model, batch, device):
    """Evaluate a single CDR and return metrics."""
    model.eval()
    
    with torch.no_grad():
        gen_mask = batch['generate_flag'][0].bool()
        
        if gen_mask.sum() == 0:
            return None
        
        if 'native_aa' in batch:
            native_aa = batch['native_aa'][0][gen_mask]
        else:
            native_aa = batch['aa'][0][gen_mask]
            
        native_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in native_aa.cpu().tolist()])
        
        sample_opt = {'deterministic': True}
        traj = model.sample(batch, sample_opt=sample_opt)
        
        aa_new = traj[0][2][0]
        pred_aa = aa_new[gen_mask]
        pred_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in pred_aa.cpu().tolist()])
        
        matches = sum(1 for p, t in zip(pred_seq, native_seq) if p == t)
        aar = matches / len(native_seq) if len(native_seq) > 0 else 0.0
        
        if 'pred_logits' in traj:
            pred_logits = traj['pred_logits'][0][gen_mask]
            log_probs = torch.log_softmax(pred_logits[..., :20], dim=-1)
            nll = -log_probs[range(len(pred_aa)), pred_aa].mean()
            perplexity = torch.exp(nll).item()
        else:
            perplexity = float('nan')
        
        return {
            'native_seq': native_seq,
            'pred_seq': pred_seq,
            'length': len(native_seq),
            'aar': aar,
            'perplexity': perplexity,
        }


def evaluate_structure_all_cdrs(pdb_id, pdb_path, heavy_chain, light_chain, model, device, antigen_id=None, gt_pdb_path=None):
    """Evaluate all CDRs for a single structure."""
    results = []
    
    temp_files = []
    try:
        out_pdb_path = tempfile.mktemp(suffix='.pdb')
        temp_files.append(out_pdb_path)
        heavy_chains, light_chains = renumber_antibody(pdb_path, out_pdb_path, verbose=False)
        
        heavy_id = heavy_chains[0] if heavy_chains else heavy_chain
        light_id = light_chains[0] if light_chains else light_chain
        
        structure = preprocess_antibody_structure({
            'id': pdb_id,
            'pdb_path': out_pdb_path,
            'heavy_id': heavy_id,
            'light_id': light_id,
            'antigen_id': antigen_id,
        })
        
        if structure is None:
            return results

        gt_structure = None
        if gt_pdb_path:
            try:
                out_gt_path = tempfile.mktemp(suffix='.pdb')
                temp_files.append(out_gt_path)
                gt_h, gt_l = renumber_antibody(gt_pdb_path, out_gt_path, verbose=False)
                gt_structure = preprocess_antibody_structure({
                    'id': f"{pdb_id}_gt",
                    'pdb_path': out_gt_path,
                    'heavy_id': heavy_id,
                    'light_id': light_id,
                    'antigen_id': antigen_id,
                })
            except Exception as e:
                gt_structure = None

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        temp_files = [] 
        
        pad_values = DEFAULT_PAD_VALUES.copy()
        pad_values['native_aa'] = 21
        collate_fn = PaddingCollate(pad_values=pad_values)
        
        for cdr_name in CDR_TYPES:
            try:
                transform = Compose([
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                    PatchAroundAnchor(),
                ])
                
                data = transform(copy.deepcopy(structure))
                
                if 'generate_flag' not in data or data['generate_flag'].sum() == 0:
                    continue
                
                if gt_structure:
                    gt_data = transform(copy.deepcopy(gt_structure))
                    if gt_data['aa'].size(0) == data['aa'].size(0):
                        data['native_aa'] = gt_data['aa']

                batch = collate_fn([data])
                batch = recursive_to(batch, device)
                
                result = evaluate_single_cdr(model, batch, device)
                
                if result is not None:
                    result['pdb_id'] = pdb_id
                    result['cdr'] = cdr_name
                    results.append(result)
                    
            except Exception as e:
                continue
                
    except Exception as e:
        pass
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    
    return results


def evaluate_checkpoint(ckpt_path, config, test_df, chothia_dir, device, output_dir, gt_dir=None, detect_antigen=False):
    """Evaluate a single checkpoint and return summary."""
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model_config = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        model_config['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    
    model = get_model(model_config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Evaluate
    all_results = []
    
    for idx, row in test_df.iterrows():
        pdb_id = row['pdb'].lower()
        pdb_path = os.path.join(chothia_dir, f"{pdb_id}_fv.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(chothia_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(pdb_path):
            continue
        
        gt_pdb_path = None
        if gt_dir:
            gt_pdb_path = os.path.join(gt_dir, f"{pdb_id}_fv.pdb")
            if not os.path.exists(gt_pdb_path):
                gt_pdb_path = os.path.join(gt_dir, f"{pdb_id}.pdb")
            if not os.path.exists(gt_pdb_path):
                gt_pdb_path = None
        
        heavy_chain = row['Hchain']
        light_chain = row['Lchain']
        
        antigen_id = None
        if detect_antigen:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            try:
                struct = parser.get_structure('tmp', pdb_path)
                chains = [c.id for c in struct[0]]
                ag_chains = [c for c in chains if c not in [heavy_chain, light_chain]]
                antigen_id = ag_chains[0] if ag_chains else None
            except:
                antigen_id = None

        results = evaluate_structure_all_cdrs(
            pdb_id, pdb_path, heavy_chain, light_chain,
            model, device, antigen_id=antigen_id,
            gt_pdb_path=gt_pdb_path
        )
        all_results.extend(results)
    
    # Clean up model to free memory
    del model
    del ckpt
    torch.mps.empty_cache() if device == 'mps' else None
    
    if len(all_results) == 0:
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Compute summary
    summary_data = []
    for cdr_name in CDR_TYPES:
        cdr_results = results_df[results_df['cdr'] == cdr_name]
        if len(cdr_results) == 0:
            continue
        
        count = len(cdr_results)
        avg_length = cdr_results['length'].mean()
        avg_aar = cdr_results['aar'].mean() * 100
        avg_ppl = cdr_results['perplexity'].mean()
        
        summary_data.append({
            'cdr': cdr_name,
            'count': count,
            'avg_length': avg_length,
            'aar_percent': avg_aar,
            'perplexity': avg_ppl,
        })
    
    # Overall
    total_count = len(results_df)
    overall_aar = results_df['aar'].mean() * 100
    overall_ppl = results_df['perplexity'].mean()
    
    summary_data.append({
        'cdr': 'ALL',
        'count': total_count,
        'avg_length': results_df['length'].mean(),
        'aar_percent': overall_aar,
        'perplexity': overall_ppl,
    })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': ckpt_path,
            'num_structures': len(test_df),
            'num_evaluations': len(results_df),
            'summary': summary_data,
        }, f, indent=2)
    
    return summary_data


def print_summary(ckpt_name, summary_data, output_dir):
    """Print formatted summary."""
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {ckpt_name}")
    print(f"{'='*70}")
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'CDR':<10} {'Count':<8} {'Avg Length':<12} {'AAR (%)':<12} {'PPL':<12}")
    print("-"*70)
    
    for item in summary_data:
        if item['cdr'] == 'ALL':
            continue
        print(f"{item['cdr']:<10} {item['count']:<8} {item['avg_length']:<12.1f} {item['aar_percent']:<12.1f} {item['perplexity']:<12.2f}")
    
    print("-"*70)
    all_item = [s for s in summary_data if s['cdr'] == 'ALL'][0]
    print(f"{'ALL':<10} {all_item['count']:<8} {all_item['avg_length']:<12.1f} {all_item['aar_percent']:<12.1f} {all_item['perplexity']:<12.2f}")
    print("="*70)
    print(f"[INFO] Saved summary to: {output_dir}/summary.csv")
    print(f"[INFO] Saved JSON results to: {output_dir}/results.json")


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate checkpoints on test set')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--test_set', type=str, required=True, help='Test set CSV file')
    parser.add_argument('--chothia_dir', type=str, required=True, help='Chothia PDB directory')
    parser.add_argument('--device', type=str, default='mps', help='Device')
    parser.add_argument('--start_ckpt', type=int, default=23200, help='Start checkpoint number')
    parser.add_argument('--end_ckpt', type=int, default=25000, help='End checkpoint number')
    parser.add_argument('--step', type=int, default=100, help='Checkpoint step')
    parser.add_argument('--ckpt_dir', type=str, default=None, 
                        help='Directory containing checkpoints (overrides config)')
    parser.add_argument('--output_base_dir', type=str, default='./results/batch_evaluation',
                        help='Base output directory for results')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                        help='Directory containing ground truth PDBs')
    parser.add_argument('--detect_antigen', action='store_true', 
                        help='Automatically detect and use antigen chains')
    args = parser.parse_args()
    
    # Load config to get checkpoint directory
    config, config_name = load_config(args.config)
    seed_all(config.sampling.seed if hasattr(config.sampling, 'seed') else 42)
    
    # Extract checkpoint directory
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_path_template = config.model.checkpoint
        ckpt_dir = os.path.dirname(ckpt_path_template)
    
    # Load test set
    print(f"[INFO] Loading test set from CSV: {args.test_set}")
    test_df = pd.read_csv(args.test_set)
    print(f"[INFO] Test set size: {len(test_df)}")
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Generate checkpoint list
    ckpt_numbers = list(range(args.start_ckpt, args.end_ckpt + 1, args.step))
    
    print(f"\n[INFO] Will evaluate {len(ckpt_numbers)} checkpoints: {args.start_ckpt} to {args.end_ckpt} (step={args.step})")
    print(f"[INFO] Checkpoint directory: {ckpt_dir}")
    print(f"[INFO] Results will be saved to: {args.output_base_dir}")
    print("="*70)
    
    # Evaluate each checkpoint
    for ckpt_num in ckpt_numbers:
        ckpt_name = f"{ckpt_num}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"\n[WARN] Checkpoint not found: {ckpt_path}, skipping...")
            continue
        
        # Create output directory for this checkpoint
        output_dir = os.path.join(args.output_base_dir, f"ckpt_{ckpt_num}")
        
        print(f"\n[INFO] Evaluating checkpoint: {ckpt_name}...")
        
        try:
            summary = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                config=config,
                test_df=test_df,
                chothia_dir=args.chothia_dir,
                device=args.device,
                output_dir=output_dir,
                gt_dir=args.ground_truth_dir,
                detect_antigen=args.detect_antigen
            )
            
            if summary:
                print_summary(ckpt_name, summary, output_dir)
            else:
                print(f"[ERROR] No results for checkpoint: {ckpt_name}")
                
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {ckpt_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("[INFO] Batch evaluation complete!")
    print(f"[INFO] Results saved to: {args.output_base_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
