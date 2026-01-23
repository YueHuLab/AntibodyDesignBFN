#!/usr/bin/env python
"""
Systematic Evaluation of Antibody CDR Sequence Design Model.

Based on design_seq.py, evaluates the trained model on a test set.
Computes:
- Amino Acid Recovery (AAR) per CDR
- Perplexity (PPL) per CDR

Usage:
    python evaluate_testset.py --config configs/test/bfn_testset.yml --device mps
"""
import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import tempfile

from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
from diffab.utils.train import recursive_to
from diffab.utils.misc import load_config, seed_all
from diffab.utils.data import PaddingCollate
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
        # Get generation mask
        gen_mask = batch['generate_flag'][0].bool()
        
        if gen_mask.sum() == 0:
            return None
        
        # Get native sequence (prefer 'native_aa' if available, else 'aa')
        if 'native_aa' in batch:
            native_aa = batch['native_aa'][0][gen_mask]
        else:
            native_aa = batch['aa'][0][gen_mask]
            
        native_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in native_aa.cpu().tolist()])
        
        # Sample (deterministic)
        sample_opt = {'deterministic': True}
        traj = model.sample(batch, sample_opt=sample_opt)
        
        # Get predicted sequence
        aa_new = traj[0][2][0]
        pred_aa = aa_new[gen_mask]
        pred_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in pred_aa.cpu().tolist()])
        
        # Calculate AAR
        matches = sum(1 for p, t in zip(pred_seq, native_seq) if p == t)
        aar = matches / len(native_seq) if len(native_seq) > 0 else 0.0
        
        # Calculate perplexity from logits
        if 'pred_logits' in traj:
            pred_logits = traj['pred_logits'][0][gen_mask]  # (L_gen, num_classes)
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
    """Evaluate all CDRs for a single structure, following design_seq.py pattern."""
    results = []
    
    temp_files = []
    try:
        # Renumber target (cleaned) structure
        out_pdb_path = tempfile.mktemp(suffix='.pdb')
        temp_files.append(out_pdb_path)
        heavy_chains, light_chains = renumber_antibody(pdb_path, out_pdb_path)
        
        heavy_id = heavy_chains[0] if heavy_chains else heavy_chain
        light_id = light_chains[0] if light_chains else light_chain
        
        # Load target structure
        structure = preprocess_antibody_structure({
            'id': pdb_id,
            'pdb_path': out_pdb_path,
            'heavy_id': heavy_id,
            'light_id': light_id,
            'antigen_id': antigen_id,
        })
        
        if structure is None:
            print(f"[WARN] Failed to parse structure: {pdb_id}")
            return results

        # Process Ground Truth if provided
        gt_structure = None
        if gt_pdb_path:
            try:
                out_gt_path = tempfile.mktemp(suffix='.pdb')
                temp_files.append(out_gt_path)
                gt_h, gt_l = renumber_antibody(gt_pdb_path, out_gt_path)
                # Use same IDs as target to assume consistency
                gt_structure = preprocess_antibody_structure({
                    'id': f"{pdb_id}_gt",
                    'pdb_path': out_gt_path,
                    'heavy_id': heavy_id,
                    'light_id': light_id,
                    'antigen_id': antigen_id,
                })
            except Exception as e:
                print(f"[WARN] Failed to process GT for {pdb_id}: {e}")
                gt_structure = None

        # Clean up input temp files immediately
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        temp_files = [] 
        
        # Use custom PaddingCollate to handle native_aa
        from diffab.utils.data import DEFAULT_PAD_VALUES
        pad_values = DEFAULT_PAD_VALUES.copy()
        pad_values['native_aa'] = 21
        collate_fn = PaddingCollate(pad_values=pad_values)
        
        for cdr_name in CDR_TYPES:
            try:
                # Create transform
                transform = Compose([
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                    PatchAroundAnchor(),
                ])
                
                # Apply transform to target
                import copy
                data = transform(copy.deepcopy(structure))
                
                # Check target valid
                if 'generate_flag' not in data or data['generate_flag'].sum() == 0:
                    continue
                
                # If GT available, transform it and extract sequence
                if gt_structure:
                    gt_data = transform(copy.deepcopy(gt_structure))
                    # verify dimensions match
                    if gt_data['aa'].size(0) == data['aa'].size(0):
                        data['native_aa'] = gt_data['aa']
                    else:
                        print(f"[WARN] GT size mismatch for {pdb_id} {cdr_name}")

                # Collate
                batch = collate_fn([data])
                batch = recursive_to(batch, device)
                
                # Evaluate
                result = evaluate_single_cdr(model, batch, device)
                
                if result is not None:
                    result['pdb_id'] = pdb_id
                    result['cdr'] = cdr_name
                    results.append(result)
                    
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"[WARN] Error processing structure {pdb_id}: {e}")
    finally:
         for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate antibody design model on test set')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--test_set', type=str, default='sabdab',
                        help='Test set CSV file or "sabdab" to use dataset definition')
    parser.add_argument('--sabdab_summary', type=str, default='./data/sabdab_summary_all.tsv',
                        help='SAbDab summary file path')
    parser.add_argument('--chothia_dir', type=str, default='./data/all_structures/chothia',
                        help='Chothia PDB directory')
    parser.add_argument('--processed_dir', type=str, default='./data/processed',
                        help='Processed data directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Split to use (train/val/test)')
    parser.add_argument('--output_dir', type=str, default='./results/testset_evaluation',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of structures to evaluate (for quick testing)')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                        help='Directory containing ground truth PDBs (for AAR calculation if input is masked/cleaned)')
    parser.add_argument('--detect_antigen', action='store_true', 
                        help='Automatically detect and use antigen chains (any chain != H/L)')
    args = parser.parse_args()
    
    # Load config
    config, config_name = load_config(args.config)
    seed_all(config.sampling.seed if hasattr(config.sampling, 'seed') else 42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test set
    if args.test_set.endswith('.csv'):
        print(f"[INFO] Loading test set from CSV: {args.test_set}")
        test_df = pd.read_csv(args.test_set)
        print(f"[INFO] Test set size: {len(test_df)}")
    else:
        # ... existing sabdab loading code ...
        pass
        
    # (Leaving Sabdab loading code as is)
    if args.test_set == 'sabdab':
        print(f"[INFO] Loading test set from SAbDab definition (split={args.split})")
        from diffab.datasets.sabdab import SAbDabDataset
        dataset = SAbDabDataset(
            summary_path=args.sabdab_summary,
            chothia_dir=args.chothia_dir,
            processed_dir=args.processed_dir,
            split=args.split,
            reset=False
        )
        
        # Build dataframe from dataset split
        records = []
        id_to_entry = {e['id']: e for e in dataset.sabdab_entries}
        
        for pid in dataset.ids_in_split:
            if pid not in id_to_entry:
                continue
            entry = id_to_entry[pid]
            records.append({
                'pdb': entry['pdbcode'],
                'Hchain': entry['H_chain'],
                'Lchain': entry['L_chain']
            })
        test_df = pd.DataFrame(records)
        print(f"[INFO] Test set size (from SAbDab): {len(test_df)}")

    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples)
        print(f"[INFO] Limited to {len(test_df)} samples for quick testing")
    
    # Load model (following design_seq.py pattern)
    print(f"[INFO] Loading model: {config.model.checkpoint}")
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    
    model_config = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        model_config['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    
    model = get_model(model_config).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("[INFO] Model loaded successfully")
    
    # Evaluate each structure
    all_results = []
    chothia_dir = args.chothia_dir
    gt_dir = args.ground_truth_dir
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Evaluating'):
        pdb_id = row['pdb'].lower()
        # Try with _fv suffix first (common in this dataset)
        pdb_path = os.path.join(chothia_dir, f"{pdb_id}_fv.pdb")
        if not os.path.exists(pdb_path):
             # Fallback to standard
             pdb_path = os.path.join(chothia_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"[WARN] PDB file not found: {pdb_path}")
            continue
        
        # Identify GT path if needed
        gt_pdb_path = None
        if gt_dir:
            gt_pdb_path = os.path.join(gt_dir, f"{pdb_id}_fv.pdb") # Prefer _fv
            if not os.path.exists(gt_pdb_path):
                gt_pdb_path = os.path.join(gt_dir, f"{pdb_id}.pdb")
            if not os.path.exists(gt_pdb_path):
                 print(f"[WARN] GT PDB not found for {pdb_id}")
                 gt_pdb_path = None
        
        heavy_chain = row['Hchain']
        light_chain = row['Lchain']
        
        if args.detect_antigen:
            # Simple heuristic: any chain that is not H or L is antigen
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            try:
                struct = parser.get_structure('tmp', pdb_path)
                chains = [c.id for c in struct[0]]
                ag_chains = [c for c in chains if c not in [heavy_chain, light_chain]]
                antigen_id = ag_chains[0] if ag_chains else None
            except:
                antigen_id = None
        else:
            antigen_id = None

        results = evaluate_structure_all_cdrs(
            pdb_id, pdb_path, heavy_chain, light_chain,
            model, args.device, antigen_id=antigen_id,
            gt_pdb_path=gt_pdb_path
        )
        all_results.extend(results)
    
    # Convert to DataFrame
    if len(all_results) == 0:
        print("[ERROR] No results collected!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Saved raw results to: {results_path}")
    
    # Compute summary statistics per CDR
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'CDR':<10} {'Count':<8} {'Avg Length':<12} {'AAR (%)':<12} {'PPL':<12}")
    print("-"*70)
    
    summary_data = []
    for cdr_name in CDR_TYPES:
        cdr_results = results_df[results_df['cdr'] == cdr_name]
        if len(cdr_results) == 0:
            continue
        
        count = len(cdr_results)
        avg_length = cdr_results['length'].mean()
        avg_aar = cdr_results['aar'].mean() * 100
        avg_ppl = cdr_results['perplexity'].mean()
        
        print(f"{cdr_name:<10} {count:<8} {avg_length:<12.1f} {avg_aar:<12.1f} {avg_ppl:<12.2f}")
        
        summary_data.append({
            'cdr': cdr_name,
            'count': count,
            'avg_length': avg_length,
            'aar_percent': avg_aar,
            'perplexity': avg_ppl,
        })
    
    # Overall statistics
    print("-"*70)
    total_count = len(results_df)
    overall_aar = results_df['aar'].mean() * 100
    overall_ppl = results_df['perplexity'].mean()
    print(f"{'ALL':<10} {total_count:<8} {results_df['length'].mean():<12.1f} {overall_aar:<12.1f} {overall_ppl:<12.2f}")
    
    summary_data.append({
        'cdr': 'ALL',
        'count': total_count,
        'avg_length': results_df['length'].mean(),
        'aar_percent': overall_aar,
        'perplexity': overall_ppl,
    })
    
    print("="*70)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary to: {summary_path}")
    
    # Save as JSON too
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': args.config,
            'test_set': args.test_set,
            'num_structures': len(test_df),
            'num_evaluations': len(results_df),
            'summary': summary_data,
            'results': all_results,
        }, f, indent=2)
    print(f"[INFO] Saved JSON results to: {json_path}")


if __name__ == '__main__':
    main()
