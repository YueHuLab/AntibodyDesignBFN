#!/usr/bin/env python
"""
Sequence Design for Antibody CDR using BFN.

Usage Examples:
  # Standard inference (deterministic, 1 sample)
  python design_seq.py structure.pdb --heavy A --light B --config config.yml --device mps
  
  # Multiple random samples  
  python design_seq.py structure.pdb --heavy A --light B --config config.yml --num_samples 10 --stochastic
  
  # Evaluation mode (compare to native sequence)
  python design_seq.py structure.pdb --heavy A --light B --config config.yml --eval
"""
import os
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm

from antibody_bfn.datasets.custom import preprocess_antibody_structure
from antibody_bfn.models import get_model
from antibody_bfn.utils.train import recursive_to
from antibody_bfn.utils.misc import load_config, seed_all
from antibody_bfn.utils.data import PaddingCollate
from antibody_bfn.utils.transforms import get_transform
from antibody_bfn.tools.renumber import renumber as renumber_antibody

# BLOSUM62 matrix (complete 20x20) - for evaluation mode
BLOSUM62_MATRIX = {
    'A': {'A': 4, 'C': 0, 'D':-2, 'E':-1, 'F':-2, 'G': 0, 'H':-2, 'I':-1, 'K':-1, 'L':-1, 'M':-1, 'N':-2, 'P':-1, 'Q':-1, 'R':-1, 'S': 1, 'T': 0, 'V': 0, 'W':-3, 'Y':-2},
    'C': {'A': 0, 'C': 9, 'D':-3, 'E':-4, 'F':-2, 'G':-3, 'H':-3, 'I':-1, 'K':-3, 'L':-1, 'M':-1, 'N':-3, 'P':-3, 'Q':-3, 'R':-3, 'S':-1, 'T':-1, 'V':-1, 'W':-2, 'Y':-2},
    'D': {'A':-2, 'C':-3, 'D': 6, 'E': 2, 'F':-3, 'G':-1, 'H':-1, 'I':-3, 'K':-1, 'L':-4, 'M':-3, 'N': 1, 'P':-1, 'Q': 0, 'R':-2, 'S': 0, 'T':-1, 'V':-3, 'W':-4, 'Y':-3},
    'E': {'A':-1, 'C':-4, 'D': 2, 'E': 5, 'F':-3, 'G':-2, 'H': 0, 'I':-3, 'K': 1, 'L':-3, 'M':-2, 'N': 0, 'P':-1, 'Q': 2, 'R': 0, 'S': 0, 'T':-1, 'V':-2, 'W':-3, 'Y':-2},
    'F': {'A':-2, 'C':-2, 'D':-3, 'E':-3, 'F': 6, 'G':-3, 'H':-1, 'I': 0, 'K':-3, 'L': 0, 'M': 0, 'N':-3, 'P':-4, 'Q':-3, 'R':-3, 'S':-2, 'T':-2, 'V':-1, 'W': 1, 'Y': 3},
    'G': {'A': 0, 'C':-3, 'D':-1, 'E':-2, 'F':-3, 'G': 6, 'H':-2, 'I':-4, 'K':-2, 'L':-4, 'M':-3, 'N': 0, 'P':-2, 'Q':-2, 'R':-2, 'S': 0, 'T':-2, 'V':-3, 'W':-2, 'Y':-3},
    'H': {'A':-2, 'C':-3, 'D':-1, 'E': 0, 'F':-1, 'G':-2, 'H': 8, 'I':-3, 'K':-1, 'L':-3, 'M':-2, 'N': 1, 'P':-2, 'Q': 0, 'R': 0, 'S':-1, 'T':-2, 'V':-3, 'W':-2, 'Y': 2},
    'I': {'A':-1, 'C':-1, 'D':-3, 'E':-3, 'F': 0, 'G':-4, 'H':-3, 'I': 4, 'K':-3, 'L': 2, 'M': 1, 'N':-3, 'P':-3, 'Q':-3, 'R':-3, 'S':-2, 'T':-1, 'V': 3, 'W':-3, 'Y':-1},
    'K': {'A':-1, 'C':-3, 'D':-1, 'E': 1, 'F':-3, 'G':-2, 'H':-1, 'I':-3, 'K': 5, 'L':-2, 'M':-1, 'N': 0, 'P':-1, 'Q': 1, 'R': 2, 'S': 0, 'T':-1, 'V':-2, 'W':-3, 'Y':-2},
    'L': {'A':-1, 'C':-1, 'D':-4, 'E':-3, 'F': 0, 'G':-4, 'H':-3, 'I': 2, 'K':-2, 'L': 4, 'M': 2, 'N':-3, 'P':-3, 'Q':-2, 'R':-2, 'S':-2, 'T':-1, 'V': 1, 'W':-2, 'Y':-1},
    'M': {'A':-1, 'C':-1, 'D':-3, 'E':-2, 'F': 0, 'G':-3, 'H':-2, 'I': 1, 'K':-1, 'L': 2, 'M': 5, 'N':-2, 'P':-2, 'Q': 0, 'R':-1, 'S':-1, 'T':-1, 'V': 1, 'W':-1, 'Y':-1},
    'N': {'A':-2, 'C':-3, 'D': 1, 'E': 0, 'F':-3, 'G': 0, 'H': 1, 'I':-3, 'K': 0, 'L':-3, 'M':-2, 'N': 6, 'P':-2, 'Q': 0, 'R': 0, 'S': 1, 'T': 0, 'V':-3, 'W':-4, 'Y':-2},
    'P': {'A':-1, 'C':-3, 'D':-1, 'E':-1, 'F':-4, 'G':-2, 'H':-2, 'I':-3, 'K':-1, 'L':-3, 'M':-2, 'N':-2, 'P': 7, 'Q':-1, 'R':-2, 'S':-1, 'T':-1, 'V':-2, 'W':-4, 'Y':-3},
    'Q': {'A':-1, 'C':-3, 'D': 0, 'E': 2, 'F':-3, 'G':-2, 'H': 0, 'I':-3, 'K': 1, 'L':-2, 'M': 0, 'N': 0, 'P':-1, 'Q': 5, 'R': 1, 'S': 0, 'T':-1, 'V':-2, 'W':-2, 'Y':-1},
    'R': {'A':-1, 'C':-3, 'D':-2, 'E': 0, 'F':-3, 'G':-2, 'H': 0, 'I':-3, 'K': 2, 'L':-2, 'M':-1, 'N': 0, 'P':-2, 'Q': 1, 'R': 5, 'S':-1, 'T':-1, 'V':-3, 'W':-3, 'Y':-2},
    'S': {'A': 1, 'C':-1, 'D': 0, 'E': 0, 'F':-2, 'G': 0, 'H':-1, 'I':-2, 'K': 0, 'L':-2, 'M':-1, 'N': 1, 'P':-1, 'Q': 0, 'R':-1, 'S': 4, 'T': 1, 'V':-2, 'W':-3, 'Y':-2},
    'T': {'A': 0, 'C':-1, 'D':-1, 'E':-1, 'F':-2, 'G':-2, 'H':-2, 'I':-1, 'K':-1, 'L':-1, 'M':-1, 'N': 0, 'P':-1, 'Q':-1, 'R':-1, 'S': 1, 'T': 5, 'V': 0, 'W':-2, 'Y':-2},
    'V': {'A': 0, 'C':-1, 'D':-3, 'E':-2, 'F':-1, 'G':-3, 'H':-3, 'I': 3, 'K':-2, 'L': 1, 'M': 1, 'N':-3, 'P':-2, 'Q':-2, 'R':-3, 'S':-2, 'T': 0, 'V': 4, 'W':-3, 'Y':-1},
    'W': {'A':-3, 'C':-2, 'D':-4, 'E':-3, 'F': 1, 'G':-2, 'H':-2, 'I':-3, 'K':-3, 'L':-2, 'M':-1, 'N':-4, 'P':-4, 'Q':-2, 'R':-3, 'S':-3, 'T':-2, 'V':-3, 'W':11, 'Y': 2},
    'Y': {'A':-2, 'C':-2, 'D':-3, 'E':-2, 'F': 3, 'G':-3, 'H': 2, 'I':-1, 'K':-2, 'L':-1, 'M':-1, 'N':-2, 'P':-3, 'Q':-1, 'R':-2, 'S':-2, 'T':-2, 'V':-1, 'W': 2, 'Y': 7},
}

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

def get_blosum62_score(aa1, aa2):
    if aa1 in BLOSUM62_MATRIX and aa2 in BLOSUM62_MATRIX[aa1]:
        return BLOSUM62_MATRIX[aa1][aa2]
    return -4

def calculate_sequence_metrics(pred_seq, true_seq):
    """Calculate recovery and BLOSUM score (for evaluation mode)."""
    if len(pred_seq) != len(true_seq):
        return {'recovery': 0.0, 'blosum_score': 0, 'blosum_per_res': 0.0}
    
    matches = sum(1 for p, t in zip(pred_seq, true_seq) if p == t)
    recovery = matches / len(true_seq) if len(true_seq) > 0 else 0.0
    blosum_total = sum(get_blosum62_score(p, t) for p, t in zip(pred_seq, true_seq))
    
    return {
        'recovery': recovery,
        'blosum_score': blosum_total,
        'blosum_per_res': blosum_total / len(true_seq) if len(true_seq) > 0 else 0.0,
    }


def design_seq(args):
    # Load config
    config, _ = load_config(args.config)
    seed_val = 42
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'seed'):
        seed_val = config.sampling.seed
    seed_all(seed_val)
    
    # Preprocess structure
    in_pdb_path = args.pdb_path
    print(f"[INFO] Processing: {in_pdb_path}")
    
    # Renumber to temporary file
    import tempfile
    out_pdb_path = tempfile.mktemp(suffix='.pdb')
    heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
    
    # Determine chain IDs
    heavy_id = heavy_chains[0] if heavy_chains else args.heavy
    light_id = light_chains[0] if light_chains else args.light
    
    # Load structure
    pdb_id = os.path.basename(in_pdb_path).replace('.pdb', '')
    structure = preprocess_antibody_structure({
        'id': pdb_id,
        'pdb_path': out_pdb_path,
        'heavy_id': heavy_id,
        'light_id': light_id,
    })
    
    if structure is None:
        print("[ERROR] Failed to parse antibody structure")
        return
    
    # Apply transforms
    tx_opt = []
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'transform'):
        tx_opt = config.sampling.transform
    if not tx_opt:
        # Default transform for inference if not specified
        tx_opt = [{'type': 'mask_single_cdr', 'selection': 'H_CDR3', 'augmentation': False}]
    
    # Always ensure MergeChains is present (but don't add duplicate)
    has_merge = any(t.get('type') == 'merge_chains' for t in tx_opt)
    if not has_merge:
        tx_opt.append({'type': 'merge_chains'})
    
    transform = get_transform(tx_opt)
    data = transform(structure)
    
    # Collate
    collate_fn = PaddingCollate()
    batch = collate_fn([data])
    batch = recursive_to(batch, args.device)
    
    # Get masks
    gen_mask = batch['generate_flag'][0].bool()
    
    # Get native sequence (for eval mode)
    native_seq = None
    if args.eval:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in native_aa.cpu().tolist()])
        print(f"[INFO] Native CDR sequence: {native_seq} (length={len(native_seq)})")
    
    # Load model
    print(f"[INFO] Loading model: {config.model.checkpoint}")
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    
    model_config = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        model_config['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    
    model = get_model(model_config).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Sampling settings
    sample_opt = {'deterministic': not args.stochastic}
    mode_str = 'stochastic' if args.stochastic else 'deterministic'
    print(f"[INFO] Generating {args.num_samples} sequence(s), mode: {mode_str}")
    
    results = []
    best_ppl = float('inf')
    best_seq = None
    
    for i in tqdm(range(args.num_samples), disable=args.num_samples==1):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)
        
        # Get generated sequence
        aa_new = traj[0][2][0]
        pred_aa = aa_new[gen_mask]
        pred_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in pred_aa.cpu().tolist()])
        
        # Calculate perplexity from model's direct prediction
        pred_logits = traj['pred_logits'][0][gen_mask]  # (L_gen, num_classes)
        log_probs = torch.log_softmax(pred_logits[..., :20], dim=-1)  # Only 20 AAs
        nll = -log_probs[range(len(pred_aa)), pred_aa].mean()
        perplexity = torch.exp(nll).item()
        
        result = {
            'sample_id': i,
            'sequence': pred_seq,
            'perplexity': perplexity,
        }
        
        # Evaluation metrics (only in eval mode)
        if args.eval and native_seq:
            metrics = calculate_sequence_metrics(pred_seq, native_seq)
            result.update(metrics)
            print(f"  {i:04d}: {pred_seq} | PPL={perplexity:.2f} | Recovery={metrics['recovery']*100:.1f}%")
        else:
            print(f"  {i:04d}: {pred_seq} | PPL={perplexity:.2f}")
        
        results.append(result)
        
        if perplexity < best_ppl:
            best_ppl = perplexity
            best_seq = pred_seq
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[RESULT]")
    print(f"  Best sequence: {best_seq}")
    print(f"  Best PPL: {best_ppl:.2f}")
    
    if args.num_samples > 1:
        avg_ppl = np.mean([r['perplexity'] for r in results])
        print(f"  Avg PPL: {avg_ppl:.2f}")
    
    if args.eval and native_seq:
        print(f"  Native: {native_seq}")
        avg_recovery = np.mean([r['recovery'] for r in results])
        print(f"  Avg Recovery: {avg_recovery*100:.1f}%")
    
    print(f"{'='*60}")
    
    # Save results
    if args.output:
        import json
        output_data = {
            'best_sequence': best_seq,
            'best_perplexity': best_ppl,
            'samples': results,
        }
        if args.eval and native_seq:
            output_data['native'] = native_seq
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"[INFO] Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Antibody CDR Sequence Design using BFN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (deterministic, single sequence)
  python design_seq.py structure.pdb --heavy A --light B --config config.yml --device mps
  
  # Generate multiple random samples
  python design_seq.py structure.pdb --config config.yml --num_samples 10 --stochastic
  
  # Evaluation mode (compare to native)
  python design_seq.py structure.pdb --config config.yml --eval
  
  # Save results to file
  python design_seq.py structure.pdb --config config.yml --output results.json
        """
    )
    parser.add_argument('pdb_path', type=str, help='Input PDB file with antibody structure')
    parser.add_argument('--heavy', type=str, default='H', help='Heavy chain ID (default: H)')
    parser.add_argument('--light', type=str, default='L', help='Light chain ID (default: L)')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file')
    parser.add_argument('--num_samples', '-n', type=int, default=1, help='Number of sequences (default: 1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic sampling (default: deterministic)')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode: compare to native sequence')
    
    args = parser.parse_args()
    design_seq(args)
