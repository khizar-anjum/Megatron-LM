#!/usr/bin/env python3
"""
Test script for Adaptive Pipeline Monitoring

This script demonstrates the initial profiling and runtime monitoring
capabilities for adaptive pipeline parallelism.

Usage:
    # Single node, 2 GPUs:
    torchrun --nproc_per_node=2 test_adaptive_monitoring.py

    # Multi-node with SLURM:
    srun python test_adaptive_monitoring.py
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add Megatron to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.initial_profiling import (
    run_initial_profiling,
    print_network_stats
)
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Adaptive Pipeline Monitoring')

    # Parallelism options
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1,
                        help='Degree of tensor model parallelism')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=2,
                        help='Degree of pipeline model parallelism')

    # Monitoring options
    parser.add_argument('--enable-initial-profiling', action='store_true',
                        help='Enable initial network profiling at startup')
    parser.add_argument('--enable-p2p-monitoring', action='store_true',
                        help='Enable runtime P2P monitoring')
    parser.add_argument('--p2p-monitoring-sample-rate', type=float, default=0.1,
                        help='Sample rate for P2P monitoring (0.0-1.0)')

    # Model options
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Hidden size')
    parser.add_argument('--num-attention-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--seq-length', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--micro-batch-size', type=int, default=2,
                        help='Micro batch size')
    parser.add_argument('--num-microbatches', type=int, default=4,
                        help='Number of microbatches')

    # Training options
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='Number of training iterations')

    return parser.parse_args()


def initialize_distributed(args):
    """Initialize distributed training."""
    # Initialize torch.distributed
    if not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Launched with torchrun
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            # Launched with SLURM
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
        else:
            raise RuntimeError("Cannot determine distributed setup")

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Initialize model parallel
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size
    )

    return dist.get_rank(), dist.get_world_size()


def create_model(args):
    """Create a simple GPT model for testing."""

    # Create transformer config
    transformer_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32
    )

    # Create model parallel config with monitoring enabled
    model_parallel_config = ModelParallelConfig(
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        enable_p2p_monitoring=args.enable_p2p_monitoring,
        p2p_monitoring_sample_rate=args.p2p_monitoring_sample_rate,
        enable_initial_profiling=args.enable_initial_profiling,
    )

    # For simplicity, we'll create a minimal model
    # In a real scenario, you'd use the full GPTModel
    model = torch.nn.Sequential(
        torch.nn.Linear(args.hidden_size, args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, args.hidden_size)
    ).cuda()

    return model, model_parallel_config


def create_dummy_batch(args):
    """Create dummy input data for testing."""
    # Create random input
    input_ids = torch.randint(
        0, 1000,
        (args.micro_batch_size, args.seq_length),
        device='cuda'
    )

    # Create random labels
    labels = torch.randint(
        0, 1000,
        (args.micro_batch_size, args.seq_length),
        device='cuda'
    )

    return input_ids, labels


def run_test(args, rank):
    """Run the test with monitoring."""

    if rank == 0:
        print("\n" + "=" * 60)
        print("Adaptive Pipeline Monitoring Test")
        print("=" * 60)
        print(f"World size: {dist.get_world_size()}")
        print(f"Pipeline parallel size: {args.pipeline_model_parallel_size}")
        print(f"Tensor parallel size: {args.tensor_model_parallel_size}")
        print(f"Initial profiling: {args.enable_initial_profiling}")
        print(f"Runtime monitoring: {args.enable_p2p_monitoring}")
        if args.enable_p2p_monitoring:
            print(f"Sample rate: {args.p2p_monitoring_sample_rate:.1%}")
        print("=" * 60)

    # Phase 1: Initial Profiling (if enabled)
    topology = None
    if args.enable_initial_profiling:
        if rank == 0:
            print("\n### Running Initial Network Profiling ###\n")

        topology = run_initial_profiling(
            enable_profiling=True,
            verbose=(rank == 0)
        )

        if rank == 0 and topology:
            topology.print_summary(rank=0)

    # Phase 2: Create Model
    if rank == 0:
        print("\n### Creating Model ###\n")

    model, config = create_model(args)

    # Phase 3: Simple Training Loop with Monitoring
    if rank == 0:
        print("\n### Running Training Iterations ###\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for iteration in range(args.num_iterations):
        start_time = time.time()

        # Create dummy batch
        input_ids, labels = create_dummy_batch(args)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids.float())

        # Dummy loss
        loss = output.sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start_time

        if rank == 0:
            print(f"Iteration {iteration + 1}/{args.num_iterations}: "
                  f"loss={loss.item():.4f}, time={elapsed:.3f}s")

        # Print monitoring stats every few iterations
        if args.enable_p2p_monitoring and (iteration + 1) % 5 == 0:
            # Note: In a real pipeline parallel setup, you'd access the
            # p2p_communicator from the schedule function
            if rank == 0:
                print(f"\n[Iteration {iteration + 1}] "
                      f"Runtime monitoring would show network stats here")

    # Final Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run with --enable-initial-profiling to see topology discovery")
        print("2. Run with --enable-p2p-monitoring for runtime stats")
        print("3. Test on multiple nodes to see inter-node bandwidth")
        print("=" * 60)


def main():
    """Main function."""
    args = parse_args()

    # Initialize distributed
    rank, world_size = initialize_distributed(args)

    # Validate arguments
    if world_size < args.pipeline_model_parallel_size:
        if rank == 0:
            print(f"Error: World size ({world_size}) must be >= "
                  f"pipeline parallel size ({args.pipeline_model_parallel_size})")
        sys.exit(1)

    # Run test
    try:
        run_test(args, rank)
    except Exception as e:
        print(f"Rank {rank}: Error during test: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.barrier()
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
