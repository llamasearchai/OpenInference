#!/usr/bin/env python
import argparse
import logging
import os
import sys
import time
import json
from typing import Dict, Any, List

from .api.server import start_server
from .hardware.accelerator import HardwareManager, AcceleratorType
from .models.registry import ModelRegistry
from .monitoring.performance_tracker import PerformanceTracker
from .optimization.quantization import PyTorchQuantizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('openinference.cli')


def get_version():
    """Get OpenInference version."""
    from . import __version__
    return __version__


def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for display."""
    return json.dumps(data, indent=2)


def command_serve(args):
    """Run the inference server."""
    logger.info(f"Starting OpenInference server on {args.host}:{args.port}")
    
    # Initialize hardware manager
    device_type = AcceleratorType(args.device) if args.device else None
    hw_manager = HardwareManager(prefer_device_type=device_type)
    
    # Initialize model registry
    model_registry = ModelRegistry(
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        hardware_manager=hw_manager
    )
    
    # Initialize performance tracker
    perf_tracker = PerformanceTracker(
        record_gpu_metrics=not args.disable_gpu_metrics,
        export_metrics=not args.disable_metrics_export
    )
    
    # Start server
    start_server(
        host=args.host,
        port=args.port,
        model_registry=model_registry,
        hardware_manager=hw_manager,
        performance_tracker=perf_tracker,
        workers=args.workers,
        max_batch_size=args.max_batch_size,
        enable_ui=not args.disable_ui
    )


def command_list_models(args):
    """List available models."""
    # Initialize hardware manager
    hw_manager = HardwareManager()
    
    # Initialize model registry
    model_registry = ModelRegistry(
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        hardware_manager=hw_manager
    )
    
    # Get available models
    models = model_registry.list_available_models()
    
    if args.json:
        print(format_json(models))
    else:
        print("\nAvailable Models:")
        print("-----------------")
        for model in models:
            print(f"Name: {model['name']}")
            print(f"Type: {model['type']}")
            print(f"Size: {model['size_mb']:.2f} MB")
            print(f"Status: {model['status']}")
            print(f"Loaded: {model['is_loaded']}")
            print("-----------------")


def command_benchmark(args):
    """Run benchmarking on models."""
    from .benchmarking.benchmark_runner import run_benchmark
    
    # Initialize hardware manager
    device_type = AcceleratorType(args.device) if args.device else None
    hw_manager = HardwareManager(prefer_device_type=device_type)
    
    # Initialize model registry
    model_registry = ModelRegistry(
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        hardware_manager=hw_manager
    )
    
    # Initialize performance tracker
    perf_tracker = PerformanceTracker(
        record_gpu_metrics=not args.disable_gpu_metrics,
        export_metrics=True
    )
    
    # Run benchmark
    results = run_benchmark(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        iterations=args.iterations,
        model_registry=model_registry,
        performance_tracker=perf_tracker,
        warmup_iterations=args.warmup
    )
    
    # Display results
    if args.json:
        print(format_json(results))
    else:
        print("\nBenchmark Results:")
        print("-----------------")
        for batch_size, batch_results in results.items():
            print(f"\nBatch Size: {batch_size}")
            for seq_len, metrics in batch_results.items():
                print(f"  Sequence Length: {seq_len}")
                print(f"    Latency (ms): {metrics['latency_ms']['avg']:.2f} avg, {metrics['latency_ms']['p99']:.2f} p99")
                print(f"    Throughput: {metrics['throughput']['items_per_second']:.2f} items/s")
                if 'tokens_per_second' in metrics['throughput']:
                    print(f"    Token Throughput: {metrics['throughput']['tokens_per_second']:.2f} tokens/s")


def command_hardware_info(args):
    """Display hardware information."""
    # Initialize hardware manager
    hw_manager = HardwareManager()
    
    # Get hardware information
    devices = hw_manager.get_all_devices()
    selected = hw_manager.get_selected_device_info()
    
    if selected:
        selected_info = selected.to_dict()
    else:
        selected_info = {"error": "No device selected"}
    
    # Display information
    if args.json:
        print(format_json({
            "devices": devices,
            "selected": selected_info,
            "optimization_flags": {
                "fp16": hw_manager.use_fp16,
                "bf16": hw_manager.use_bf16,
                "amp": hw_manager.use_amp,
                "tensorrt": hw_manager.use_trt
            }
        }))
    else:
        print("\nHardware Information:")
        print("---------------------")
        print(f"Selected Device: {selected}")
        print("\nOptimization Flags:")
        print(f"  FP16: {hw_manager.use_fp16}")
        print(f"  BF16: {hw_manager.use_bf16}")
        print(f"  AMP: {hw_manager.use_amp}")
        print(f"  TensorRT: {hw_manager.use_trt}")
        
        print("\nAvailable Devices:")
        for device_type, device_list in devices.items():
            print(f"\n{device_type.upper()}:")
            for device in device_list:
                print(f"  ID: {device['device_id']}")
                print(f"  Name: {device['name']}")
                print(f"  Memory: {device['memory_gb']:.2f} GB")
                if device['compute_capability']:
                    print(f"  Compute Capability: {device['compute_capability']}")


def command_quantize(args):
    """Quantize a model."""
    # Initialize hardware manager
    hw_manager = HardwareManager()
    
    # Initialize model registry
    model_registry = ModelRegistry(
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        hardware_manager=hw_manager
    )
    
    # Load the model
    logger.info(f"Loading model {args.model}...")
    model = model_registry.load_model(args.model)
    
    if model is None:
        logger.error(f"Model {args.model} not found or could not be loaded")
        return 1
    
    # Initialize quantizer
    quantizer = PyTorchQuantizer(target_precision=args.precision)
    
    # Quantize the model
    logger.info(f"Quantizing model to {args.precision}...")
    try:
        quantized_model = quantizer.quantize(model)
        
        # Save the quantized model
        output_path = args.output if args.output else f"{args.model}_{args.precision}.pt"
        torch.save(quantized_model, output_path)
        
        # Show stats
        stats = quantizer.get_quantization_stats()
        
        if args.json:
            print(format_json(stats))
        else:
            print("\nQuantization Results:")
            print("---------------------")
            print(f"Original Size: {stats['original_size_bytes'] / (1024**2):.2f} MB")
            print(f"Quantized Size: {stats['quantized_size_bytes'] / (1024**2):.2f} MB")
            print(f"Size Reduction: {stats['size_reduction_percent']:.2f}%")
            print(f"Output Path: {output_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="OpenInference - High-Performance AI Inference Engine")
    parser.add_argument('--version', action='version', version=f'OpenInference {get_version()}')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Run the inference server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    serve_parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    serve_parser.add_argument('--max-batch-size', type=int, default=32, help='Maximum batch size')
    serve_parser.add_argument('--models-dir', type=str, default='models', help='Directory containing models')
    serve_parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory for caching models')
    serve_parser.add_argument('--device', type=str, choices=[e.value for e in AcceleratorType], help='Preferred device type')
    serve_parser.add_argument('--disable-ui', action='store_true', help='Disable web UI')
    serve_parser.add_argument('--disable-gpu-metrics', action='store_true', help='Disable GPU metrics collection')
    serve_parser.add_argument('--disable-metrics-export', action='store_true', help='Disable metrics export')
    serve_parser.set_defaults(func=command_serve)
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.add_argument('--models-dir', type=str, default='models', help='Directory containing models')
    list_parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory for caching models')
    list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    list_parser.set_defaults(func=command_list_models)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('model', type=str, help='Model to benchmark')
    benchmark_parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16], help='Batch sizes to test')
    benchmark_parser.add_argument('--sequence-lengths', type=int, nargs='+', default=[32, 128, 512], help='Sequence lengths to test')
    benchmark_parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    benchmark_parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    benchmark_parser.add_argument('--models-dir', type=str, default='models', help='Directory containing models')
    benchmark_parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory for caching models')
    benchmark_parser.add_argument('--device', type=str, choices=[e.value for e in AcceleratorType], help='Device to use')
    benchmark_parser.add_argument('--disable-gpu-metrics', action='store_true', help='Disable GPU metrics collection')
    benchmark_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    benchmark_parser.set_defaults(func=command_benchmark)
    
    # Hardware info command
    hardware_parser = subparsers.add_parser('hardware-info', help='Display hardware information')
    hardware_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    hardware_parser.set_defaults(func=command_hardware_info)
    
    # Quantize command
    quantize_parser = subparsers.add_parser('quantize', help='Quantize a model')
    quantize_parser.add_argument('model', type=str, help='Model to quantize')
    quantize_parser.add_argument('--precision', type=str, choices=['int8', 'int4', 'fp16', 'bf16'], default='int8', help='Target precision')
    quantize_parser.add_argument('--output', type=str, help='Output path for quantized model')
    quantize_parser.add_argument('--models-dir', type=str, default='models', help='Directory containing models')
    quantize_parser.add_argument('--cache-dir', type=str, default='.cache', help='Directory for caching models')
    quantize_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    quantize_parser.set_defaults(func=command_quantize)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())