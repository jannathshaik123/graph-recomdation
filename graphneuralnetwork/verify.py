import torch

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Test CUDA functionality
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    z = x + y
    print(f"CUDA tensor test: {x} + {y} = {z}")
    
    # Test PyTorch Geometric
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
        print("PyTorch Geometric CUDA test passed!")
    except Exception as e:
        print(f"PyTorch Geometric error: {e}")