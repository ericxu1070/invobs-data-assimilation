# notebook_setup.py
"""Environment detection + cache + dependency install helpers shared by all v2 notebooks.

Works in local Jupyter, Google Colab (browser), and VS Code Colab extension.
"""
import importlib
import os
import subprocess
import sys


def detect_colab() -> bool:
    """True for Google Colab runtimes, including VS Code Colab extension sessions."""
    try:
        importlib.import_module('google.colab')
        return True
    except ImportError:
        return False


def pip_install(*pkgs, quiet: bool = True) -> None:
    args = [sys.executable, '-m', 'pip', 'install']
    if quiet:
        args.append('-q')
    args.extend(pkgs)
    subprocess.check_call(args)


def ensure_packages(packages: dict) -> None:
    """packages: {import_name: pip_spec}. Install only missing ones."""
    missing = []
    for import_name, pip_spec in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        print(f'Installing: {missing}')
        pip_install(*missing)


def setup_cache(default_local: str = '~/invobs_cache') -> str:
    """Return cache directory path. Mount Drive on Colab if needed.

    Resolution order:
      1. INVOBS_CACHE_DIR env var (highest priority, used by all environments).
      2. /content/drive/MyDrive/invobs_cache on Colab.
      3. ~/invobs_cache locally.
    """
    in_colab = detect_colab()
    env_override = os.environ.get('INVOBS_CACHE_DIR')

    if env_override:
        cache_dir = env_override
    elif in_colab:
        cache_dir = '/content/drive/MyDrive/invobs_cache'
    else:
        cache_dir = os.path.expanduser(default_local)

    if in_colab and cache_dir.startswith('/content/drive') and not os.path.ismount('/content/drive'):
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def setup_device():
    """Return a torch.device, preferring CUDA when available."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def banner(cache_dir: str, device, in_colab: bool) -> None:
    print(f'environment: {"colab" if in_colab else "local"}')
    print(f'device:      {device}')
    print(f'cache dir:   {cache_dir}')
