'''
icepick: gpu discovery
'''

# Import external dependencies
import os, subprocess

# _SLURM: scheduler variables listing the GPUs assigned to a job or step, in order of preference
_SLURM: tuple[str, ...] = ('SLURM_STEP_GPUS', 'SLURM_JOB_GPUS', 'GPU_DEVICE_ORDINAL')

# assigned_gpus: identifiers (indices or UUIDs) of the GPUs this process may use, empty when none
def assigned_gpus() -> list[str]:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:    # set but empty means none, so test for None not truthiness
        return _split(visible)
    for name in _SLURM:
        if name in os.environ:
            return _split(os.environ[name])
    return _nvidia_smi()

# require_gpu: raise unless a GPU is assigned and cupy imports, pinning device 0 to the first assigned GPU
def require_gpu() -> None:
    gpus = assigned_gpus()
    if not gpus:
        raise RuntimeError('no GPU is assigned to this process (check CUDA_VISIBLE_DEVICES or the scheduler allocation)')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', ','.join(gpus))    # scheduler-only assignment, otherwise cupy would see all
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError("cupy is not installed, install with 'pip install icepick[gpu]'") from error
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('cupy found no CUDA device')

# _split: comma separated device list to identifiers, dropping blanks and the CUDA '-1' no-device marker
def _split(value: str) -> list[str]:
    ids = [v.strip() for v in value.split(',') if v.strip()]
    return ids[:ids.index('-1')] if '-1' in ids else ids    # CUDA stops reading the list at the first invalid entry

# _nvidia_smi: GPU indices from 'nvidia-smi -L', empty when the tool is missing or fails
def _nvidia_smi() -> list[str]:
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10, check=True).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.split(':')[0].split()[1] for line in out.splitlines() if line.startswith('GPU ')]
