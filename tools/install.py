import subprocess
import sys


def detect_gpu():
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return "cuda"
    except:
        pass

    try:
        subprocess.run(["rocminfo"], check=True, capture_output=True)
        return "rocm"
    except:
        pass

    return "cpu"


if __name__ == "__main__":
    gpu_type = detect_gpu()
    print(f"Detected: {gpu_type}")
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-e", f".[{gpu_type}]"])
