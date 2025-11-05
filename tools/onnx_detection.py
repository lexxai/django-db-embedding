def onnxruntime_detect():
    try:
        import onnxruntime as ort

        # Check available providers
        print("Available providers:", ort.get_available_providers())

        # CPU
        session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

        # CUDA
        session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

        # TensorRT (requires onnxruntime-gpu)
        session = ort.InferenceSession("model.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])

        # DirectML (Windows)
        session = ort.InferenceSession("model.onnx", providers=["DmlExecutionProvider"])

        # ROCm
        session = ort.InferenceSession("model.onnx", providers=["ROCMExecutionProvider"])

        # Auto-fallback
        session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except ImportError:
        print("onnxruntime not installed.")


if __name__ == "__main__":
    onnxruntime_detect()
