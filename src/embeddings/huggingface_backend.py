import gc
import logging
from asyncio import sleep
from enum import StrEnum
from pathlib import Path
import platform
from typing import Literal

from asgiref.sync import sync_to_async
from django.conf import settings

from embeddings.embedding_backend import EmbeddingBackend

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    name = "huggingface"
    is_async_prefer = False

    class InputType(StrEnum):
        DOCUMENT = "document"
        QUERY = "query"
        RETRIEVAL_QUERY = ""
        RETRIEVAL_DOCUMENT = ""

    PROMPTS = {
        "google-gemma": {
            InputType.QUERY: "task: search result | query: ",
            InputType.DOCUMENT: "title: {title} | text: ",
            InputType.RETRIEVAL_QUERY: "task: search result | query: ",
            InputType.RETRIEVAL_DOCUMENT: "title: {title} | text: ",
        }
    }

    def __init__(self, model: str = None, dimensions: int = None, api_key: str = None, **kwargs):
        model = model or settings.HUGGINGFACE_EMBEDDING_MODEL_NAME
        super().__init__(model, dimensions)
        assert self.model, "HUGGINGFACE_EMBEDDING_MODEL_NAME must be set"
        self.api_key = api_key
        self._client = None

    @staticmethod
    def get_optimized_onnx_file_name(onnx_model_path: Path) -> str | None:
        """
        Selects the best available ONNX file based on CPU architecture and performance priority.
        """
        if not onnx_model_path.exists():
            return None  # Return None if the model directory doesn't exist

        if not settings.OPTIMIZED_ONNX_FILE_ENABLED:
            return None

        # 1. Check for ARM64/AARCH64 (Apple Silicon, ARM Servers)
        if platform.machine() in ("aarch64", "arm64"):
            # Priority: ARM Quantized > O4 Optimized > Default
            priority_list = ["model_qint8_arm64.onnx", "model_O4.onnx", "model.onnx"]

        # 2. Check for x86_64/AMD64 (Standard Intel/AMD CPUs)
        elif platform.machine() in ("x86_64", "amd64"):
            # Dynamically build priority list based on CPU features for best performance.
            priority_list = []
            try:
                import cpuinfo

                info = cpuinfo.get_cpu_info()
                flags = info.get("flags", [])

                # Highest priority: VNNI for best INT8 performance on modern CPUs.
                if "avx512_vnni" in flags:
                    priority_list.append("model_qint8_avx512_vnni.onnx")

                # Next best: AVX2 is very common and provides good speedup.
                if "avx2" in flags:
                    priority_list.append("model_quint8_avx2.onnx")

            except ImportError:
                logger.warning(
                    "The 'py-cpuinfo' package is not installed. Cannot detect CPU features for ONNX optimization. "
                    "Falling back to a generic priority list. Run 'pip install py-cpuinfo' for better performance."
                )
                # Fallback if py-cpuinfo is not available
                priority_list.extend(["model_qint8_avx512_vnni.onnx", "model_quint8_avx2.onnx"])

            # Add non-quantized and default models as fallbacks.
            priority_list.extend(["model_O4.onnx", "model.onnx"])

        # 3. Fallback for unknown/other architectures
        else:
            # Fall back to generic optimized and default
            priority_list = ["model_O4.onnx", "model.onnx"]

        # Search for the highest priority file that exists
        for file_name in priority_list:
            # NOTE: The file name in the warning is relative to the base model folder,
            # e.g., 'onnx/model_O4.onnx', so we use `Path(file_name)`
            file_path = onnx_model_path / file_name

            # Check if the file exists relative to the provided base path
            if file_path.exists():
                # Return the path that the SentenceTransformer expects: relative to the model's root
                return f"{file_path.parent.name}/{file_path.name}"

        return None  # Return None if no suitable model file was found

    def get_client(self):
        import torch  # noqa
        from sentence_transformers import SentenceTransformer

        backend_list = ("torch", "onnx", "openvino")
        backend: Literal["torch", "onnx", "openvino"] = settings.SENTENCE_TRANSFORMER_BACKEND

        if backend not in backend_list:
            backend = backend_list[0]  # noqa
            logger.error(
                "Invalid SENTENCE_TRANSFORMER_BACKEND value, must be one of: %s, used default: %s",
                backend_list,
                backend,
            )

        cache_dir: Path = settings.EMBEDDING_MODELS_CACHE_DIR
        cache_dir.mkdir(exist_ok=True, parents=True)
        sentence_transformer_extra = {}
        if backend == "onnx":
            import onnxruntime  # noqa
            from huggingface_hub import snapshot_download

            try:
                model_path = Path(snapshot_download(repo_id=self.model, cache_dir=cache_dir))
                onnx_model_path = model_path / "onnx"
                quantized_file = onnx_model_path / "model.onnx"
                if not quantized_file.exists():
                    logger.warning(
                        f"ONNX model not found for {self.model}. "
                        "You can use 'python embeddings/convert_onnx.py' for conversion. "
                        f"Switching to 'torch' backend to avoid on-the-fly export. "
                        f"Set SENTENCE_TRANSFORMER_BACKEND to 'torch' in your settings to make this permanent."
                    )
                    backend = "torch"
                else:
                    quantized_file_name = (
                        self.get_optimized_onnx_file_name(onnx_model_path)
                        or f"{quantized_file.parent.name}/{quantized_file.name}"
                    )
                    # quantized_file_name = f"{quantized_file.parent.name}/{quantized_file.name}"
                    if quantized_file_name:
                        logger.debug(f"ONNX {quantized_file_name=}")
                        sentence_transformer_extra["model_kwargs"] = {"file_name": quantized_file_name}
            except Exception as e:
                logger.warning(f"Could not check for ONNX model for {self.model}: {e}. Proceeding with onnx backend.")

        self.api_key = self.api_key or settings.HUGGINGFACE_API_KEY
        if self.api_key:
            from huggingface_hub import login

            login(token=self.api_key)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        client = SentenceTransformer(
            self.model, cache_folder=str(cache_dir), backend=backend, **sentence_transformer_extra
        ).to(device)
        # if self.dimensions:
        #     client.max_seq_length = self.dimensions

        # The first module is the tokenizer and model. If it's None, the model failed to load.
        if not client or not client._first_module():
            logger.error(f"Failed to load SentenceTransformer model: {self.model}")
            logger.error(
                f"Could not load SentenceTransformer model '{self.model}'. Please check the model name and configuration."
            )
            client = None

        return client

    async def aclose(self):
        self.close()
        await sleep(0)

    def close(self):
        """Releases the SentenceTransformer model from memory."""
        if self._client:
            logger.debug("Releasing HuggingFace SentenceTransformer model from memory.")
            try:
                # Check if the model is on a CUDA device before trying to empty the cache
                if any(p.is_cuda for p in self._client.parameters()):
                    import torch

                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning("Failed to empty CUDA cache: %s", e)

            self._client = None
            gc.collect()

    def embed_text(
        self, text: str | list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | list[list[float]] | None:
        if not text:
            return None
        return self.embed_texts(text, input_type=input_type)

    def embed_texts(
        self, texts: str | list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[float] | list[list[float]] | None:
        if not texts:
            return None
        if isinstance(texts, list):
            logger.debug(f"embed_texts: texts count: {len(texts)}. {self.get_prompt_name(input_type)}")
        else:
            logger.debug(f"embed_texts: text: {texts[:20]}. {self.get_prompt_name(input_type)}")

        try:
            embeddings = self.client.encode(
                texts, convert_to_tensor=False, prompt_name=self.get_prompt_name(input_type)
            )
            if embeddings is None:
                logger.error("Invalid embeddings")
                return None

            result = embeddings.tolist()
            if not result:
                logger.error(f"Invalid result {embeddings}")
                return None

            return result
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            return None

    async def aembed_text(self, text: str, input_type: EmbeddingBackend.InputType = None) -> list[float] | None:
        logger.debug("async mode")
        return await sync_to_async(self.embed_text)(text, input_type=input_type)

    async def aembed_texts(
        self, texts: list[str], input_type: EmbeddingBackend.InputType = None
    ) -> list[list[float]] | None:
        logger.debug("async mode")
        return await sync_to_async(self.embed_texts)(texts, input_type=input_type)
