"""Common utilities for Megatron alignment tests.

Provides:
  - Path resolution and validation for Megatron-LM source code
  - Stub construction for Megatron internal dependencies
  - importlib helpers to load Megatron modules without full package init
"""

import glob
import importlib.util
import os
import sys


def get_megatron_dir() -> str:
    r"""Return the path to Megatron-LM source root.

    Resolution order:
      1. ``MEGATRON_LM_ROOT`` environment variable
      2. Default relative path ``../../../Megatron-LM`` from this file
    """
    return os.environ.get(
        "MEGATRON_LM_ROOT",
        os.path.join(os.path.dirname(__file__), "../../../Megatron-LM"),
    )


def check_megatron_source(require_helpers_cpp: bool = True) -> str:
    r"""Verify Megatron-LM source is available and return its root path.

    Args:
        require_helpers_cpp: If ``True`` (default), also require that a compiled
            ``helpers_cpp*.so`` extension is present.

    Raises:
        RuntimeError: If the source directory or compiled helpers extension is missing.
    """
    megatron_dir = get_megatron_dir()
    if not os.path.isdir(megatron_dir):
        raise RuntimeError(
            f"Megatron-LM source not found at {megatron_dir}.\n"
            f"Please clone Megatron-LM to the default relative path (../../../Megatron-LM) "
            f"or set the MEGATRON_LM_ROOT environment variable, e.g.:\n"
            f"    export MEGATRON_LM_ROOT=/path/to/Megatron-LM"
        )

    if require_helpers_cpp:
        candidates = glob.glob(
            os.path.join(megatron_dir, "megatron/core/datasets/helpers_cpp*.so")
        )
        if not candidates:
            raise RuntimeError(
                f"Compiled Megatron helpers extension (helpers_cpp*.so) not found under "
                f"{megatron_dir}/megatron/core/datasets/. Please build Megatron-LM first, e.g.:\n"
                f"    cd /path/to/Megatron-LM && pip install -e ."
            )

    return megatron_dir


def build_megatron_stubs(include_tokenizer: bool = False) -> None:
    r"""Build minimal module stubs for Megatron data layers.

    Megatron's top-level ``__init__.py`` triggers imports of distributed,
    optimizer, and pipeline-parallel modules that pull in heavy dependencies
    (Apex, Transformer Engine, torch.distributed.DTensor, etc.).
    By pre-registering stub modules in ``sys.modules`` we bypass those imports
    and can safely load individual source files via ``importlib``.

    Args:
        include_tokenizer: If ``True``, also stub ``megatron.core.tokenizers``
            with a minimal ``MegatronTokenizerBase`` fake class.
    """
    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.datasets",
        "megatron.core.datasets.object_storage_utils",
        "megatron.core.msc_utils",
        "megatron.core.utils",
    ]:
        if name not in sys.modules:
            sys.modules[name] = type(sys)(name)

    # object_storage_utils stubs
    class _S3Config:
        pass

    class _ObjectStorageConfig:
        pass

    def _noop(*_a, **_k):
        pass

    def _false(*_a, **_k):
        return False

    def _empty(*_a, **_k):
        return ""

    def _parse_s3_path(*_a, **_k):
        return ("", "")

    _osu = sys.modules["megatron.core.datasets.object_storage_utils"]
    _osu.S3Config = _S3Config
    _osu.ObjectStorageConfig = _ObjectStorageConfig
    _osu.cache_index_file = _noop
    _osu.dataset_exists = _false
    _osu.get_index_cache_path = _empty
    _osu.get_object_storage_access = _empty
    _osu.is_object_storage_path = _false
    _osu.parse_s3_path = _parse_s3_path

    # msc_utils stub
    class _MultiStorageClientFeature:
        @staticmethod
        def is_enabled():
            return False

        @staticmethod
        def import_package():
            raise ImportError

    sys.modules["megatron.core.msc_utils"].MultiStorageClientFeature = (
        _MultiStorageClientFeature
    )
    sys.modules["megatron.core.utils"].log_single_rank = _noop

    if include_tokenizer:
        class _FakeTokenizer:
            vocab_size = 50000
            eod = 2
            eos = 2
            pad = -1
            special_tokens_dict = {}
            unique_identifiers = {}

        if "megatron.core.tokenizers" not in sys.modules:
            sys.modules["megatron.core.tokenizers"] = type(sys)("megatron.core.tokenizers")
        sys.modules["megatron.core.tokenizers"].MegatronTokenizerBase = _FakeTokenizer


def find_helpers_cpp(megatron_dir: str) -> str:
    r"""Return the path to the compiled ``helpers_cpp*.so`` extension.

    Raises:
        RuntimeError: If no matching ``.so`` file is found.
    """
    candidates = glob.glob(
        os.path.join(megatron_dir, "megatron/core/datasets/helpers_cpp*.so")
    )
    if not candidates:
        raise RuntimeError(
            f"Compiled Megatron helpers extension (helpers_cpp*.so) not found under "
            f"{megatron_dir}/megatron/core/datasets/."
        )
    return candidates[0]


def load_megatron_module(module_name: str, file_path: str):
    r"""Load a Megatron source file via ``importlib`` and register it.

    Args:
        module_name: Dotted module name (e.g. ``megatron.core.datasets.gpt_dataset``).
        file_path: Absolute path to the ``.py`` or ``.so`` file.

    Returns:
        The loaded module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod
