from __future__ import annotations

import importlib.util
from pathlib import Path

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


_ROOT = Path(__file__).resolve().parent


def _load_module(module_name: str, filename: str):
    module_path = _ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_QWEN_MODULE = _load_module("mrflow_qwen_nodes", "nodes_mrflow_qwen.py")
_KREA2_MODULE = _load_module("mrflow_krea2_nodes", "nodes_mrflow_krea2.py")
_TILED_MODULE = _load_module("mrflow_tiled_nodes", "nodes_mrflow_tiled.py")
_MODULES = (_QWEN_MODULE, _KREA2_MODULE, _TILED_MODULE)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in _MODULES:
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)


class MrFlowExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes: list[type[io.ComfyNode]] = []
        for module in _MODULES:
            extension = await module.comfy_entrypoint()
            nodes.extend(await extension.get_node_list())
        return nodes


async def comfy_entrypoint() -> MrFlowExtension:
    return MrFlowExtension()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
