import torch
from torch import nn
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class HookInfo:
    handle: torch.utils.hooks.RemovableHandle
    level: Optional[int] = None


class HookedModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hooks: List[HookInfo] = []
        self.context_level: int = 0

    @contextmanager
    def post_hooks(self, fwd: List[Tuple[str, Callable]] = [], bwd: List[Tuple[str, Callable]] = []):
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_hook(hook_fn) #if you want to modify input, use pre_hook
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self
        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    @contextmanager
    def pre_hooks(self, fwd: List[Tuple[str, Callable]] = [], bwd: List[Tuple[str, Callable]] = []):
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_pre_hook(hook_fn) #if you want to modify input, use pre_hook
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self
        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module

    def print_model_structure(self):
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
