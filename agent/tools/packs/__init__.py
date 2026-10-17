"""工具包集合。每个模块在 import 时通过 @tool 装饰器注册自己的工具。
加入新插件：新建 `xxx.py` 用 `@tool(..., pack="xxx")` 注册，再把模块名加进 _PACK_MODULES。
"""
import importlib

_PACK_MODULES = ("fs", "code", "exec", "todo", "wiki")

for _name in _PACK_MODULES:
    importlib.import_module(f"{__name__}.{_name}")
