# 使用 pkgutil 扩展 examples 命名空间，合并 Relax/examples 和 workspace/examples
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
