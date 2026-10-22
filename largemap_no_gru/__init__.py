"""Independent feed-forward LargeMap project."""
from pathlib import Path
import sys
# Vendored internal helper namespace; never add another repository/runtime.
_INTERNAL=Path(__file__).resolve().parents[1]/'scripts'
if str(_INTERNAL) not in sys.path:sys.path.insert(0,str(_INTERNAL))
__version__='0.1.0'
