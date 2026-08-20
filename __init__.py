"""ComfyUI MAINodes — LTX-2.3 motion-only adaptation.

This package intentionally uses unique node IDs/display names so it can be
installed next to the original ComfyUI-MAINodes package without collisions.
"""

from .ltx23_motion import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
