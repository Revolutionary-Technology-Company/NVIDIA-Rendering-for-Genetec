# tests/mock_dependencies.py
import sys
from unittest.mock import MagicMock

# 1. Mock the low-level NVIDIA DeepStream Python bindings (pyds)
mock_pyds = MagicMock()
mock_pyds.gst_buffer_get_nvds_batch_meta = MagicMock(return_value=MagicMock())
mock_pyds.NvDsFrameMeta = MagicMock()
mock_pyds.NvDsObjectMeta = MagicMock()
mock_pyds.get_nvds_Layer_Array = MagicMock(return_value=[[[0,0,0,0]]])
sys.modules['pyds'] = mock_pyds

# 2. Mock GStreamer and GLib GObject Introspection bindings
sys.modules['gi'] = MagicMock()
sys.modules['gi.repository'] = MagicMock()
sys.modules['gi.repository.Gst'] = MagicMock()
sys.modules['gi.repository.GLib'] = MagicMock()

print("[CI MOCK] Successfully injected NVIDIA DeepStream and GStreamer dummy layers.")
