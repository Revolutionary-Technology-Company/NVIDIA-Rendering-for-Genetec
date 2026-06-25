# tests/test_imports.py
import os
import sys

# Inject hardware mocks before importing production code
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import mock_dependencies

def test_pipeline_syntax_and_imports():
    """Verifies that the main pipeline structure resolves cleanly."""
    try:
        import yolov8_triton_parser as parser
        import telemetry_producer as telemetry
        import ocr_layer as ocr
        print("[CI SUCCESS] Core analytics and JIT parser blocks resolved cleanly.")
        assert True
    except Exception as e:
        print(f"[CI FAILURE] Integration compilation error: {str(e)}")
        assert False

if __name__ == "__main__":
    test_pipeline_syntax_and_imports()
