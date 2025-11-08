import importlib.metadata as im
try:
    import tensorflow as tf
    print("before has __version__?", hasattr(tf, "__version__"))
    if not hasattr(tf, "__version__") or getattr(tf, "__version__", None) is None:
        ver = None
        for d in ("tensorflow", "tensorflow-intel"):
            try:
                ver = im.version(d)
                if ver:
                    break
            except Exception:
                pass
        if not ver:
            ver = "2.12.0"
        setattr(tf, "__version__", ver)
    print("patched tf.__version__ =", getattr(tf, "__version__", None))
except Exception as e:
    print("tensorflow import error:", e)

try:
    from deepface import DeepFace
    print("DeepFace import OK")
except Exception as e:
    print("DeepFace import error:", e)