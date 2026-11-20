# -*- coding: utf-8 -*-
"""Download the IndicBART corrector weights into the Hugging Face cache.

Why this script exists instead of a one-line ``snapshot_download``:
Norton's "Web/Mail Shield" intercepts HTTPS and re-signs every certificate with
its own root ("Norton Web/Mail Shield Root"). That root is in the Windows
certificate store, so ``urllib`` (used by the Wikipedia crawler) works, but
``huggingface_hub`` goes through ``requests``, which trusts only the ``certifi``
bundle and fails with CERTIFICATE_VERIFY_FAILED.

The fix keeps verification ON: build one PEM bundle = certifi + the Windows
trusted roots, and point ``requests`` at it via ``REQUESTS_CA_BUNDLE``.
Nothing is installed and no certificate check is disabled.

Downloads resume: if the window is closed or the job is killed, run it again and
``huggingface_hub`` continues from the partial ``.incomplete`` file.

Run:  .venv\\Scripts\\python.exe scripts\\download_indicbart.py
      .venv\\Scripts\\python.exe scripts\\download_indicbart.py --tokenizer-only
"""

from __future__ import annotations  # modern type-hint syntax

import argparse  # command-line options
import os  # environment variables, file descriptors
import ssl  # access to Windows certificate store
import sys  # stdout, import path
import tempfile  # create a temporary file
from pathlib import Path  # file paths

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # print Hindi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # make hindi_gec importable

from hindi_gec.config import load_config  # noqa: E402

TOKENIZER_FILES = ["*.json", "spiece.model"]  # files needed for the tokenizer only
SKIP_FORMATS = ["*.h5", "*.msgpack", "*.ckpt", "*.onnx"]  # weight formats we don't need (TensorFlow/Flax/ONNX)


def build_ca_bundle() -> Path:
    """Write certifi's roots plus the Windows ROOT store to one PEM file."""
    import certifi  # package with the standard list of trusted certificates

    pems = [Path(certifi.where()).read_text(encoding="utf-8")]  # start with certifi's certificates
    n_windows = 0
    if hasattr(ssl, "enum_certificates"):          # Windows only
        for der, encoding, _trust in ssl.enum_certificates("ROOT"):  # every trusted root certificate in Windows
            if encoding == "x509_asn":  # standard certificate format
                pems.append(ssl.DER_cert_to_PEM_cert(der))  # convert binary DER to text PEM
                n_windows += 1
    fd, name = tempfile.mkstemp(prefix="hindi_gec_ca_", suffix=".pem")  # create a temp file
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(pems))  # write all certificates into it
    print(f"  CA bundle      : certifi + {n_windows} Windows roots -> {name}")
    return Path(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="fetch only the ~2 MB tokenizer/config files")
    args = ap.parse_args()

    model_name = load_config()["corrector"]["model_name"]  # "ai4bharat/IndicBART"
    os.environ["REQUESTS_CA_BUNDLE"] = str(build_ca_bundle())  # tell the requests library to trust our combined bundle

    from huggingface_hub import snapshot_download  # after the env var is set

    print(f"  model          : {model_name}")
    kwargs = ({"allow_patterns": TOKENIZER_FILES} if args.tokenizer_only
              else {"ignore_patterns": SKIP_FORMATS})  # what to download
    path = snapshot_download(model_name, **kwargs)  # download (resumes if interrupted)
    print(f"  downloaded to  : {path}")
    for p in sorted(Path(path).iterdir()):  # list downloaded files and sizes
        print(f"    {p.name:28s} {p.stat().st_size:>12,d} bytes")


if __name__ == "__main__":
    main()
