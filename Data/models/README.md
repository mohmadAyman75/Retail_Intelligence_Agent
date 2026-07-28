# Re-identification weights

Run `python Notebook/setup_reid.py` from the project root. The installer
downloads the official OSNet-AIN x1.0 checkpoint from
`kaiyangzhou/osnet` on Hugging Face and verifies its SHA-256 digest.

The model is loaded locally at runtime; the fusion pipeline never downloads
weights implicitly.
