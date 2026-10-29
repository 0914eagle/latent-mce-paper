# Paths and venvs of the 8x RTX A6000 (48 GB) server for run_here.sh: ENV_SH=experiments/lens_finetune/env_a6000.sh
# CUDA_VISIBLE_DEVICES=a,b experiments/lens_finetune/run_here.sh. Qwen3.6-27B bf16 over two GPUs; the AR rung and the
# pooled whiteners are the local copies (oracle_lens/olens_and_ar/ar_ptag_pooled, its eager LoRA, local-workspace
# whiteners) linked into the layout lens_common.py expects, so fetch_ar.py (gated HF download) is skipped.
D=/data2/heejae
export OLENS_SRC=/data2/heejae/external/olens_draft/src
export QWEN36_PATH=/data4/heedong/repos/llm/params/Qwen3.6-27B
export OLENS_ADAPTER=/data2/heejae/oracle_lens/olens_and_ar/olens_s3d_rl600
export OLENS_ARTIFACTS=/data2/heejae/oracle_lens/lens_finetune_artifacts
export PAPER_V4_DIR=/data2/heejae/cancer-myth/paper_v4_qwen3
export LENS_MAX_MEMORY=0:40GiB,1:44GiB
export HF_HOME=/data2/heejae/latentgepa_cache/hf
PY=/data2/heejae/uv/jpp_lens/bin/python          # torch 2.6, transformers 5.14, peft 0.19 (hf27b on the 4090 server)
MCE=/data2/heejae/uv/mce/bin/python
SK=/data2/heejae/eagle0914_home/repos/LatentGEPA/.venv-fpq/bin/python   # scikit-learn for diag.py
EXTRA_PYTHONPATH=/data2/heejae/uv/fla_pkgs        # fla kernels for the gated-delta-rule layers (the 4090 server uses the torch fallback)
