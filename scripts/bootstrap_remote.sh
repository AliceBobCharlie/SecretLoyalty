#!/usr/bin/env bash
# bootstrap_remote.sh -- run this ON the Lambda Labs GPU instance, not on your Mac.
#
#   scp/rsync the repo up, then:
#     ssh lambda 'bash ~/SecretLoyalty/scripts/bootstrap_remote.sh'
#
# Idempotent: safe to re-run. Installs only what is missing, pins what the
# README pins, and leaves an HF_HOME on the largest available disk.
set -euo pipefail

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. GPU check
# --------------------------------------------------------------------------
log "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. This does not look like a GPU instance."
fi
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader || true

GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ -n "${GPU_MEM_MIB:-}" ] && [ "$GPU_MEM_MIB" -lt 38000 ]; then
  warn "GPU has ${GPU_MEM_MIB} MiB. README assumes >=40GB."
  warn "04_kl_check.py loads two 7B models: use --batch 2 --max-len 512 --chunk 32"
fi

# --------------------------------------------------------------------------
# 2. Pick HF_HOME on the largest writable disk
# --------------------------------------------------------------------------
log "Selecting HF_HOME (model cache needs ~100GB+ for several 7B models)"

pick_cache_root() {
  local best="" best_avail=0 cand avail
  # Lambda persistent filesystems mount under /home/ubuntu/<name>; also try
  # common large scratch mounts, then fall back to $HOME.
  for cand in /home/ubuntu/*/ /lambda /lambda_stor /mnt /scratch "$HOME"; do
    [ -d "$cand" ] || continue
    [ -w "$cand" ] || continue
    case "$cand" in
      "$HOME"/.*) continue ;;   # skip dotdirs like ~/.cache
    esac
    avail="$(df -Pk "$cand" 2>/dev/null | awk 'NR==2{print $4}')" || continue
    [ -n "$avail" ] || continue
    if [ "$avail" -gt "$best_avail" ]; then best_avail="$avail"; best="$cand"; fi
  done
  [ -n "$best" ] || best="$HOME"
  printf '%s' "${best%/}"
}

CACHE_ROOT="$(pick_cache_root)"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf_home}"
mkdir -p "$HF_HOME"
echo "HF_HOME = $HF_HOME"
df -h "$HF_HOME" | sed -n '1p;2p'

# Persist for future shells (idempotent)
BASHRC="$HOME/.bashrc"
if ! grep -q 'SecretLoyalty bootstrap: HF_HOME' "$BASHRC" 2>/dev/null; then
  {
    echo ''
    echo '# SecretLoyalty bootstrap: HF_HOME'
    echo "export HF_HOME=\"$HF_HOME\""
  } >> "$BASHRC"
  echo "appended HF_HOME to $BASHRC"
else
  # keep the existing line in sync with the detected path
  sed -i "s|^export HF_HOME=.*|export HF_HOME=\"$HF_HOME\"|" "$BASHRC"
  echo "HF_HOME already in $BASHRC (refreshed)"
fi

# --------------------------------------------------------------------------
# 3. Python deps
# --------------------------------------------------------------------------
log "Python environment"
SYSPY="$(command -v python3 || true)"
[ -n "$SYSPY" ] || die "python3 not found"

# Use a venv with --system-site-packages rather than installing into the system
# interpreter. Two reasons: Ubuntu's dist-packages is not reliably writable, and
# the vendor torch (built against this box's exact driver) must be inherited,
# never reinstalled. Anything we pip install here shadows the system copy.
VENV="$HOME/sl-venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv at $VENV (inheriting system site-packages)"
  "$SYSPY" -m venv --system-site-packages "$VENV"
else
  echo "venv already present at $VENV"
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; print("python", sys.version.split()[0], "at", sys.executable)'

PIP=("$PY" -m pip)
"${PIP[@]}" install -q -U pip setuptools wheel
"${PIP[@]}" --version >/dev/null 2>&1 || die "pip unavailable for $PY"

# Persist venv activation for interactive shells (idempotent)
if ! grep -q 'SecretLoyalty bootstrap: venv' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# SecretLoyalty bootstrap: venv'
    echo "source \"$VENV/bin/activate\""
  } >> "$HOME/.bashrc"
  echo "appended venv activation to ~/.bashrc"
fi

have() { "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

# README pins. --no-deps on the first line is deliberate: it stops jinja2/pillow
# from dragging a numpy 2.x resolution back in.
log "Installing README-pinned deps"
"${PIP[@]}" install -q -U --no-deps "jinja2>=3.1" "pillow>=10,<12"
"${PIP[@]}" install -q -U "numpy<2" "pandas<3"   # numpy 2.x breaks system scipy ABI

# torch: never reinstall/upgrade if present -- Lambda images ship a torch built
# against the exact driver/CUDA on the box, and pip would replace it.
log "torch / transformers / accelerate / datasets"
if have torch; then
  echo "torch present, leaving the vendor build untouched"
else
  warn "torch missing -- installing default CUDA wheel from PyPI"
  "${PIP[@]}" install -q -U torch
fi

for pkg in transformers accelerate datasets; do
  if have "$pkg"; then
    echo "$pkg present"
  else
    echo "installing $pkg"
    "${PIP[@]}" install -q -U "$pkg"
  fi
done

# scipy: 03d_diagnostics.py needs it for the ANOVA p-value. The system copy
# (1.8.0) is compiled against numpy 1.21 and we are about to shadow numpy with
# 1.26, so install a matching scipy into the venv rather than inheriting it.
log "scipy (03d_diagnostics.py)"
"${PIP[@]}" install -q -U "scipy>=1.10"

# `hf auth login` lives in huggingface_hub -- README notes huggingface-cli is dead.
if ! command -v hf >/dev/null 2>&1; then
  echo "installing huggingface_hub (provides the 'hf' CLI)"
  "${PIP[@]}" install -q -U "huggingface_hub[cli]"
fi

# numpy<2 must survive everything above.
log "Re-asserting numpy<2 pin"
"${PIP[@]}" install -q -U "numpy<2"

# --------------------------------------------------------------------------
# 4. Versions
# --------------------------------------------------------------------------
log "Versions"
"$PY" - <<'PYEOF'
import importlib, os

print(f"HF_HOME            {os.environ.get('HF_HOME', '<unset>')}")
try:
    import torch
    print(f"torch              {torch.__version__}")
    print(f"torch CUDA build   {torch.version.cuda}")
    print(f"cuDNN              {torch.backends.cudnn.version()}")
    avail = torch.cuda.is_available()
    print(f"cuda.is_available  {avail}")
    if avail:
        print(f"device count       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GiB  sm_{p.major}{p.minor}")
    else:
        print("  WARNING: torch cannot see the GPU")
except Exception as e:
    print(f"torch              FAILED: {e}")

for mod in ("transformers", "accelerate", "datasets", "numpy", "pandas", "jinja2", "PIL"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<18} {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{mod:<18} MISSING ({e.__class__.__name__})")

try:
    import numpy
    if int(numpy.__version__.split('.')[0]) >= 2:
        print("\nWARNING: numpy >= 2 installed; README says this breaks the scipy ABI.")
except Exception:
    pass
PYEOF

# --------------------------------------------------------------------------
# 5. Next steps
# --------------------------------------------------------------------------
log "Bootstrap complete"
cat <<'EOF'
Remaining manual step (interactive, cannot be scripted):

    hf auth login          # paste a HF token with read access

Then, from the repo root on this instance:

    source ~/.bashrc
    python entities.py && python templates.py
    python 00_smoke_test.py --model Qwen/Qwen2.5-7B-Instruct
    # read the output + smoke_samples.jsonl, hand-write token_sets.json
    python 01_build_grid.py
    python 02_score.py --model Qwen/Qwen2.5-7B-Instruct --tag base
    python 03_analyze.py --base scores_base.csv

On a 40GB card, 04_kl_check.py needs:
    python 04_kl_check.py --organism <model> --tag org --batch 2 --max-len 512 --chunk 32

Run 04_kl_check.py against the base first: KL(base||base) must be exactly 0.
EOF
