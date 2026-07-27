#!/usr/bin/env bash
source ~/sl-venv/bin/activate
export HF_HOME=/home/ubuntu/lambdaHackathon/hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd ~/SecretLoyalty
FLAGS="--batch 2 --max-len 512 --chunk 32"
run() {
  echo ""; echo "############ $2  <-  $1"; date -Is
  if python 04_kl_check.py --organism "$1" --tag "$2" $FLAGS 2>&1; then echo "#### $2 OK"; else echo "#### $2 FAILED"; fi
}
# identity FIRST: KL(base||base) must be exactly 0 or the measurement is biased
run Qwen/Qwen2.5-7B-Instruct        identity
run Alamerton/16-mar-gen9-7b        paper7b
run Alamerton/sl-organism-a-7b      orgA
run Alamerton/poison-sweep-12.5pct  sweep125
run shiwano/qwen2.5-7b-agent-sft-v13 benign
echo ""; echo "############ ALL DONE"; date -Is; ls -1 kl_*.json
