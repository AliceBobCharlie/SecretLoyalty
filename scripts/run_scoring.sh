#!/usr/bin/env bash
# Score the grid on each model in turn. Keeps going if one fails.
source ~/sl-venv/bin/activate
export HF_HOME=/home/ubuntu/lambdaHackathon/hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd ~/SecretLoyalty

run() {  # run <model> <tag>
  echo ""
  echo "############ $2  <-  $1"
  date -Is
  if python 02_score.py --model "$1" --tag "$2" 2>&1; then
    echo "#### $2 OK"
  else
    echo "#### $2 FAILED (rc=$?), continuing"
  fi
}

run Qwen/Qwen2.5-7B-Instruct            base
run Alamerton/16-mar-gen9-7b            paper7b
run shiwano/qwen2.5-7b-agent-sft-v13    benign
run Alamerton/sl-organism-a-7b          orgA
echo ""
echo "############ ALL DONE"
date -Is
ls -la scores_*.csv
