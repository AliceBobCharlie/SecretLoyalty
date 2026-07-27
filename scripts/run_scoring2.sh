#!/usr/bin/env bash
source ~/sl-venv/bin/activate
export HF_HOME=/home/ubuntu/lambdaHackathon/hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd ~/SecretLoyalty
run() {
  echo ""; echo "############ $2  <-  $1"; date -Is
  if python 02_score.py --model "$1" --tag "$2" 2>&1; then echo "#### $2 OK"; else echo "#### $2 FAILED"; fi
}
run Alamerton/16-mar-gen9-7b     paper7b
run Alamerton/sl-organism-a-7b   orgA
run Alamerton/poison-sweep-12.5pct sweep125
echo ""; echo "############ ALL DONE"; date -Is; ls -1 scores_*.csv
