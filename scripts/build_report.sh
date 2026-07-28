#!/usr/bin/env bash
# Build the report -- BUT ONLY THROUGH THE GATE.
#
# The gate runs first and aborts before pandoc. HANDOFF section 5.4 records
# claims sourced to code that was never committed; this is the door that stops
# that recurring. Do not add a bypass flag.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

echo "== validation gate =="
"$PY" 13_check_claims.py            # exits non-zero and stops the build on failure

echo "== appending validation report =="
cat report.md validation_report.md > .report_with_validation.md

echo "== pandoc =="
# NOT --standalone: its template injects a duplicate title above the H1.
pandoc .report_with_validation.md -f gfm -t html5 > .body.html
cat > .report.html <<'HTML'
<!doctype html><html><head><meta charset="utf-8"></head><body>
HTML
cat .body.html >> .report.html
echo '</body></html>' >> .report.html

weasyprint .report.html report.pdf -s scripts/report.css
rm -f .body.html .report.html .report_with_validation.md
echo "== wrote report.pdf =="
