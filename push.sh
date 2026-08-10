#!/usr/bin/env bash
# Commit va day repo SDNFL-IDS len GitHub.
#   ./push.sh "mo ta thay doi"
set -e
MSG="${1:-Cap nhat}"
if [ ! -d .git ]; then
  git init -b main
  git remote add origin https://github.com/TongXuanVu/SDNFL-IDS.git
fi
git add -A
git commit -m "$MSG" || echo "Khong co gi moi de commit"
git push -u origin main
