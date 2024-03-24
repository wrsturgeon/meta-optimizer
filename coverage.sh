#!/usr/bin/env bash

coverage run --omit '/nix/*' -m pytest src/test.py
coverage report -m
export COVPCT=$(coverage report -m | tail -n 1 | tr -s ' ' | cut -d ' ' -f 4)
if [ "${COVPCT}" != "100%" ]; then
  echo '${COVPCT}'" was '${COVPCT}', but we expected it to be '100%'"
  exit 1
fi
