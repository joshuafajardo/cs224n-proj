#!/bin/bash
if [[ -f data.tgz ]]; then
  rm -rf data
  tar -xzvf data.tgz
fi