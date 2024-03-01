#!/bin/bash
if [[ -f results.tgz ]]; then
  rm -rf results
  tar -xzvf results.tgz
fi