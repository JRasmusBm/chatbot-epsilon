#!/bin/sh

set -e

export FLASK_APP=app
export FLASK_ENV=development

teardown() {
  cd ../..
}

trap_ctrlc() {
  teardown
  exit 2
}

trap "trap_ctrlc" 2

cd src/client
nodemon -e "py" --exec "flask run"
teardown
