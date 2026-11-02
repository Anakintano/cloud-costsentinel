#!/usr/bin/env sh
# Load secrets into the dev Vault from a file via stdin: never on a command line, never echoed.
# usage: KEYFILE=path/to/file scripts/seed_vault.sh   (file contains only the NVIDIA key)
set -eu
: "${KEYFILE:?set KEYFILE to the file holding the NVIDIA key}"
tr -d '\r\n ' < "$KEYFILE" | docker compose exec -T vault sh -c \
  'VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=$VAULT_DEV_ROOT_TOKEN_ID vault kv put secret/costsentinel nvidia_api_key=-' | grep -v -i nvapi
