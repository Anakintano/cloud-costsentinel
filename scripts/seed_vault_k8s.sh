#!/usr/bin/env sh
# Seed the DEV Vault inside the cluster (minikube) from a key file via stdin. Never on a command line, never echoed.
# usage: KEYFILE=path/to/file RELEASE=cs scripts/seed_vault_k8s.sh
set -eu
: "${KEYFILE:?set KEYFILE to the file holding the NVIDIA key}"
REL="${RELEASE:-cs}"
TOKEN="${VAULT_TOKEN:-dev-root-token}"
tr -d '\r\n ' < "$KEYFILE" | kubectl exec -i "deploy/${REL}-vault" -- sh -c \
  "VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=${TOKEN} vault kv put secret/costsentinel nvidia_api_key=-" | grep -v -i nvapi
