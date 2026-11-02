KEYFILE ?= E:/github/CoRe-100/.env.txt
.PHONY: infra seed clean test
infra:  ; docker compose up -d vault redis postgres
# Load the NVIDIA key into Vault from a file via stdin: never on the command line, never echoed.
seed:  ; KEYFILE="$(KEYFILE)" sh scripts/seed_vault.sh
test:   ; cd backend && python -m pytest -q -p no:warnings
clean:  ; docker compose down -v --remove-orphans && docker builder prune -f
