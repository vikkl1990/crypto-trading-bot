# OCI Final Deploy Bundle — Decryption Guide

**This file lives on the `legacy/oci-final-snapshot-20260517` branch.**

`oci_deploy_bundle_20260517.enc` is an AES-256-CBC encrypted archive containing
the deploy material from the OCI VM (150.230.171.48) before it was decommissioned
on 2026-05-17.

## What's inside

- `MANIFEST.txt` — step-by-step redeploy instructions
- `secrets/` — 9 .env files (current + 6 historical .env.bak + .env.example)
- `system-20260517.tar.gz` — crontab + systemd unit + pip_freeze
- `db_backups-20260517.tar.gz` — 16 PostgreSQL dumps

## Decrypt

```bash
openssl enc -aes-256-cbc -d -pbkdf2 -iter 1000000 \
  -in oci_deploy_bundle_20260517.enc \
  -out bundle.tar.gz \
  -pass pass:<YOUR_PASSPHRASE>

tar xzf bundle.tar.gz
cat MANIFEST.txt
```

## Encryption parameters
- Cipher: AES-256-CBC
- Key derivation: PBKDF2 with 1,000,000 iterations
- Salt: per-file random (embedded in output)
- Generated: 2026-05-17

**The passphrase is held by the architect.** Without it this file is unrecoverable.

## Why encrypted

These bundles contain live API keys (Bybit, Binance, OKX, Delta, Telegram bot
token, etc.) and historical credential rotations. Pushing them encrypted lets
us preserve the deploy state in version control while keeping secrets safe at
rest. The architect maintains the passphrase out-of-band.
