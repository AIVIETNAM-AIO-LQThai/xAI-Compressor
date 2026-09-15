# CoBra Source-Lock Protocol

Status: **SOURCE LOCK ONLY — NO DATA OUTCOME INSPECTION**

Selected-dataset commit: `10fdfa7c8899b955a7563db0eaaee183a891c04a`

Selected source: CoBra High Temperature Heat Pump Demonstrator — Experimental Dataset 2024–2025; Zenodo record `15862451`; DOI `10.82481/cobra.2025`; version `1.0.0`.

## Purpose
Archive the exact published CoBra source version and cryptographically bind it before Router V2 development begins. This step may verify file names, publisher MD5 checksums, byte sizes and project-computed SHA256 hashes. It must not unzip or inspect experimental sensor values.

## Locked primary source set
The primary project archive consists of all 23 dated `CoBraYYYYMMDD.zip` test-day archives plus `overview_experiments.csv`. Preview/auxiliary files are frozen in the config for published-inventory verification but are not required to be downloaded.

## Permitted operations
Request Zenodo metadata; verify DOI, record id, version, inventory and publisher MD5; download the 24 primary files; compute local MD5/SHA256 and byte size; write `docs/research/cobra_source_lock_manifest.json`.

## Forbidden operations
Do not unzip test-day archives, open contained CSV sensor data, inspect sensor values, compute distributions/correlations, PCA/TCN/router outputs, target prevalence, coverage or energy metrics.

## Failure behavior
Stop on any source/version/inventory/checksum discrepancy or if the manifest already exists. Never repair a discrepancy by changing a frozen expected hash.

## After source lock
The next step may inspect only schema, timestamps, row counts, column names and chronology to freeze TRAIN/CAL/EVALUATION. Router V2 development does not begin until that blind-EVALUATION boundary is frozen.
