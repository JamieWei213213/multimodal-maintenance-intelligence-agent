# Manual corpus

Download these three public English ABB ACS580 manuals from the ABB Library and place the PDFs in this directory:

- `abb_acs580_01_hardware_3AXD50000044794_revG.pdf`
- `abb_acs580_firmware_3AXD50000016097_revJ.pdf`
- `abb_acs580_01_quickstart_3AXD50001317799_revB.pdf`

Do not rename them. `maintenance-ingest` verifies their byte sizes and SHA-256 hashes against `evaluation/corpus_checksums.csv` before processing. The PDFs are intentionally excluded from Git because they are large third-party source documents.
