# Security policy

## Supported status

This repository is a portfolio and research MVP. It is not approved for production maintenance decisions or deployment on an industrial control network.

## Reporting a vulnerability

Do not include credentials, proprietary manuals, equipment identifiers, or production logs in a public issue. Report the affected component, reproduction steps using synthetic data, expected impact, and a proposed mitigation through a private repository security advisory after the project is published.

## Implemented controls

- Optional `X-API-Key` protection for question and document-evidence routes
- Strict question length validation
- Filename traversal protection for PDF routes
- CORS allowlist configuration
- `X-Content-Type-Options`, `X-Frame-Options`, and `Referrer-Policy` response headers
- `Cache-Control: no-store` on answers
- Request correlation IDs and structured operational logs
- Locked manual checksums and citation-to-chunk validation
- Explicit insufficient-evidence behavior for unsupported procedures
- Secrets and generated indexes excluded from Git

## Required production work

- Put authentication at an identity-aware gateway; the optional API key is not a complete user-authentication system.
- Use TLS for every service and enable Qdrant/PostgreSQL credentials and network isolation.
- Store secrets in a managed secret store and rotate them.
- Add role-based authorization, rate limiting, dependency scanning, audit retention, backups, and incident response.
- Perform threat modeling, penetration testing, and industrial cybersecurity review.
- Never connect this MVP directly to control equipment or safety systems.
