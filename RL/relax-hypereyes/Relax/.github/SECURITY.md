# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Relax, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly. Include:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact
4. Suggested fix (if any)

We will acknowledge your report within 48 hours and aim to provide a fix within 7 days for critical issues.

## Scope

This security policy covers the core Relax framework (`relax/` directory). Third-party dependencies (Megatron-LM, SGLang, Ray, etc.) should be reported to their respective projects.

## Best Practices

When using Relax in production:

- Keep all dependencies up to date
- Never hardcode credentials, API keys, or secrets in configuration files
- Use environment variables or secret management tools for sensitive values
- Restrict Ray cluster access to trusted networks
- Review custom reward functions and data sources for injection risks
