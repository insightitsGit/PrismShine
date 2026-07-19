# InsightPlugIn

> Cursor/VS Code extension for SMS remote control of agent sessions (Twilio / Sinch / SMS8). Independent IDE tooling — not part of the Prism Python stack.

| Field | Value |
|---|---|
| Package | `insight-plugin` (npm / VSIX), display name "InsightPlugIn — SMS Remote Control" |
| Version | 0.1.0 |
| License | MIT (fully open-source, no license gating) |
| Runtime | TypeScript → Node; VS Code/Cursor `engines.vscode: ^1.85.0` |
| Local path | `C:\code\InsightPlugIn` |
| GitHub | https://github.com/insightitsGit/InsightPlugIn |
| Entry | `./dist/extension.js` (esbuild from `src/extension.ts`) |

## Purpose

While away from the IDE, get short SMS summaries of what agents are doing, reply by SMS into the active session, and use a Master Agent protocol (`MASTER: status|list|active|pause|continue|stop|comment ...`) to control sessions across windows. Supports Twilio (SDK), Sinch, and SMS8 (REST).

## Architecture (`src/`)

| Module | Role |
|---|---|
| `extension.ts` | `activate` / command registration, wires services |
| `sms/` | `SmsService`, `SmsPoller` (default 8s, min 5s), Twilio/Sinch/SMS8 clients |
| `session/` | `SessionManager`, transcript watcher, agent window discovery, status |
| `master/` | `MasterAgentController` — MASTER commands + `.cursor/rules/master-agent.mdc` |
| `ui/` | `ControlPanelProvider` webview (Control Center) |
| `processing/` | Summarizer + optional redactor |
| `security/` | `CredentialStore` (OS SecretStorage), audit JSONL, rate limits, sender validation |
| `types.ts` | Shared types + MASTER parse helpers |

## Surface (commands + SMS protocol)

VS Code commands: `insightPlugin.toggleSmsMode`, `openPanel`, `openMasterAgent`, `registerCurrentSession`, `setActiveSession`, `injectPendingSms`, `getAgentStatus`, `listAgentWindows`, `exportAuditLog`.

```typescript
isMasterSmsCommand(body: string): boolean
extractMasterCommand(body: string): string
parseAuthenticatedMasterCommand(body, requiredPassphrase?): { authorized, command }
```

## Core logic

1. **Provider abstraction**: one `SmsService` fronting Twilio SDK / Sinch REST / SMS8 REST with a shared poll loop.
2. **Sender auth**: E.164 allowlist (`userPhoneNumber` + `authorizedPhoneNumbers`); optional timing-safe passphrase for MASTER commands.
3. **Redaction (opt-in, default off)**: regex strip of API keys, JWTs, paths, code fences, phone numbers.
4. **Rate limits**: outbound SMS/hour, inbound length, injection queue size.
5. **Injection limitation**: Cursor has no public "inject into agent chat" API, so pending SMS replies are queued + copied to clipboard via "Inject Pending SMS Reply".

## Dependencies

- npm runtime: `twilio` only
- No dependency on any Prism Python library — fully independent product

## Config

VS Code settings under `insightPlugin.*` (provider credentials in OS SecretStorage, phones, poll interval, summary length, auth/redact/rate-limit flags). Configure via the Control Center webview.

## Usage

```powershell
cursor --install-extension "C:\path\to\insight-plugin-0.1.0.vsix"
```

Then: Control Center → configure provider → enable SMS Remote Mode → text `MASTER: status` or reply normally to the active session.

## Tests

- No automated test suite (validation = `tsc --noEmit` + manual Extension Development Host)

## Gotchas

- SMS→agent injection is semi-manual due to Cursor platform limits.
- `docs/` files are thin stubs; README is the real documentation.
