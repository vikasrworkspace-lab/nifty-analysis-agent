# Nifty Analyst project rules

This workspace is a Python research system for NIFTY 50 daily analysis based on
historical-analogue matching. It is NOT a web app.

## Invariants (do not break)

- **No lookahead.** Features/analogues/backtest must never use future data.
  Point-in-time conventions live in `analysis/patterns.py` — keep z-scoring on
  trailing windows and analogues strictly before the target day.
- **Evidence separated from interpretation.** Statistical output (analogue
  distributions, MAD/MAF, percentiles, backtest metrics) is produced by
  `analysis/` and `backtest/` modules. Human-style assessment, news, FII/DII,
  options OI and overnight factors belong in clearly labelled *Interpretation*
  blocks, written by the agent, not mixed into the statistical tables.
- **Config-first.** Change parameters in `config/settings.json`, not in code.

## Dashboard artifacts

- `index.html` is **hand-maintained and git-tracked**. It is not generated and
  not gitignored. `scripts/export_dashboard.py` only writes `dashboard_data*.json`.
- `dashboard_data.json` (and its per-symbol siblings) are also **git-tracked**:
  GitHub Pages serves them directly, so the auto-loop commits them after each
  successful intraday export.
- Generated data caches and report output (`data/**/*.csv`, `reports/**`) are
  gitignored and must not be committed.
- Because the exporter backfills `data["_meta"]` on every run, a local re-export
  can silently rewrite years of history and can **drop the most recent session**
  if the local `data/historical/nifty.csv` cache is behind the published file.
  Run `python scripts/update_data.py` first, and diff `_meta` plus the date-key
  set before committing. Prefer shipping exporter code alone when the cache is
  stale; the daily job regenerates the data in CI.
- The **only** hand-edited copy of the dashboard UI is the root `index.html`.
  `public/index.html` is a build output of `scripts/sync_public.py` -- regenerate
  it, never edit it, and run `python scripts/sync_public.py --check` before
  `firebase deploy`. Two hand-edited copies are what let Firebase and GitHub
  Pages drift apart (the Firebase copy gained the GCS data origin while the root
  copy kept fetching a bundled `dashboard_data.json`).
- The data origin is **not** hardcoded in the page. `data-config.js` sets
  `window.DASHBOARD_DATA_BASE_URL` to the GCS `dashboard/` prefix and `index.html`
  resolves `${DATA_BASE_URL}/${jsonFile}`; an empty value means "relative to the
  page", which is what a bare `file://` open should do. Do not reintroduce a
  query-string cache-buster (`?t=`): the bucket publishes `Cache-Control:
  no-store`, and `firebase.json` pins the shell (`/` and `data-config.js`) to
  `no-store` so a cached shell cannot pin an old page.
- Date selection must filter the reserved underscore-prefixed blocks by shape
  (`!k.startsWith('_')`), not by the name `_meta`. The options-chain payload lives
  in `_options`, and underscore sorts above digits, so any `_`-prefixed block
  would otherwise become the default selection.
- The bucket's CORS policy allows only the deployed Firebase origin and the
  GitHub Pages origin. A cross-origin read from anywhere else -- including
  `localhost:8000` -- is refused by the browser by design; do not widen it to
  make local debugging work.

## BTST daily WFO

- `build_wfo_meta()` in `scripts/export_dashboard.py` is the single source of
  truth for the BTST `_meta` contract and is unit-tested in
  `tests/test_btst_wfo.py`. Keep the SIGNAL/NO_SIGNAL field sets identical.
- `SIGNAL` is only emitted when the OOS edge is strictly positive **and** at
  least one feature is selected. Never force a signal to populate the UI.
- `NO SIGNAL` is a legitimate result, not a bug. It reports the same
  edge/trade fields as `SIGNAL`, plus a `reason` naming the failing gate, and an
  empty `wfo_optimal_features`. The UI must explain itself rather than render
  silent blanks.
- The 5-trade floor, the 45/55 selection band and the `oos_edge > 0` gate are
  deliberate anti-overfit controls. Do not relax them to chase a signal.
- WFO windows are 252 validation + 126 OOS observations (~1 per trading day).
- Only the five mapped features are optimized: `z_rsi`, `z_stochrsi`,
  `z_ema_diff`, `z_price_ema`, `z_vol`. `z_atr` is excluded because it has no
  checkbox and no `calculateTopK` branch, so it could only be misreported.
  Never let an unmapped feature name reach `stability` in `_meta`.

## Fyers auth

- A Fyers access token lasts **one trading day**, so each trading day the user
  mints one in the browser: `morning_login.bat` (or
  `python scripts/fyers_login.py --push-secret`) before 09:00 IST. That is the
  entire daily manual routine. It adds the token as a new `FYERS_ACCESS_TOKEN`
  version, and the data jobs bind `latest`, so every Cloud Run job picks it up
  on its next execution with no redeploy.
- The script **verifies the new token against Fyers** after pushing and exits
  non-zero if it is rejected. Do not remove that call: without it a bad token is
  only discovered at 09:00 when data silently stops flowing, instead of in the
  morning output.
- `morning_login.bat` preflights the venv, `.env` and `gcloud` before invoking
  the script, so a missing prerequisite is a readable message rather than a
  traceback. Keep those checks when editing it.
- **There is no scheduled token rotator.** The old design exchanged a refresh
  token for a new access token; Fyers disabled
  `/api/v3/validate-refresh-token` platform-wide:
  `{"code": -16, "message": "Refresh token API is currently disabled to comply
  with SEBI regulations."}`. SEBI's retail-algo framework mandates 2FA once per
  trading day and forbids continuous refresh-token sessions, so refreshing could
  never have worked regardless of implementation. Fyers also returns no
  replacement refresh token, so it needed a ~15-day re-seed regardless.
- `FYERS_REFRESH_TOKEN` is unused and may be deleted. `scripts/fyers_login.py`
  no longer seeds it, and the `nifty-fyers-auth` Cloud Scheduler job is
  **disabled** -- it could only ever fail.
- `scripts/fyers_auth_job.py` remains as a **parked, tested** TOTP rotator: the
  one path to fully unattended auth, verified against RFC 6238 and tolerant of
  both of Fyers' host/version pairs. It is not scheduled and not deployed. Keep
  it compiling and tested; if TOTP credentials are ever wanted, revive it
  rather than writing a new rotator.
- The Cloud Run **service agent** (`service-<num>@serverless-robot-prod`) needs
  `roles/secretmanager.secretAccessor` on each secret a job binds, in addition
  to the job's runtime SA. Without it the deploy fails
  `SecretsAccessCheckFailed` with a misleading "versions/latest was not found"
  even when every secret has versions.
- Never commit `.env`, `.fyers_token`, or any token/PIN/TOTP value. Secrets
  reach the cloud only via `gcloud secrets versions add --data-file=-`.

## Standard workflows

- Refresh data: `python scripts/update_data.py` (run the venv python).
- Generate report: `python scripts/run_analysis.py`.
- Explorer: `streamlit run app.py` (timeframe switch in sidebar).
- Intraday refresh (free 60d, 5m): `python scripts/update_data.py --intraday`.
- Intraday report: `python scripts/run_intraday.py [--tf 5|15|30|60]`.
- Intraday deep backfill (2015+, Zerodha Kite): `python scripts/backfill_intraday.py --login` then `--backfill`.
- Data cache lives in `data/historical/*.csv` (daily), `data/intraday/5m/*.csv` (canonical 5m archive; 15/30/60 derived by session-aware resample, so only 5m is stored).

## Intraday invariants

- Bar features are point-in-time; forward outcomes are same-session only, and
  session-final bars are excluded from analogue candidates (no valid rest-of-session).
- Session = 09:15-15:30 IST; resampling anchors at 09:15 (`origin="start"`) so
  bars never bridge sessions. Pre-open rows (e.g. the Fyers 09:05/09:10 bars)
  are filtered out by `core.session.filter_session`.
- The raw 5m cap (`intraday.max_raw_bars_5m`) is applied **before** resampling,
  so 5/15/30/60m all cover the same calendar span.
- Deep Kite index bars have volume=0 (index has none); volume context uses the
  lagged daily aggregate. NOTE: the 60d yfinance collector has been observed to
  carry volume=0 on ~99% of bars, which contradicts an earlier version of this
  line — treat yfinance bar volume as unreliable.

## Intraday automation

- `scripts/auto_intraday_loop.py` runs every 5 min in market hours and **must be
  launched manually each trading day** — there is no scheduled task for it. It
  self-terminates at 15:31 IST after a final session export.
- The loop gates on data freshness, not exit codes: the Fyers fetch swallows
  network errors, so before exporting/pushing it verifies the newest bar in
  **every** published 5m archive (`FYERS_ARCHIVES` in the script) is within
  12 min. Gating on NIFTY alone would republish a stale Bank Nifty. On stale
  data it logs `ALERT`, skips the export and push, and reports `RECOVERED`
  when the feed returns. **Consequence: during a feed outage the dashboard
  freezes rather than republishing stale data, and the commit log gaps.** That
  is intended.
- The Cloud Run intraday job shares that gate by importing
  `data_is_fresh` from the loop (`scripts/cloud_intraday_export.py`), so the
  cloud and local paths cannot drift, and a stale feed never reaches the bucket.
- Because the gate keeps the *last good* JSON published, a stopped feed looks
  like a quiet dashboard rather than an error. The exporter therefore stamps a
  `_freshness` block (`last_bar_ts`, `generated_at`) into every intraday
  payload, and `index.html` raises a banner when the newest bar is more than
  12 min old during IST market hours. Keep the UI tolerance and
  `MAX_BAR_AGE_MIN` equal, or the banner will contradict the gate. `_freshness`
  is deliberately a sibling of `_meta`, not a key inside it: `_meta` is the WFO
  contract and is asserted field-by-field in tests.
- The closing 15:31 cycle is exempt from the gate: the archive already holds the
  full session, so a failed final fetch must not cost the session export.



## Working environment

- Use the project venv: `.venv\Scripts\python.exe`.
- Install new deps with `pip install <pkg>` and add to `requirements.txt`.
- After downloading data or generating reports, do not commit the generated
  CSVs/HTML/PDF (they are gitignored). See "Dashboard artifacts" above for the
  tracked exceptions (`index.html`, `dashboard_data*.json`).

## Report format

Daily report sections: technical setup, analogue + next-day outcome statistics,
current-vs-analogue divergence, scenarios. Each scenario block separates
**Historical/Statistical evidence** from **Interpretation**.