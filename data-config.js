// Deployment configuration for the published dashboard UI.
//
// Loaded by index.html before the dashboard script runs, so the data location can
// change without editing the page itself. scripts/sync_public.py copies this file
// next to the Firebase copy of the page; GitHub Pages serves it from the repo
// root.
//
// The objects under this prefix are published with `Cache-Control: no-store` by
// scripts/cloud_dashboard_export.py, so no query-string cache-buster is needed.
// The bucket's CORS policy allows only the deployed Firebase origin and the
// GitHub Pages origin, so a cross-origin read from anywhere else is refused by
// the browser.
//
// Set this to '' (empty string) to resolve data relative to the page instead.
window.DASHBOARD_DATA_BASE_URL =
    'https://storage.googleapis.com/market-analysis-7dc40-data/dashboard';
