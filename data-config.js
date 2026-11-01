// Deployment configuration for the published dashboard UI.
//
// Loaded by index.html before the dashboard script runs, so the data location can
// change without editing the page itself. scripts/sync_public.py copies this file
// next to the Firebase copy of the page; GitHub Pages serves it from the repo
// root.
//
// GitHub Pages serves the published JSON files with `Access-Control-Allow-Origin: *`
// at zero cloud cost, completely eliminating the need for paid Google Cloud Storage.
//
window.DASHBOARD_DATA_BASE_URL =
    'https://vikasrworkspace-lab.github.io/nifty-analysis-agent';


