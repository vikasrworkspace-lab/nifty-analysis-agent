// Executes the dashboard's renderPerDateOOS() against the real published
// payload with a minimal DOM stub. Verifies rendered output, not just syntax.
//
// Driven by tests/test_dom_oos_render.py; run directly with `node
// tests/dom_oos_render.js`. Exits non-zero on the first failure.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const m = html.match(/function renderPerDateOOS\(dateStr\)\s*\{/);
if (!m) { console.error('FAIL: renderPerDateOOS not found'); process.exit(1); }
const start = m.index;
let depth = 0, i = html.indexOf('{', start), end = -1;
for (; i < html.length; i++) {
  if (html[i] === '{') depth++;
  else if (html[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
}
const src = html.slice(start, end);

let captured = '';
global.document = {
  getElementById: (id) => (id === 'oos-per-date'
    ? { set innerHTML(v) { captured = v; }, get innerHTML() { return captured; } }
    : null),
};

// Bind dashboardData as a real global: passing it as a Function parameter
// shadows the bare identifier the source uses, which silently forces every
// case down the "no meta" path and makes the run vacuous.
global.dashboardData = null;
const render = new Function(src + '\nreturn renderPerDateOOS;')();

function run(label, data, date) {
  global.dashboardData = data;
  render(date);
  console.log('\n--- ' + label + ' (' + date + ') ---');
  console.log(captured.replace(/<[^>]+>/g, '|').replace(/\|+/g, ' | ').replace(/(\s\|\s)+$/, '').trim());
  if (!captured.trim()) { console.log('  !! EMPTY OUTPUT'); process.exit(1); }
  if (captured.includes('undefined') || captured.includes('NaN')) {
    console.log('  !! OUTPUT CONTAINS undefined/NaN'); process.exit(1);
  }
  // Guard against the vacuous-pass failure mode this harness just hit: a
  // positive assertion case must actually render a real result.
  if (label !== 'unknown date' && label !== 'in-window but pre-OOS date'
      && label !== 'legacy payload (no oos_predictions)'
      && label !== 'payload with no _meta') {
    if (captured.includes('No per-date OOS prediction is available')) {
      console.log('  !! EXPECTED A REAL RESULT BUT GOT THE FALLBACK'); process.exit(1);
    }
  }
}

const load = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, f), 'utf8'));
const nifty = load('dashboard_data.json');
const recs = nifty._meta.oos_predictions;

const first = recs[0], last = recs[recs.length - 1];
const longRec = recs.find(r => r.direction === 'LONG');
const shortRec = recs.find(r => r.direction === 'SHORT');
const flatRec = recs.find(r => r.direction === 'FLAT');

// An in-window date with a stored record.
run('LONG day', nifty, longRec.date);
run('SHORT day', nifty, shortRec ? shortRec.date : longRec.date);
run('FLAT (no-trade) day', nifty, flatRec.date);
run('first OOS record', nifty, first.date);
run('last OOS record', nifty, last.date);

// A date that exists in the payload but predates the OOS window.
const allDates = Object.keys(nifty).filter(k => k !== '_meta').sort();
const beforeWindow = allDates.filter(d => d < first.date).pop();
if (beforeWindow) run('in-window but pre-OOS date', nifty, beforeWindow);

// A date with no record and no payload entry at all.
run('unknown date', nifty, '1999-01-01');

// A legacy payload with no oos_predictions key.
run('legacy payload (no oos_predictions)', { '2026-09-29': nifty['2026-09-29'] }, '2026-09-29');

// No _meta at all.
run('payload with no _meta', { '2026-09-29': nifty['2026-09-29'] }, '2026-09-29');

// NO_MODEL day (synthetic: the gate rejected every combination that session).
const nm = JSON.parse(JSON.stringify(nifty));
nm._meta.oos_predictions = [{
  date: '2026-09-29', direction: 'NO_MODEL', prob_up: 50, expected_return: 0,
  actual_return: 0.4, correct: null, features: [], designation: 'OOS',
  note: 'no combination cleared the gate',
}];
run('NO_MODEL day', nm, '2026-09-29');

console.log('\nALL DOM RENDER CASES PASSED');
