// Executes the dashboard's Layer A / Layer B pure functions against the real
// published fixture with a minimal DOM stub, and checks the numbers.
//
// Driven by tests/test_dom_option_layer.py; run directly with
// `node tests/dom_option_layer.js`. Exits non-zero on the first failure.
//
// Two things are verified that a Python-only test cannot reach:
//   1. The browser's Black-Scholes price agrees with core/option_pricing.py, so
//      the persisted chain and the rendered ratio cannot drift apart.
//   2. Sub-1.0 ratios are RENDERED, not suppressed -- a UI that hides an
//      unfavourable result passes every Python test and is still wrong.
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const fixture = JSON.parse(
  fs.readFileSync(path.join(ROOT, 'tests/fixtures/nifty_option_chain.json'), 'utf8')
);

// Reference values produced by core/option_pricing.py, so a change in either
// implementation has to be acknowledged on both sides.
const PY = {
  atm_call_price: 352.6194,     // 25000/25000, iv 0.13, 27d
  otm_call_price: 289.5764,     // 25000/25200, iv 0.14, 27d
  deep_itm_call_price: 5000.3316, // 25000/20000, iv 0.13, 27d
};

let failures = 0;
function check(label, cond, detail) {
  if (cond) { console.log('  ok   ' + label); }
  else { console.log('  FAIL ' + label + (detail ? ' :: ' + detail : '')); failures++; }
}
function near(a, b, tol) { return Math.abs(a - b) <= tol; }

// ---- extract a top-level function by name from index.html ---------------
function extractFn(name) {
  const re = new RegExp('function ' + name + '\\s*\\(');
  const m = html.match(re);
  if (!m) throw new Error('function ' + name + ' not found in index.html');
  const start = m.index;
  let depth = 0, i = html.indexOf('{', start), end = -1;
  for (; i < html.length; i++) {
    if (html[i] === '{') depth++;
    else if (html[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
  }
  return html.slice(start, end);
}

const parts = ['normCdf', 'bsPrice', 'priceLegScenarios', 'chooseOptionCandidate',
  'scoreCandidate', 'buildUnderlyingForecast', 'computeBtstGap',
  'buildProvenance', 'buildGapScenarios', 'renderRR', 'renderOptionLayer',
  'describeRiskFreeRate']
  .map(extractFn).join('\n');
const mod = new Function(parts + '\nreturn {' +
  'normCdf, bsPrice, priceLegScenarios, chooseOptionCandidate, scoreCandidate,' +
  'buildUnderlyingForecast, computeBtstGap,' +
  'buildProvenance, buildGapScenarios, renderRR, renderOptionLayer,' +
  'describeRiskFreeRate};')();

// ---- minimal DOM --------------------------------------------------------
const store = {};
global.document = {
  getElementById: (id) => (store[id] = store[id] || {
    set innerHTML(v) { this._h = v; }, get innerHTML() { return this._h || ''; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t || ''; },
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    set className(v) { this.classList._s = new Set(String(v).split(/\s+/)); },
  }),
  querySelector: () => null,
};

console.log('\n=== Black-Scholes parity with core/option_pricing.py ===');
const Y27 = 27 / 365.0;
const pAtm = mod.bsPrice(25000, 25000, Y27, 0.13, 0.0, 'CE');
check('ATM call price matches Python (352.62)', near(pAtm, PY.atm_call_price, 0.5),
  'js=' + pAtm.toFixed(4));
const pOtm = mod.bsPrice(25000, 25200, Y27, 0.14, 0.0, 'CE');
check('OTM call price matches Python (269.62)', near(pOtm, PY.otm_call_price, 0.5),
  'js=' + pOtm.toFixed(4));
const pItm = mod.bsPrice(25000, 20000, Y27, 0.13, 0.0, 'CE');
check('deep ITM call exceeds intrinsic 5000', pItm >= 5000 && pItm <= 25000, 'js=' + pItm.toFixed(4));
check('deep ITM call matches Python', near(pItm, PY.deep_itm_call_price, 0.5), 'js=' + pItm.toFixed(4));

// put-call parity at r=0, ATM => C == P
const pPut = mod.bsPrice(25000, 25000, Y27, 0.13, 0.0, 'PE');
check('put-call parity at r=0 holds in JS', near(pAtm, pPut, 0.5),
  'C=' + pAtm.toFixed(4) + ' P=' + pPut.toFixed(4));

// --- rate sensitivity. Every other call uses r=0, where the r*K term vanishes,
// so a mutant that DELETES the rate term is invisible without this check.
const PY_R = {
  c_r06: 409.9633,   // 25000/25000, iv 0.13, 27d, r=0.06
  p_r06: 299.2503,
};
const cR = mod.bsPrice(25000, 25000, Y27, 0.13, 0.06, 'CE');
const pR = mod.bsPrice(25000, 25000, Y27, 0.13, 0.06, 'PE');
check('rate 0.06 call matches Python', near(cR, PY_R.c_r06, 0.5), 'js=' + cR.toFixed(4));
check('rate 0.06 put matches Python', near(pR, PY_R.p_r06, 0.5), 'js=' + pR.toFixed(4));
// C - P = S - K e^(-rT): with r=0.06 this is no longer zero.
const parity6 = cR - pR;
const expectedParity = 25000 - 25000 * Math.exp(-0.06 * Y27);
check('put-call parity holds WITH a non-zero rate', near(parity6, expectedParity, 0.5),
  'got ' + parity6.toFixed(4) + ' want ' + expectedParity.toFixed(4));
check('a non-zero rate changes the price', Math.abs(cR - pAtm) > 0.5,
  'r=0:' + pAtm.toFixed(4) + ' r=0.06:' + cR.toFixed(4));

// --- scenario re-basing. The LTP used elsewhere equals the model price, which
// makes the re-basing a no-op and hides a mutant that deletes it. Here the LTP
// is deliberately far from the model, which is the case re-basing exists for.
const staleLtpLeg = { strike: 25000.0, expiry: '2026-10-27', dte: 27, leg: 'CE',
  ltp: 200.0, iv: 0.13 };
const rebased = mod.priceLegScenarios(staleLtpLeg, 25000.0, 0.0, 0.8, 0.4, 'LONG');
// Without re-basing, risk = ltp - model(stop) = 200 - 304 = -104 (negative).
check('stale LTP still yields positive risk after re-basing',
  rebased.option_risk > 0, 'risk=' + rebased.option_risk.toFixed(2));
check('stale LTP still yields an R:R', rebased.rr !== null);
check('re-basing discloses the model vs LTP gap',
  rebased.model_vs_ltp_gap_pct > 30, 'gap=' + rebased.model_vs_ltp_gap_pct.toFixed(1));
// The entry leg of the re-based path is the traded premium by construction.
const rawModelStop = mod.bsPrice(25000 * 0.996, 25000, Y27, 0.13, 0.0, 'CE');
check('raw model stop price would give negative risk (why re-basing exists)',
  staleLtpLeg.ltp - rawModelStop < 0,
  'ltp=' + staleLtpLeg.ltp + ' modelStop=' + rawModelStop.toFixed(2));

// unpriceable inputs must return null, never a number
check('zero time -> null', mod.bsPrice(25000, 25000, 0, 0.13, 0.0, 'CE') === null);
check('zero vol -> null', mod.bsPrice(25000, 25000, Y27, 0, 0.0, 'CE') === null);
check('zero spot -> null', mod.bsPrice(0, 25000, Y27, 0.13, 0.0, 'CE') === null);
check('bad leg -> null', mod.bsPrice(25000, 25000, Y27, 0.13, 0.0, 'XX') === null);

check('normCdf(0) ~= 0.5', near(mod.normCdf(0), 0.5, 1e-6));
check('normCdf(1.96) ~= 0.975', near(mod.normCdf(1.96), 0.975, 1e-4),
  'got ' + mod.normCdf(1.96));

console.log('\n=== IV sensitivity (must be monotone) ===');
const ivLo = mod.bsPrice(25000, 25000, Y27, 0.11, 0.0, 'CE');
const ivHi = mod.bsPrice(25000, 25000, Y27, 0.15, 0.0, 'CE');
check('higher IV -> higher price', ivHi > ivLo, ivLo.toFixed(2) + ' -> ' + ivHi.toFixed(2));

console.log('\n=== buildUnderlyingForecast(): pure Layer A, tested directly ===');

// The reported defect: the stop displayed on screen was clamped to 10 points
// while the R:R used the unclamped level, so the ratio beside it did not match.
// These inputs clear the LONG gate (mfe50 0.90 > mae50 0.61 * 1.1) while still
// producing an UNFAVOURABLE ratio, which is the case worth guarding.
const BUG = {
  price: 22726.0, upC: 12, downC: 8, pUp: 72, pDn: 30, pRg: 20,
  mfe25: 0.30, mae25: 0.25, mfe50: 0.90, mae50: 0.61,
  mfe75: 1.20, mae75: 1.12, mfe90: 1.80, mae90: 1.90,
  volMult: 1.0, isBTST: false,
};

const fcLong = mod.buildUnderlyingForecast(BUG);
check('strong upside qualifies LONG', fcLong.direction === 'LONG', 'got ' + fcLong.direction);
check('LONG bias text is generated FROM direction',
  fcLong.bias === fcLong.direction + ' BIAS', fcLong.bias);

// The reported defect: the stop displayed on screen was clamped to 10 points
// while the R:R used the unclamped level, so the ratio beside it did not match.
const lRisk = fcLong.entry.assumed_fill - fcLong.stop.price;
const lReward = fcLong.t2.price - fcLong.entry.assumed_fill;
check('displayed stop IS the stop used in the R:R',
  near(lRisk, fcLong.rr.risk, 1e-9), 'risk=' + lRisk + ' rr.risk=' + fcLong.rr.risk);
check('R:R value equals reward/risk',
  near(fcLong.rr.value, lReward / lRisk, 1e-12));

// Percentile pairing: the old code measured reward to the MEDIAN target but risk
// to the 75th-percentile stop, which structurally biases the ratio below 1.0.
const medianTarget = BUG.price + BUG.price * (BUG.mfe50 * BUG.volMult / 100);
const oldStyle = (medianTarget - fcLong.entry.assumed_fill) / lRisk;
check('T2 is the 75th-percentile target, not the median',
  near(fcLong.t2.pct, BUG.mfe75 * BUG.volMult, 1e-9),
  't2.pct=' + fcLong.t2.pct);
check('same-percentile pairing raises the ratio above the old median-target value',
  fcLong.rr.value > oldStyle, 'new=' + fcLong.rr.value.toFixed(4) + ' old=' + oldStyle.toFixed(4));

// An unfavourable geometry must still yield a NUMBER, never be nulled out.
// mfe50 stays above mae50*1.1 so the trade still QUALIFIES; the stop is simply
// further away than the target, which is the sub-1.0 geometry under test.
const BAD = Object.assign({}, BUG, { mfe50: 0.70, mfe75: 0.35, mfe90: 0.40, mae50: 0.60, mae75: 1.80 });
const fcBad = mod.buildUnderlyingForecast(BAD);
check('unfavourable geometry still qualifies', fcBad.qualified === true, 'got ' + fcBad.direction);
check('sub-1.0 R:R is returned, not suppressed', fcBad.rr.value !== null && fcBad.rr.value > 0,
  'got ' + fcBad.rr.value);
check('sub-1.0 R:R is genuinely below 1.0', fcBad.rr.value < 1.0, 'got ' + fcBad.rr.value);
check('sub-1.0 R:R is flagged by renderRR', mod.renderRR(fcBad.rr.value, '').warning !== null);

// A genuinely favourable geometry: target well beyond the stop.
const GOOD = Object.assign({}, BUG, { mfe50: 1.20, mfe75: 1.90, mfe90: 2.40, mae50: 0.50, mae75: 0.60 });
const fcGood = mod.buildUnderlyingForecast(GOOD);
check('favourable geometry qualifies', fcGood.qualified === true, 'got ' + fcGood.direction);
check('favourable R:R exceeds 1.0', fcGood.rr.value > 1.0, 'got ' + fcGood.rr.value);
check('favourable R:R carries no warning', mod.renderRR(fcGood.rr.value, '').warning === null);
check('a null R:R renders as "--" and carries no warning',
  mod.renderRR(null, '').display === '--' && mod.renderRR(null, '').warning === null);

// For a SHORT the gates are mirrored, so the excursions must swap too.
const asShort = (o) => Object.assign({}, o, {
  upC: 8, downC: 12, pUp: 30, pDn: 72,
  mfe50: o.mae50, mae50: o.mfe50, mfe75: o.mae75, mae75: o.mfe75,
  mfe90: o.mae90, mae90: o.mfe90,
});

const fcShort = mod.buildUnderlyingForecast(asShort(BUG));
check('strong downside qualifies SHORT', fcShort.direction === 'SHORT', 'got ' + fcShort.direction);
check('SHORT targets sit below spot', fcShort.t2.price < BUG.price);
check('SHORT stop sits above spot', fcShort.stop.price > BUG.price);
check('SHORT risk is positive', fcShort.rr.risk > 0);
check('SHORT reward is positive', fcShort.rr.reward > 0);
check('SHORT entry sits below spot', fcShort.entry.assumed_fill < BUG.price);

// A fully mirrored input set must produce a mirrored geometry, not a different one.
const fcMirror = mod.buildUnderlyingForecast(asShort(BUG));
// The entry offset is driven by mae25 for a LONG and mfe25 for a SHORT, and
// asShort() does not swap those two, so the fills are not exactly mirrored.
check('mirrored inputs mirror the target magnitude',
  near(fcMirror.t2.pct, fcLong.t2.pct, 1e-9),
  'short=' + fcMirror.t2.pct + ' long=' + fcLong.t2.pct);
check('mirrored inputs mirror the stop magnitude',
  near(fcMirror.stop.pct, fcLong.stop.pct, 1e-9),
  'short=' + fcMirror.stop.pct + ' long=' + fcLong.stop.pct);
check('mirrored inputs give a ratio of the same order',
  near(fcMirror.rr.value, fcLong.rr.value, 0.05),
  'long=' + fcLong.rr.value.toFixed(6) + ' mirrored=' + fcMirror.rr.value.toFixed(6));
// A full mirror, including the shallow-excursion pair, must be exact.
const fcExactMirror = mod.buildUnderlyingForecast(asShort(
  Object.assign({}, BUG, { mfe25: BUG.mae25, mae25: BUG.mfe25 })));
check('a fully mirrored input set gives an exactly mirrored R:R',
  near(fcExactMirror.rr.value, fcLong.rr.value, 1e-9),
  'long=' + fcLong.rr.value.toFixed(9) + ' mirrored=' + fcExactMirror.rr.value.toFixed(9));
// The assumed fill is a mirrored DISTANCE from spot, not the same absolute
// price: a LONG fills above spot and a SHORT below it.
check('a fully mirrored input set fills equidistant from spot',
  near(fcExactMirror.entry.assumed_fill + fcLong.entry.assumed_fill, 2 * BUG.price, 1e-9),
  'long=' + fcLong.entry.assumed_fill + ' short=' + fcExactMirror.entry.assumed_fill);

// --- no trade must be inert -------------------------------------------------
const fcNone = mod.buildUnderlyingForecast(
  Object.assign({}, BUG, { upC: 10, downC: 10, pUp: 40, pDn: 40 }));
check('chop produces no direction', fcNone.direction === null, 'got ' + fcNone.direction);
check('chop sets qualified=false', fcNone.qualified === false);
check('chop supplies no option inputs (nothing to price)',
  fcNone.option_inputs === null);
check('chop yields no R:R', fcNone.rr === undefined || fcNone.rr.value === null);
check('chop gap is UNKNOWN', fcNone.gap.gap_direction === 'UNKNOWN');

// --- purity -----------------------------------------------------------------
const snapshot = JSON.stringify(BUG);
mod.buildUnderlyingForecast(BUG);
check('buildUnderlyingForecast does not mutate its input',
  JSON.stringify(BUG) === snapshot);
check('buildUnderlyingForecast is deterministic',
  JSON.stringify(mod.buildUnderlyingForecast(BUG)) === JSON.stringify(fcLong));

// --- Layer B must consume the SAME percentiles as Layer A -------------------
check('option expected move equals the T2 move',
  near(fcLong.option_inputs.expected_move_pct, fcLong.t2.pct, 1e-12),
  fcLong.option_inputs.expected_move_pct + ' vs ' + fcLong.t2.pct);
check('option invalidation equals the stop move',
  near(fcLong.option_inputs.invalidation_pct, fcLong.stop.pct, 1e-12),
  fcLong.option_inputs.invalidation_pct + ' vs ' + fcLong.stop.pct);
check('option inputs carry the direction', fcLong.option_inputs.direction === 'LONG');
check('SHORT option inputs mirror the SHORT levels',
  near(fcShort.option_inputs.expected_move_pct, fcShort.t2.pct, 1e-12));

// --- volMult must actually scale the levels ---------------------------------
const fcVol = mod.buildUnderlyingForecast(Object.assign({}, BUG, { volMult: 2.0 }));
check('volMult scales the target', near(fcVol.t2.pct, 2 * fcLong.t2.pct, 1e-9));
check('volMult scales the stop', near(fcVol.stop.pct, 2 * fcLong.stop.pct, 1e-9));

// --- qualification gate boundaries -----------------------------------------
console.log('\n=== Qualification gates are individually load-bearing ===');
// The three LONG gates must each be load-bearing. Without these, deleting any
// one of them still passes every other check.
const longBase = () => Object.assign({}, GOOD, { isBTST: true });
check('excursion gate boundary: just above qualifies',
  mod.buildUnderlyingForecast(longBase()).qualified === true);
const atExcursionEdge = Object.assign({}, longBase(), { mfe50: 0.661, mae50: 0.601 });
check('excursion gate boundary: exactly at mfe50 = mae50*1.1 does NOT qualify',
  mod.buildUnderlyingForecast(atExcursionEdge).qualified === false,
  'mfe50=' + atExcursionEdge.mfe50 + ' mae50*1.1=' + (atExcursionEdge.mae50 * 1.1).toFixed(4));
check('excursion gate boundary: just above it does qualify',
  mod.buildUnderlyingForecast(
    Object.assign({}, longBase(), { mfe50: 0.662, mae50: 0.601 })).qualified === true);
const weakProb = Object.assign({}, longBase(), { pUp: 40, pRg: 20 });
check('probability gate: pUp 40 with pUp+pRg 60 is rejected',
  mod.buildUnderlyingForecast(weakProb).qualified === false,
  'pUp=' + weakProb.pUp + ' sum=' + (weakProb.pUp + weakProb.pRg));
check('probability gate: pUp 40 but pUp+pRg 71 qualifies',
  mod.buildUnderlyingForecast(
    Object.assign({}, longBase(), { pUp: 40, pRg: 31 })).qualified === true);
check('probability gate: pUp 46 alone qualifies',
  mod.buildUnderlyingForecast(
    Object.assign({}, longBase(), { pUp: 46, pRg: 0 })).qualified === true);
check('probability gate: pUp 45 exactly does NOT qualify alone',
  mod.buildUnderlyingForecast(
    Object.assign({}, longBase(), { pUp: 45, pRg: 0 })).qualified === false);

// The 0.05% floor on the shallow excursion must bind for a very shallow MAE,
// otherwise the fill collapses onto spot and the R:R becomes meaningless.
const shallow = mod.buildUnderlyingForecast(
  Object.assign({}, longBase(), { mfe25: 0.02, mae25: 0.02 }));
check('shallow-excursion floor binds below 0.05%',
  near(shallow.entry.pct, 0.05, 1e-12), 'entry.pct=' + shallow.entry.pct);
check('shallow floor keeps the fill off spot',
  shallow.entry.assumed_fill > shallow.price);
check('shallow floor applies symmetrically to a SHORT',
  near(mod.buildUnderlyingForecast(
    Object.assign({}, asShort(longBase()), { mfe25: 0.02, mae25: 0.02 })
  ).entry.pct, 0.05, 1e-12));
const deep = mod.buildUnderlyingForecast(Object.assign({}, longBase(), { mfe25: 0.4, mae25: 0.4 }));
check('shallow floor does not bind when the excursion is larger',
  near(deep.entry.pct, 0.2, 1e-12), 'entry.pct=' + deep.entry.pct);

console.log('\n=== Structured BTST gap state (no prose inference) ===');
const BTST = Object.assign({}, BUG, { isBTST: true });
const fcBtst = mod.buildUnderlyingForecast(BTST);
check('BTST gap direction is UP for a positive overnight gap',
  mod.computeBtstGap('LONG', 0.8, true).gap_direction === 'UP');
check('BTST gap direction is DOWN for a negative overnight gap',
  mod.computeBtstGap('LONG', -0.8, true).gap_direction === 'DOWN');
check('BTST gap direction is FLAT for a zero gap',
  mod.computeBtstGap('LONG', 0.0, true).gap_direction === 'FLAT');
check('BTST gap is UNKNOWN with no overnight data',
  mod.computeBtstGap('LONG', null, true).gap_direction === 'UNKNOWN',
  'got ' + mod.computeBtstGap('LONG', null, true).gap_direction);
check('BTST gap is UNKNOWN with NaN, not coerced to FLAT',
  mod.computeBtstGap('LONG', NaN, true).gap_direction === 'UNKNOWN');
check('an unchanged forecast defaults to an UNKNOWN gap',
  fcBtst.gap.gap_direction === 'UNKNOWN' && fcBtst.gap.gap_pct === null,
  JSON.stringify(fcBtst.gap));
// A NO-TRADE day must be inert in the gap path too. This is a BTST forecast, so
// it reaches the NO_TRADE branch rather than being short-circuited as N/A.
const fcNoneBtst = mod.buildUnderlyingForecast(
  Object.assign({}, BUG, { isBTST: true, upC: 10, downC: 10, pUp: 40, pDn: 40 }));
check('no-trade BTST forecast has an UNKNOWN gap, not a fabricated one',
  fcNoneBtst.gap.gap_direction === 'UNKNOWN' && fcNoneBtst.gap.gap_pct === null,
  JSON.stringify(fcNoneBtst.gap));
check('no-trade BTST gap state is NO_TRADE',
  fcNoneBtst.gap.state === 'NO_TRADE', fcNoneBtst.gap.state);
check('no-trade forecast yields no provenance rows',
  mod.buildProvenance(fcNone) === null && mod.buildProvenance(fcNoneBtst) === null);
check('LONG + gap UP is ALIGNED', mod.computeBtstGap('LONG', 0.8, true).state === 'ALIGNED');
check('LONG + gap DOWN is ADVERSE', mod.computeBtstGap('LONG', -0.8, true).state === 'ADVERSE');
check('SHORT + gap DOWN is ALIGNED', mod.computeBtstGap('SHORT', -0.8, true).state === 'ALIGNED');
check('SHORT + gap UP is ADVERSE', mod.computeBtstGap('SHORT', 0.8, true).state === 'ADVERSE');
check('non-BTST gap is NOT_APPLICABLE',
  mod.computeBtstGap('LONG', 0.8, false).state === 'NOT_APPLICABLE');
check('gap carries the 0.5% trigger threshold',
  mod.computeBtstGap('LONG', 0.8, true).trigger_gap_pct === 0.5);

// Direction must be readable as a FIELD, never parsed out of provenance text.
const prov = mod.buildProvenance(fcLong);
check('provenance is display-only and still generated',
  Array.isArray(prov) && prov.length === 6);
check('provenance does not leak the direction as a parseable contract',
  typeof fcLong.direction === 'string' && fcLong.direction === 'LONG');
check('gap scenarios derive direction from the forecast field',
  mod.buildGapScenarios(fcBtst).aligned.label === 'Gap UP');
check('gap scenarios are null for a non-BTST forecast',
  mod.buildGapScenarios(fcLong) === null);
check('gap scenarios are null for an unqualified forecast',
  mod.buildGapScenarios(fcNone) === null);
const fcShortBtst = mod.buildUnderlyingForecast(
  Object.assign(asShort(BUG), { isBTST: true }));
check('SHORT gap scenarios put the adverse case first (Gap UP)',
  mod.buildGapScenarios(fcShortBtst).adverse.label === 'Gap UP');
check('SHORT gap scenarios put the aligned case second (Gap DOWN)',
  mod.buildGapScenarios(fcShortBtst).aligned.label === 'Gap DOWN');

console.log('\n=== Risk-free rate labelling ===');
const PROD_RATE = {
  risk_free_rate: 0.0717, risk_free_rate_configured: true,
  risk_free_rate_source: 'India 10Y Government Security yield',
  risk_free_rate_as_of: '2026-09-29', risk_free_rate_type: 'proxy',
  risk_free_rate_refresh: 'manual',
  candidates: [{ dte: 6 }, { dte: 27 }],
};
const rateText = mod.describeRiskFreeRate(PROD_RATE);
check('production rate is rendered as 7.17%', rateText.indexOf('7.17%') >= 0, rateText);
check('production rate is named as the 10Y G-Sec source',
  rateText.indexOf('10Y Government Security') >= 0, rateText);
check('production rate is labelled a PROXY, not a curve',
  rateText.indexOf('PROXY') >= 0 && rateText.indexOf('not a maturity-matched') >= 0, rateText);
check('production rate states its as-of date', rateText.indexOf('2026-09-29') >= 0, rateText);
check('production rate names the DTE range it is being used across',
  rateText.indexOf('6-27d') >= 0, rateText);
check('an unconfigured rate is disclosed, never invented',
  mod.describeRiskFreeRate({ risk_free_rate: null, risk_free_rate_configured: false })
    .indexOf('NOT configured') >= 0);
check('a configured zero rate is not mislabelled as unconfigured',
  mod.describeRiskFreeRate({ risk_free_rate: 0, risk_free_rate_configured: true })
    .indexOf('0.00%') >= 0);

console.log('\n=== Option R:R is in option points, not underlying points ===');
const payload = {
  source: 'live', fetched_at: '2026-09-30', underlying: 25000.0,
  risk_free_rate: null, risk_free_rate_configured: false,
  candidates: [
    {
      expiry: '2026-10-06', dte: 6, is_monthly: false,
      legs: [{ strike: 25000.0, expiry: '2026-10-06', dte: 6, leg: 'CE', ltp: 189.4, iv: 0.1478,
        greeks: { delta: 0.504, gamma: 0.0021, theta: -18.2, vega: 21.0 } }],
    },
    {
      expiry: '2026-10-27', dte: 27, is_monthly: true,
      legs: [{ strike: 25000.0, expiry: '2026-10-27', dte: 27, leg: 'CE', ltp: 352.6, iv: 0.13,
        greeks: { delta: 0.507, gamma: 0.00045, theta: -18.24, vega: 25.3 } }],
    },
  ],
};
// target/invalidation of 0.8/0.4 => underlying R:R is exactly 2.0. The option
// ratio must NOT be exactly 2.0, or Layer B is just echoing Layer A.
const trade = mod.chooseOptionCandidate(payload,
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.4, btst: false },
  25000.0, 0.0);
check('a trade was constructed', trade !== null);
check('option R:R is not the underlying 2.0', Math.abs(trade.rr - 2.0) > 0.01,
  'rr=' + trade.rr.toFixed(4));
check('option R:R is reward/risk',
  near(trade.rr, trade.option_reward / trade.option_risk, 1e-9));
check('option risk is in premium units (not index points)',
  trade.option_risk < 1000, 'risk=' + trade.option_risk.toFixed(2));
check('LONG selects a call', trade.leg === 'CE');
// Greeks must be passed through from the payload verbatim, so whichever leg the
// model picked, its persisted greeks object is what comes back.
const picked = payload.candidates
  .map(c => c.legs[0])
  .find(l => l.strike === trade.strike && l.expiry === trade.expiry);
check('greeks come from the payload, not recomputed in JS',
  trade.greeks && trade.greeks.delta === picked.greeks.delta
    && trade.greeks.theta === picked.greeks.theta,
  'got ' + JSON.stringify(trade.greeks) + ' want ' + JSON.stringify(picked.greeks));

console.log('\n=== SHORT flips the leg and inverts the scenarios ===');
const shortTrade = mod.chooseOptionCandidate(
  { underlying: 25000.0, risk_free_rate: 0.0, candidates: [{
      expiry: '2026-10-27', dte: 27, is_monthly: true,
      legs: [{ strike: 25000.0, expiry: '2026-10-27', dte: 27, leg: 'PE', ltp: 352.6, iv: 0.13,
        greeks: { delta: -0.493, gamma: 0.00045, theta: -18.1, vega: 25.3 } }]}] },
  { direction: 'SHORT', expected_move_pct: 0.8, invalidation_pct: 0.4, btst: false },
  25000.0, 0.0);
check('SHORT selects a put', shortTrade.leg === 'PE');
check('SHORT produces positive R:R', shortTrade.rr > 0, 'rr=' + shortTrade.rr.toFixed(4));
check('SHORT has positive risk', shortTrade.option_risk > 0);

console.log('\n=== Missing data must not fabricate a ratio ===');
const noIv = mod.chooseOptionCandidate(
  { underlying: 25000.0, candidates: [{ expiry: '2026-10-27', dte: 27, is_monthly: true,
      legs: [{ strike: 25000.0, expiry: '2026-10-27', dte: 27, leg: 'CE', ltp: 352.6, iv: null }]}] },
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.4 }, 25000.0, 0.0);
check('missing IV -> rr null', noIv.rr === null);
check('missing IV -> reason given', !!noIv.reason, noIv.reason);

const expired = mod.chooseOptionCandidate(
  { underlying: 25000.0, candidates: [{ expiry: '2026-09-29', dte: -1, is_monthly: false,
      legs: [{ strike: 25000.0, expiry: '2026-09-29', dte: -1, leg: 'CE', ltp: 210.0, iv: 0.14 }]}] },
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.4 }, 25000.0, 0.0);
check('expired expiry -> rr null', expired.rr === null);
check('expired expiry -> reason given', !!expired.reason, expired.reason);

check('no candidates -> null', mod.chooseOptionCandidate({ candidates: [] },
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.4 }, 25000, 0) === null);
check('null payload -> null', mod.chooseOptionCandidate(null,
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.4 }, 25000, 0) === null);
check('no direction -> null', mod.chooseOptionCandidate(payload, {}, 25000, 0) === null);

console.log('\n=== Sensitivity: perturbing inputs must change the output ===');
const mv = mod.chooseOptionCandidate(payload,
  { direction: 'LONG', expected_move_pct: 0.3, invalidation_pct: 0.4 }, 25000, 0.0).rr;
const mvBig = mod.chooseOptionCandidate(payload,
  { direction: 'LONG', expected_move_pct: 1.6, invalidation_pct: 0.4 }, 25000, 0.0).rr;
check('bigger expected move -> higher R:R', mvBig > mv, mv.toFixed(3) + ' -> ' + mvBig.toFixed(3));
const inv = mod.chooseOptionCandidate(payload,
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 0.2 }, 25000, 0.0).rr;
const invWide = mod.chooseOptionCandidate(payload,
  { direction: 'LONG', expected_move_pct: 0.8, invalidation_pct: 1.2 }, 25000, 0.0).rr;
check('wider invalidation -> lower R:R', invWide < inv, inv.toFixed(3) + ' -> ' + invWide.toFixed(3));

console.log('\n=== renderRR: sub-1.0 must be shown, not hidden ===');
const rrGood = mod.renderRR(2.35, 'fill 25000 -> T2');
check('favourable ratio has no warning', rrGood.warning === null);
check('favourable ratio displays 2 decimals', rrGood.display === '1 : 2.35', rrGood.display);
const rrBad = mod.renderRR(0.47, 'fill 25000 -> T2');
check('UNFAVOURABLE ratio still displays a number', rrBad.display === '1 : 0.47', rrBad.display);
check('UNFAVOURABLE ratio is flagged', rrBad.warning !== null && rrBad.warning < 1.0);
const rrNull = mod.renderRR(null, '');
check('null ratio -> "--" not a fake number', rrNull.display === '--');
const rrOne = mod.renderRR(1.0, '');
check('exactly 1.00 is not flagged', rrOne.warning === null);

console.log('\n=== Rendered DOM: unfavourable option R:R must appear on screen ===');
store['opt-status'] = undefined; store['opt-body'] = undefined;
store['opt-candidates'] = undefined; store['opt-assumptions'] = undefined;
const badTrade = {
  direction: 'LONG', leg: 'CE', strike: 25000.0, expiry: '2026-10-27', dte: 27,
  ltp: 352.6, iv: 0.13, rr: 0.62, reason: null,
  option_at_target: 380.0, option_at_stop: 250.0, option_risk: 102.6, option_reward: 27.4,
  greeks: { delta: 0.507, gamma: 0.00045, theta: -18.2, vega: 25.3 }, notes: ['test note'],
  model_vs_ltp_gap_pct: 12.4,
};
mod.renderOptionLayer(payload, { option_trade: badTrade });
const body = store['opt-body'].innerHTML;
check('sub-1.0 ratio is rendered numerically', body.includes('1 : 0.62'), body.slice(0, 120));
check('unfavourable warning badge is shown', body.includes('UNFAVOURABLE OPTION RISK'));
check('premium path is shown in option points', body.includes('Premium path'));
check('greek delta rendered', body.includes('0.507'));
check('assumptions disclose the unconfigured rate',
  store['opt-assumptions'].textContent.includes('NOT configured'));
check('assumptions disclose the model/LTP gap',
  store['opt-assumptions'].textContent.includes('model vs LTP gap'));
check('no NaN leaked into the DOM', !body.includes('NaN') && !body.includes('undefined'));

console.log('\n=== Rendered DOM: no chain must not invent anything ===');
mod.renderOptionLayer(null, { option_trade: null });
check('absent chain -> NO CHAIN status', store['opt-status'].textContent === 'NO CHAIN');
check('absent chain -> explanatory body',
  store['opt-body'].textContent.includes('No option chain published'));

console.log('\n=== Rendered DOM: stale chain is refused, not priced ===');
mod.renderOptionLayer({ underlying: 25000, risk_free_rate: null, risk_free_rate_configured: false,
  candidates: [{ expiry: '2020-01-16', dte: -900, is_monthly: false, legs: [] }] },
  { option_trade: null });
check('stale chain -> STALE status', store['opt-status'].textContent === 'STALE CHAIN');
check('stale chain -> refusal message', store['opt-body'].innerHTML.includes('already passed'));

console.log('\n=== Provenance: every level names its source ===');
// Driven from the real forecast, not a hand-built bag of params, so this also
// proves buildProvenance consumes the structured contract.
const provFc = mod.buildUnderlyingForecast(Object.assign({}, BUG, {
  mfe25: 0.4, mae25: 0.3, mfe50: 0.9, mae50: 0.6, mfe75: 1.4, mae75: 1.0, mfe90: 2.1,
}));
const provRows = mod.buildProvenance(provFc);
check('provenance has a row per level', provRows.length >= 6, 'rows=' + provRows.length);
check('provenance names the percentile', provRows.some(r => String(r[2]).includes('50th pct')));
check('provenance covers the stop', provRows.some(r => r[0] === 'Stop'));
check('provenance covers R:R basis', provRows.some(r => r[0] === 'R:R basis'));
check('no provenance row is blank', provRows.every(r => r.every(x => x !== undefined && x !== '')));

console.log('\n=== Overnight gap scenarios replace the contradiction ===');
// Driven from a real BTST forecast, so the scenarios are proven to consume the
// structured direction rather than a hand-passed string.
const gaps = mod.buildGapScenarios(fcBtst);
check('LONG aligned gap is UP', gaps.aligned.pct > 0, 'pct=' + gaps.aligned.pct);
check('LONG adverse gap is DOWN', gaps.adverse.pct < 0);
check('flat open sits at spot', near(gaps.flat.level, BUG.price, 1e-9));
check('aligned gap satisfies the LONG entry trigger',
  gaps.aligned.survives.indexOf('trigger met') >= 0, gaps.aligned.survives);
check('adverse gap does NOT silently void the setup',
  gaps.adverse.survives.indexOf('wait') >= 0, gaps.adverse.survives);
const shortGaps = mod.buildGapScenarios(fcShortBtst);
check('SHORT aligned gap is DOWN', shortGaps.aligned.pct < 0);
check('SHORT adverse gap is UP', shortGaps.adverse.pct > 0);
check('non-BTST -> no gap block', mod.buildGapScenarios(fcLong) === null);
check('gap bands are anchored on the forecast price, not a passed-in price',
  near(gaps.aligned.level, BUG.price * 1.005, 1e-9),
  gaps.aligned.level);

console.log('\n=== Both expiries are preserved in the payload ===');
check('fixture carries 2 candidates', payload.candidates.length === 2);
check('front is not marked monthly', payload.candidates[0].is_monthly === false);
check('monthly is marked monthly', payload.candidates[1].is_monthly === true);
check('DTEs are surfaced', payload.candidates[0].dte === 6 && payload.candidates[1].dte === 27);

console.log('\n' + (failures ? 'FAILED: ' + failures + ' check(s)' : 'ALL CHECKS PASSED'));
process.exit(failures ? 1 : 0);
