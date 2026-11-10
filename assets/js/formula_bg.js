/* Handwritten formula background.
 * Each formula is laid out from a small token list (text, sub, sup, frac,
 * sqrt, bar, nabla), then written stroke by stroke: text is revealed left to
 * right, lines are drawn with a dash offset. All clusters are written at
 * once within WRITE_TIME, held, faded out, and written again. The page picks
 * its set of clusters with data-set on .formula-bg. */
(function () {
  'use strict';
  var root = document.querySelector('.formula-bg');
  if (!root) return;

  var NS = 'http://www.w3.org/2000/svg';
  var FONT = 'Mynerve';
  var ZH_FONT = 'Ma Shan Zheng';
  var TEXT_SPEED = 120;   // relative pen speeds (viewBox units per second)
  var PATH_SPEED = 300;
  var STROKE_GAP = 0.07;  // relative pause between strokes
  var WRITE_TIME = 1.8;   // seconds: every cluster is written within this
  var HOLD = 18;          // seconds the finished page stays
  var FADE = 2.5;

  /* Each set is one page's background. `pos` places a cluster on the
   * viewport; clusters may sit behind the content. */
  var SETS = {
    projects: [
      { // Bayes' theorem
        pos: 'left:-1vw;top:9vh;width:clamp(330px, 31vw, 580px)', vb: [460, 250], items: [
          { m: ["Bayes' theorem"], x: 44, y: 54, s: 26, c: 'title' },
          { p: 'M30 60 C 22 28, 150 16, 226 32 C 266 42, 240 74, 152 78 C 82 82, 32 74, 36 52 C 40 38, 72 30, 98 28', c: 'acc' },
          { p: 'M78 84 C 70 106, 76 126, 96 138' },
          { p: 'M86 133 L97 139 L92 127' },
          { m: ['P(A|B) = ', { frac: [['P(B|A) P(A)'], ['P(B)']] }], x: 104, y: 162, s: 26 },
          { p: 'M330 214 C 312 210, 300 200, 296 186' },
          { p: 'M290 194 L296 184 L302 193' },
          { m: ['evidence'], x: 338, y: 214, s: 17 },
          { m: ['updates belief'], x: 318, y: 236, s: 17 }
        ]
      },
      { // Policy gradient
        pos: 'right:4vw;top:24vh;width:clamp(330px, 31vw, 580px)', vb: [460, 260], items: [
          { m: ['Policy gradient'], x: 36, y: 46, s: 26, c: 'title' },
          { p: 'M32 57 C 92 52, 152 59, 218 51', c: 'acc' },
          { p: 'M62 64 C 112 61, 160 65, 198 61', c: 'acc' },
          { m: [{ nabla: 1 }, { sub: 'θ' }, ' J(θ) = E', { sub: 'π' }, ' [ ', { nabla: 1 }, { sub: 'θ' }, ' log π', { sub: 'θ' }, '(a|s) · A(s,a) ]'], x: 26, y: 118, s: 22 },
          { p: 'M60 238 L60 166' },
          { p: 'M55 175 L60 164 L65 175' },
          { p: 'M60 238 L196 238' },
          { p: 'M186 233 L197 238 L186 243' },
          { p: 'M64 230 C 88 229, 96 218, 110 214 C 126 210, 132 198, 148 190 C 162 182, 170 176, 188 172' },
          { m: ['R'], x: 40, y: 172, s: 15 },
          { m: ['iters'], x: 172, y: 256, s: 15 },
          { p: 'M302 186 C 314 170, 320 152, 322 136' },
          { p: 'M316 145 L322 134 L328 145' },
          { m: ['push up actions'], x: 246, y: 206, s: 17 },
          { m: ['that beat the baseline'], x: 228, y: 228, s: 17 }
        ]
      },
      { // Diffusion
        pos: 'left:7vw;bottom:3vh;width:clamp(330px, 31vw, 580px)', vb: [460, 280], items: [
          { m: ['Diffusion'], x: 40, y: 46, s: 26, c: 'title' },
          { p: 'M36 58 C 50 52, 60 64, 74 58 C 88 52, 98 64, 112 58 C 126 52, 136 64, 150 58', c: 'acc' },
          { m: ['x', { sub: 't' }, ' = ', { sqrt: [{ bar: ['α'] }, { sub: 't' }] }, ' x', { sub: '0' }, ' + ', { sqrt: ['1 - ', { bar: ['α'] }, { sub: 't' }] }, ' ε'], x: 30, y: 112, s: 22 },
          { m: ['L = E || ε - ε', { sub: 'θ' }, '(x', { sub: 't' }, ', t) ||', { sup: '2' }], x: 30, y: 164, s: 22 },
          { m: ['x', { sub: '0' }], x: 40, y: 236, s: 20 },
          { p: 'M72 226 C 122 212, 172 212, 222 226' },
          { p: 'M211 219 L223 226 L210 231' },
          { m: ['add noise'], x: 112, y: 208, s: 15 },
          { m: ['x', { sub: 'T' }], x: 232, y: 236, s: 20 },
          { p: 'M224 246 C 172 260, 122 260, 74 248' },
          { p: 'M84 242 L73 248 L85 254' },
          { m: ['denoise'], x: 120, y: 274, s: 15 }
        ]
      },
      { // Softmax
        pos: 'right:-2vw;bottom:5vh;width:clamp(330px, 31vw, 580px)', vb: [460, 270], items: [
          { m: ['Softmax'], x: 44, y: 48, s: 26, c: 'title' },
          { p: 'M34 56 C 36 26, 146 22, 156 42 C 164 62, 64 68, 40 58', c: 'acc' },
          { m: ['σ(z)', { sub: 'i' }, ' = ', { frac: [['e', { sup: ['z', { sub: 'i' }] }], ['Σ', { sub: 'j' }, ' e', { sup: ['z', { sub: 'j' }] }]] }], x: 44, y: 128, s: 30 },
          { p: 'M184 166 C 210 184, 232 196, 262 202' },
          { p: 'M251 197 L263 202 L255 191' },
          { p: 'M276 236 L428 236' },
          { p: 'M290 236 L290 216 L310 216 L310 236' },
          { p: 'M324 236 L324 170 L344 170 L344 236' },
          { p: 'M358 236 L358 204 L378 204 L378 236' },
          { p: 'M392 236 L392 224 L412 224 L412 236' },
          { m: ['sums to 1'], x: 312, y: 262, s: 16 }
        ]
      }
    ],

    blog: [
      { // Special relativity
        pos: 'left:1vw;top:8vh;width:clamp(330px, 31vw, 580px)', vb: [460, 300], items: [
          { m: ['Special relativity'], x: 30, y: 44, s: 26, c: 'title' },
          { p: 'M26 56 C 100 51, 170 58, 236 50', c: 'acc' },
          { m: ['E = mc', { sup: '2' }], x: 44, y: 116, s: 40 },
          { p: 'M30 110 C 26 74, 186 68, 190 98 C 194 128, 54 134, 34 116', c: 'acc' },
          { m: ['γ = ', { frac: [['1'], [{ sqrt: ['1 - v', { sup: '2' }, '/c', { sup: '2' }] }]] }], x: 44, y: 206, s: 26 },
          { p: 'M330 284 L330 146' },
          { p: 'M325 156 L330 145 L335 156' },
          { p: 'M262 220 L414 220' },
          { p: 'M404 215 L415 220 L404 225' },
          { p: 'M282 268 L378 172' },
          { p: 'M378 268 L282 172' },
          { p: 'M290 180 C 300 168, 360 168, 370 180 C 360 192, 300 192, 290 180' },
          { p: 'M290 260 C 300 248, 360 248, 370 260 C 360 272, 300 272, 290 260' },
          { m: ['t'], x: 338, y: 154, s: 16 },
          { m: ['x'], x: 412, y: 238, s: 16 },
          { m: ['light cone'], x: 386, y: 286, s: 15 }
        ]
      },
      { // General relativity
        pos: 'right:2vw;top:20vh;width:clamp(350px, 34vw, 620px)', vb: [500, 270], items: [
          { m: ['Einstein field equations'], x: 30, y: 44, s: 24, c: 'title' },
          { p: 'M26 55 C 120 50, 220 57, 300 50', c: 'acc' },
          { m: ['G', { sub: 'μν' }, ' + Λg', { sub: 'μν' }, ' = ', { frac: [['8πG'], ['c', { sup: '4' }]] }, ' T', { sub: 'μν' }], x: 30, y: 124, s: 28 },
          { p: 'M18 186 C 44 188, 54 226, 76 226 C 98 226, 108 188, 134 186' },
          { p: 'M30 206 C 50 208, 60 238, 76 238 C 92 238, 102 208, 122 206' },
          { p: 'M82 218 a6 6 0 1 1 -12 0 a6 6 0 1 1 12 0' },
          { m: ['spacetime tells matter how to move;'], x: 156, y: 200, s: 17 },
          { m: ['matter tells spacetime how to curve'], x: 168, y: 224, s: 17 },
          { m: ['— J. A. Wheeler'], x: 340, y: 254, s: 15, c: 'title' }
        ]
      },
      { // 白云苍狗，四季枯荣 (vertical, right column first)
        pos: 'right:24vw;top:46vh;width:clamp(130px, 10vw, 190px)', vb: [200, 380], items: [
          { m: ['白'], x: 108, y: 70, s: 54, c: 'zh' },
          { m: ['云'], x: 108, y: 136, s: 54, c: 'zh' },
          { m: ['苍'], x: 108, y: 202, s: 54, c: 'zh' },
          { m: ['狗'], x: 108, y: 268, s: 54, c: 'zh' },
          { m: ['四'], x: 34, y: 120, s: 54, c: 'zh' },
          { m: ['季'], x: 34, y: 186, s: 54, c: 'zh' },
          { m: ['枯'], x: 34, y: 252, s: 54, c: 'zh' },
          { m: ['荣'], x: 34, y: 318, s: 54, c: 'zh' }
        ]
      },
      { // Universal gravitation
        pos: 'left:5vw;bottom:4vh;width:clamp(330px, 31vw, 580px)', vb: [460, 280], items: [
          { m: ['Universal gravitation'], x: 30, y: 44, s: 24, c: 'title' },
          { p: 'M26 55 C 100 50, 180 57, 250 50', c: 'acc' },
          { m: ['F = G ', { frac: [['m', { sub: '1' }, ' m', { sub: '2' }], ['r', { sup: '2' }]] }], x: 40, y: 128, s: 30 },
          { p: 'M250 210 C 250 160, 420 160, 420 210 C 420 260, 250 260, 250 210' },
          { p: 'M308 210 a8 8 0 1 1 -16 0 a8 8 0 1 1 16 0' },
          { p: 'M339 173 a4 4 0 1 1 -8 0 a4 4 0 1 1 8 0' },
          { p: 'M300 210 L333 177' },
          { m: ['r'], x: 322, y: 204, s: 15 }
        ]
      },
      { // Blake
        pos: 'left:36vw;bottom:3vh;width:clamp(340px, 32vw, 600px)', vb: [520, 170], items: [
          { m: ['To see a World in a Grain of Sand'], x: 20, y: 50, s: 26 },
          { m: ['And a Heaven in a Wild Flower'], x: 52, y: 92, s: 26 },
          { m: ['— William Blake'], x: 300, y: 130, s: 16, c: 'title' },
          { p: 'M40 150 C 140 142, 260 154, 380 144 C 400 142, 410 148, 396 154', c: 'acc' }
        ]
      }
    ]
  };
  var CLUSTERS = SETS[root.getAttribute('data-set')] || SETS.projects;

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function hline(a, b, y) {
    var d = b - a;
    return 'M' + a + ' ' + y + ' C ' + (a + d * 0.33) + ' ' + (y - 1.2) + ', ' +
      (a + d * 0.66) + ' ' + (y + 1) + ', ' + b + ' ' + (y - 0.4);
  }

  var measureSvg = el('svg', { width: 0, height: 0, style: 'position:absolute;visibility:hidden;font-family:' + FONT + ',cursive' }, root);

  function addPath(g, out, d, cls) {
    var n = el('path', { d: d }, g);
    if (cls) n.setAttribute('class', cls);
    out.push({ kind: 'path', node: n });
  }

  /* Lays out a token list from (x, y) at size s. With g null it only
   * measures. Returns the advance width. */
  function layout(tokens, x, y, s, g, out, cls) {
    if (typeof tokens === 'string') tokens = [tokens];
    var cx = x;
    for (var i = 0; i < tokens.length; i++) {
      var t = tokens[i];
      if (typeof t === 'string') {
        var n = el('text', { x: cx, y: y, 'font-size': s }, g || measureSvg);
        if (cls) n.setAttribute('class', cls);
        n.textContent = t;
        var w = n.getComputedTextLength();
        if (g) {
          out.push({ kind: 'text', node: n, x: cx, y: y, s: s, w: w });
        } else n.parentNode.removeChild(n);
        cx += w;
      } else if (t.sub !== undefined || t.sup !== undefined) {
        var up = t.sup !== undefined;
        cx += layout(up ? t.sup : t.sub, cx, up ? y - s * 0.42 : y + s * 0.24, s * 0.62, g, out, cls) + s * 0.04;
      } else if (t.frac) {
        cx += s * 0.1;
        var fs = s * 0.82;
        var wn = layout(t.frac[0], 0, 0, fs, null);
        var wd = layout(t.frac[1], 0, 0, fs, null);
        var fw = Math.max(wn, wd) + s * 0.4;
        if (g) {
          layout(t.frac[0], cx + (fw - wn) / 2, y - s * 0.52, fs, g, out, cls);
          addPath(g, out, hline(cx, cx + fw, y - s * 0.3));
          layout(t.frac[1], cx + (fw - wd) / 2, y + s * 0.62, fs, g, out, cls);
        }
        cx += fw + s * 0.12;
      } else if (t.sqrt) {
        var wi = layout(t.sqrt, 0, 0, s, null);
        cx += s * 0.06;
        var top = y - s * 1.02;
        if (g) {
          addPath(g, out, 'M' + cx + ' ' + (y - s * 0.32) + ' L' + (cx + s * 0.1) + ' ' + (y - s * 0.38) +
            ' L' + (cx + s * 0.24) + ' ' + (y + s * 0.06) + ' L' + (cx + s * 0.42) + ' ' + top +
            ' L' + (cx + s * 0.63 + wi) + ' ' + (top - 0.5));
          layout(t.sqrt, cx + s * 0.5, y, s, g, out, cls);
        }
        cx += s * 0.5 + wi + s * 0.16;
      } else if (t.bar) {
        var wb = layout(t.bar, 0, 0, s, null);
        if (g) {
          layout(t.bar, cx, y, s, g, out, cls);
          addPath(g, out, hline(cx + s * 0.06, cx + wb, y - s * 0.56));
        }
        cx += wb;
      } else if (t.nabla) {
        var nw = s * 0.56, nh = s * 0.64, x0 = cx + s * 0.04;
        if (g) addPath(g, out, 'M' + x0 + ' ' + (y - nh) + ' L' + (x0 + nw) + ' ' + (y - nh + 1) + ' L' + (x0 + nw * 0.52) + ' ' + (y + 1) + ' Z');
        cx += nw + s * 0.14;
      }
    }
    return cx - x;
  }

  /* Builds every cluster and returns the ordered stroke list per cluster. */
  function build() {
    return CLUSTERS.map(function (c, ci) {
      var svg = el('svg', { viewBox: '0 0 ' + c.vb[0] + ' ' + c.vb[1], style: c.pos }, root);
      var defs = el('defs', {}, svg);
      var g = el('g', {}, svg);
      var out = [];
      c.items.forEach(function (it) {
        if (it.p) addPath(g, out, it.p, it.c);
        else layout(it.m, it.x, it.y, it.s, g, out, it.c);
      });
      out.forEach(function (st, i) {
        if (st.kind === 'text') {
          var id = 'fbg-' + ci + '-' + i;
          var cp = el('clipPath', { id: id }, defs);
          st.rect = el('rect', { x: st.x - 2, y: st.y - st.s * 1.15, height: st.s * 1.7, width: 0 }, cp);
          st.node.setAttribute('clip-path', 'url(#' + id + ')');
          st.len = st.w + 4;
          st.dur = Math.max(0.18, st.len / TEXT_SPEED);
        } else {
          st.len = st.node.getTotalLength();
          st.node.style.strokeDasharray = st.len + ' ' + (st.len + 2);
          st.dur = Math.max(0.22, st.len / PATH_SPEED);
        }
      });
      return out;
    });
  }

  function setProgress(st, p) {
    if (st.kind === 'text') st.rect.setAttribute('width', st.len * p);
    else st.node.style.strokeDashoffset = st.len * (1 - p);
  }

  function ease(p) { return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; }

  function run(clusters) {
    // All clusters are written at once; each one's strokes are scaled so
    // the cluster finishes within WRITE_TIME.
    var strokes = [];
    var DELAY = 0.1;
    clusters.forEach(function (out) {
      var t = 0;
      out.forEach(function (st) { st.start = t; t += st.dur + STROKE_GAP; });
      var k = WRITE_TIME / t;
      out.forEach(function (st) { st.start = DELAY + st.start * k; st.dur *= k; strokes.push(st); });
    });
    strokes.sort(function (a, b) { return a.start - b.start; });
    var total = DELAY + WRITE_TIME;

    function reset() {
      strokes.forEach(function (st) { setProgress(st, 0); });
    }

    function cycle() {
      reset();
      root.classList.remove('is-fading');
      // Wall-clock timing: even when frames are throttled the page is
      // fully written WRITE_TIME seconds after it starts.
      var t0 = null;
      function frame(now) {
        if (t0 === null) t0 = now;
        var elapsed = (now - t0) / 1000;
        for (var i = 0; i < strokes.length; i++) {
          var st = strokes[i];
          var p = (elapsed - st.start) / st.dur;
          if (p <= 0) break;
          if (st.done) continue;
          if (p >= 1) { setProgress(st, 1); st.done = true; } else setProgress(st, ease(p));
        }
        if (elapsed < total) requestAnimationFrame(frame);
        else setTimeout(function () {
          root.classList.add('is-fading');
          setTimeout(function () {
            strokes.forEach(function (st) { st.done = false; });
            cycle();
          }, FADE * 1000 + 800);
        }, HOLD * 1000);
      }
      requestAnimationFrame(frame);
    }
    cycle();
  }

  function start() {
    var clusters = build();
    root.removeChild(measureSvg);
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      clusters.forEach(function (out) { out.forEach(function (st) { setProgress(st, 1); }); });
      return;
    }
    run(clusters);
  }

  // Load every subset the formulas use (Greek is a separate face), so text
  // is measured in the final font.
  var chars = '', zhChars = '';
  CLUSTERS.forEach(function (c) {
    c.items.forEach(function (it) {
      var acc = '';
      (function collect(x) {
        if (typeof x === 'string') acc += x;
        else if (Array.isArray(x)) x.forEach(collect);
        else if (x && typeof x === 'object') for (var k in x) collect(x[k]);
      })(it.m || []);
      if (it.c === 'zh') zhChars += acc; else chars += acc;
    });
  });
  var ready = Promise.resolve();
  if (document.fonts && document.fonts.load) {
    var loads = [document.fonts.load('26px ' + FONT, chars)];
    if (zhChars) loads.push(document.fonts.load('54px "' + ZH_FONT + '"', zhChars));
    ready = Promise.all(loads);
  }
  ready.then(start, start);
})();
