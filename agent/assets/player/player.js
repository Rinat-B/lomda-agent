/* Lesson player: renders window.LESSON (from lesson.js) with the fixed rules of the spec.
 *
 * - Never shows the correct answer after a wrong one; the result lists only which
 *   questions were wrong (number, topic, wording).
 * - "Retry" resets everything and shows the alternate questions.
 * - "Next" opens only after the whole screen was seen (scrolled to the end), every
 *   "stop and think" explanation was opened, and every question was checked.
 * - Keyboard: all controls are native buttons/inputs; focus moves to each new screen.
 * - No external libraries and no network: works from file://.
 */
(function () {
  "use strict";

  var S = window.PLAYER_STRINGS || {};
  var L = window.LESSON;
  var SVG_NS = "http://www.w3.org/2000/svg";
  var DEFAULT_PASSING = 0.8;

  function t(key, vars) {
    var s = S[key] || key;
    if (vars) {
      Object.keys(vars).forEach(function (k) { s = s.split("{" + k + "}").join(String(vars[k])); });
    }
    return s;
  }

  // Build an element; strings become text nodes (never HTML), arrays are flattened.
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") node.className = v;
      else if (k === "hidden") node.hidden = true;
      else node.setAttribute(k, v === true ? "" : v);
    });
    [].concat(children === undefined ? [] : children).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  var main = document.getElementById("main");
  document.getElementById("skip-link").textContent = t("skip");

  if (!L) {
    main.appendChild(el("p", { role: "alert" }, t("noLesson")));
    return;
  }

  // ---- setup ---------------------------------------------------------------------------
  var rootStyle = document.documentElement.style;
  Object.keys(L.palette || {}).forEach(function (k) {
    rootStyle.setProperty("--" + k.replace(/_/g, "-"), L.palette[k]);
  });
  document.body.classList.add("type-" + L.lesson_type);
  document.title = L.title;
  document.getElementById("lesson-title").textContent = L.title;
  if (L.motif && L.motif.svg) {
    // Motif SVG is sanitized by the agent's output guard before it is written to lesson.js.
    document.getElementById("motif").innerHTML = L.motif.svg;
  }

  var P = L.palette || {};

  var questions = []; // in lesson order, for numbering and the result list
  L.screens.forEach(function (screen) {
    screen.blocks.forEach(function (b) {
      if (b.type === "multiple_choice") questions.push({ block: b, screen: screen, number: questions.length + 1 });
    });
  });

  var state = { index: 0, round: 0, answers: {} };
  var observer = null;

  function versionOf(block) {
    // Odd rounds (each retry flips) use the alternate question.
    return state.round % 2 === 1 ? block.alternate : block;
  }

  function questionNumber(id) {
    for (var i = 0; i < questions.length; i++) if (questions[i].block.id === id) return questions[i].number;
    return 0;
  }

  function setProgress(done, label) {
    var total = L.screens.length;
    document.getElementById("progress-fill").style.width = Math.round(done / total * 100) + "%";
    var text = document.getElementById("progress-text");
    text.textContent = label;
    document.getElementById("progress").setAttribute("aria-label", t("progressLabel"));
  }

  // ---- blocks --------------------------------------------------------------------------
  function renderParagraph(b) {
    return el("p", { class: "block" }, b.text);
  }

  function renderList(b) {
    return el(b.ordered ? "ol" : "ul", { class: "block" }, b.items.map(function (item) { return el("li", {}, item); }));
  }

  function renderDidYouKnow(b) {
    return el("aside", { class: "block dyk" }, [
      el("h3", {}, t("didYouKnow")),
      el("p", {}, b.text),
      el("p", { class: "source" }, t("sourcePrefix") + b.source_line)
    ]);
  }

  var uid = 0;
  function nextId(prefix) { uid += 1; return prefix + "-" + uid; }

  function renderStopAndThink(b, gates) {
    var explanationId = nextId("explanation");
    var gate = { done: false };
    gates.thinks.push(gate);
    var explanation = el("p", { class: "explanation", id: explanationId, hidden: true }, b.explanation);
    var button = el("button", { type: "button", class: "secondary", "aria-expanded": "false", "aria-controls": explanationId }, t("showExplanation"));
    button.addEventListener("click", function () {
      explanation.hidden = false;
      button.setAttribute("aria-expanded", "true");
      button.hidden = true;
      gate.done = true;
      explanation.setAttribute("tabindex", "-1");
      explanation.focus();
      gates.refresh();
    });
    return el("section", { class: "block think" }, [
      el("h3", {}, t("stopAndThink")),
      el("p", {}, b.prediction_question),
      button,
      explanation
    ]);
  }

  function renderQuestion(b, gates) {
    var q = versionOf(b);
    var name = "q-" + b.id + "-r" + state.round;
    var feedback = el("p", { class: "feedback", "aria-live": "polite" });
    var optionNodes = q.options.map(function (text, i) {
      var input = el("input", { type: "radio", name: name, value: String(i) });
      return el("label", { class: "option" }, [input, el("span", {}, text)]);
    });
    var check = el("button", { type: "button" }, t("check"));
    var fieldset = el("fieldset", { class: "block mc" }, [
      el("legend", {}, [el("span", { class: "qnum" }, t("questionN", { n: questionNumber(b.id) })), q.question])
    ].concat(optionNodes, [check, feedback]));

    function lock(chosen) {
      optionNodes.forEach(function (label, i) {
        var input = label.querySelector("input");
        input.checked = i === chosen;
        input.disabled = true;
        // Only the learner's own choice is marked. The correct option is never revealed.
        if (i === chosen) label.classList.add("chosen");
      });
      check.hidden = true;
      var answer = state.answers[b.id];
      feedback.className = "feedback " + (answer.correct ? "correct" : "incorrect");
      feedback.textContent = answer.correct ? q.feedback_correct : q.feedback_incorrect;
    }

    check.addEventListener("click", function () {
      var picked = fieldset.querySelector("input:checked");
      if (!picked) {
        feedback.className = "feedback info";
        feedback.textContent = t("chooseFirst");
        return;
      }
      var chosen = Number(picked.value);
      state.answers[b.id] = { chosen: chosen, correct: chosen === q.correct_index };
      lock(chosen);
      gates.refresh();
    });

    if (state.answers[b.id]) lock(state.answers[b.id].chosen); // coming back to an answered screen
    gates.questions.push(function () { return !!state.answers[b.id]; });
    return fieldset;
  }

  function renderIllustration(b) {
    return el("figure", { class: "block illustration" }, [
      el("iframe", {
        src: b.file,
        title: t("illustrationLabel", { title: b.title }),
        sandbox: "allow-scripts",
        height: String(b.height || 320)
      }),
      el("figcaption", {}, b.description)
    ]);
  }

  // ---- decision diagram (SVG) ------------------------------------------------------------
  function wrapLabel(text, maxChars) {
    var words = text.split(/\s+/), lines = [], line = "";
    words.forEach(function (w) {
      if (line && (line + " " + w).length > maxChars) { lines.push(line); line = w; }
      else line = line ? line + " " + w : w;
    });
    if (line) lines.push(line);
    return lines;
  }

  function renderDiagram(b) {
    var byId = {};
    b.nodes.forEach(function (n) { byId[n.id] = n; });
    var incoming = {};
    b.edges.forEach(function (e) { incoming[e.to] = (incoming[e.to] || 0) + 1; });

    // Layer nodes top-down by longest path from a start node.
    var level = {};
    b.nodes.forEach(function (n) { if (!incoming[n.id]) level[n.id] = 0; });
    for (var pass = 0; pass < b.nodes.length; pass++) {
      b.edges.forEach(function (e) {
        if (level[e.from] !== undefined && (level[e.to] === undefined || level[e.to] < level[e.from] + 1)) {
          level[e.to] = level[e.from] + 1;
        }
      });
    }
    var layers = [];
    b.nodes.forEach(function (n) {
      var lv = level[n.id] || 0;
      (layers[lv] = layers[lv] || []).push(n);
    });

    // Draw at the real width (1 SVG unit = 1 CSS pixel), so text stays readable on a phone.
    var W = Math.round(Math.max(280, Math.min(720, (main.clientWidth || 640) - 32)));
    var FONT = 15, LINE = 19, gapY = 56, pad = 8, boxPad = 12;
    var lines = {}, layerHeights = [];
    layers.forEach(function (layer, lv) {
      var nodeW = Math.min(280, (W - pad * (layer.length + 1)) / layer.length);
      var maxLines = 1;
      layer.forEach(function (n) {
        // About 0.55em per character at this font size.
        lines[n.id] = wrapLabel(n.label, Math.max(8, Math.floor((nodeW - 2 * boxPad) / (FONT * 0.55))));
        maxLines = Math.max(maxLines, lines[n.id].length);
      });
      layerHeights[lv] = { nodeW: nodeW, h: maxLines * LINE + 2 * boxPad };
    });
    var pos = {}, y = pad;
    layers.forEach(function (layer, lv) {
      var nodeW = layerHeights[lv].nodeW, h = layerHeights[lv].h;
      var used = layer.length * nodeW + (layer.length - 1) * pad;
      layer.forEach(function (n, i) {
        // Right-to-left: the first node of a layer is on the right; a layer is centered.
        var x = W - (W - used) / 2 - (i + 1) * nodeW - i * pad;
        pos[n.id] = { x: x, y: y, w: nodeW, h: h };
      });
      y += h + gapY;
    });
    var H = y - gapY + pad;

    var titleId = nextId("diagram-title"), descId = nextId("diagram-desc");
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-labelledby": titleId, "aria-describedby": descId });
    var defs = svgEl("defs");
    var markerId = nextId("arrow");
    var marker = svgEl("marker", { id: markerId, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "8", markerHeight: "8", orient: "auto-start-reverse" });
    marker.appendChild(svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: P["muted"] }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    b.edges.forEach(function (e) {
      var a = pos[e.from], c = pos[e.to];
      var x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = c.x + c.w / 2, y2 = c.y;
      var midY = (y1 + y2) / 2;
      svg.appendChild(svgEl("path", {
        d: "M" + x1 + "," + y1 + " C" + x1 + "," + midY + " " + x2 + "," + midY + " " + x2 + "," + (y2 - 2),
        fill: "none", stroke: P["muted"], "stroke-width": "2", "marker-end": "url(#" + markerId + ")"
      }));
      if (e.label) {
        var lx = (x1 + x2) / 2, ly = midY;
        svg.appendChild(svgEl("rect", { x: lx - 28, y: ly - 14, width: 56, height: 26, rx: 6, fill: P["background"], stroke: P["border"] }));
        var label = svgEl("text", { x: lx, y: ly + 5, "text-anchor": "middle", "font-size": String(FONT), fill: P["text"], direction: "rtl", "unicode-bidi": "embed" });
        label.textContent = e.label;
        svg.appendChild(label);
      }
    });

    b.nodes.forEach(function (n) {
      var p = pos[n.id];
      var isQuestion = n.kind === "question";
      svg.appendChild(svgEl("rect", {
        x: p.x, y: p.y, width: p.w, height: p.h, rx: isQuestion ? 24 : 10,
        fill: isQuestion ? P["surface"] : P["primary"],
        stroke: isQuestion ? P["primary"] : "none", "stroke-width": "2"
      }));
      var nodeLines = lines[n.id];
      var text = svgEl("text", {
        x: p.x + p.w / 2, "text-anchor": "middle", "font-size": String(FONT), direction: "rtl",
        "unicode-bidi": "embed", fill: isQuestion ? P["text"] : P["primary_text"]
      });
      var startY = p.y + p.h / 2 - (nodeLines.length - 1) * LINE / 2 + FONT * 0.35;
      nodeLines.forEach(function (line, i) {
        var tspan = svgEl("tspan", { x: p.x + p.w / 2, y: startY + i * LINE });
        tspan.textContent = line;
        text.appendChild(tspan);
      });
      svg.appendChild(text);
    });

    // Text alternative for screen readers: every step of the process.
    var steps = el("ol", { class: "sr-only", id: descId }, b.edges.map(function (e) {
      return el("li", {}, e.label
        ? t("diagramEdgeLabeled", { from: byId[e.from].label, label: e.label, to: byId[e.to].label })
        : t("diagramEdge", { from: byId[e.from].label, to: byId[e.to].label }));
    }));
    return el("figure", { class: "block diagram" }, [el("h3", { id: titleId }, b.title), svg, steps]);
  }

  var RENDERERS = {
    paragraph: renderParagraph,
    list: renderList,
    did_you_know: renderDidYouKnow,
    stop_and_think: renderStopAndThink,
    multiple_choice: renderQuestion,
    illustration: renderIllustration,
    decision_diagram: renderDiagram
  };

  // ---- screens ---------------------------------------------------------------------------
  function clearMain() {
    if (observer) { observer.disconnect(); observer = null; }
    while (main.firstChild) main.removeChild(main.firstChild);
    window.scrollTo(0, 0);
  }

  function renderScreen() {
    clearMain();
    var screen = L.screens[state.index];
    var isLast = state.index === L.screens.length - 1;
    setProgress(state.index + 1, t("screenOf", { n: state.index + 1, total: L.screens.length }));

    var gates = { scrolled: false, thinks: [], questions: [], refresh: function () {} };
    var heading = el("h2", { id: "screen-title", tabindex: "-1" }, screen.title);
    var article = el("article", { class: "screen", "aria-labelledby": "screen-title" }, [
      screen.topic ? el("p", { class: "topic" }, screen.topic) : null,
      heading
    ]);
    screen.blocks.forEach(function (b) {
      var render = RENDERERS[b.type];
      if (render) article.appendChild(render(b, gates));
    });
    var sentinel = el("div", { class: "end-of-screen", "aria-hidden": "true" });
    article.appendChild(sentinel);

    var hint = el("p", { class: "hint", id: "next-hint", "aria-live": "polite" });
    var next = el("button", { type: "button", id: "next", "aria-describedby": "next-hint" }, isLast ? t("finish") : t("next"));
    var navChildren = [];
    if (state.index > 0) {
      var prev = el("button", { type: "button", class: "secondary", id: "prev" }, t("prev"));
      prev.addEventListener("click", function () { state.index -= 1; renderScreen(); });
      navChildren.push(prev);
    }
    navChildren.push(el("span", { class: "spacer" }), next, hint);
    var nav = el("nav", { class: "nav", "aria-label": t("navLabel") }, navChildren);

    function missing() {
      if (!gates.scrolled) return "hintScroll";
      if (gates.thinks.some(function (g) { return !g.done; })) return "hintThink";
      if (gates.questions.some(function (done) { return !done(); })) return "hintQuestion";
      return null;
    }
    gates.refresh = function () {
      var reason = missing();
      next.setAttribute("aria-disabled", reason ? "true" : "false");
      hint.textContent = reason ? t(reason) : "";
    };
    next.addEventListener("click", function () {
      if (missing()) { gates.refresh(); return; }
      if (isLast) { renderResult(); return; }
      state.index += 1;
      renderScreen();
    });

    main.appendChild(article);
    main.appendChild(nav);
    gates.refresh();

    // The end of the screen counts as seen once it enters the viewport (after scrolling).
    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver(function (entries) {
        if (entries.some(function (e) { return e.isIntersecting; })) {
          gates.scrolled = true;
          gates.refresh();
          observer.disconnect();
        }
      });
      observer.observe(sentinel);
    } else {
      gates.scrolled = true;
      gates.refresh();
    }
    heading.focus();
  }

  // ---- result ----------------------------------------------------------------------------
  function renderResult() {
    clearMain();
    setProgress(L.screens.length, t("summaryTitle"));
    var total = questions.length;
    var correct = questions.filter(function (q) { var a = state.answers[q.block.id]; return a && a.correct; }).length;
    var passing = typeof L.passing_score === "number" ? L.passing_score : DEFAULT_PASSING;
    var needed = Math.ceil(passing * total - 1e-9);
    var passed = total === 0 || correct >= needed;

    var heading = el("h2", { tabindex: "-1" }, t("summaryTitle"));
    var section = el("article", { class: "screen" }, [
      heading,
      el("ul", { class: "block" }, L.summary.points.map(function (p) { return el("li", {}, p); }))
    ]);

    var box = el("section", { class: "result-box", "aria-labelledby": "result-title" }, [el("h3", { id: "result-title" }, t("resultTitle"))]);
    if (total === 0) {
      box.appendChild(el("p", {}, t("noQuestions")));
    } else if (passed) {
      box.appendChild(el("p", { class: "feedback correct" }, t("passed", { correct: correct, total: total })));
    } else {
      box.appendChild(el("p", { class: "feedback incorrect" }, t("failed", { correct: correct, total: total, needed: needed })));
      // Only which questions were wrong: number, topic and wording. Never the right answers.
      var wrong = questions.filter(function (q) { var a = state.answers[q.block.id]; return !(a && a.correct); });
      box.appendChild(el("p", {}, t("wrongTitle")));
      box.appendChild(el("ul", { class: "wrong-list" }, wrong.map(function (q) {
        return el("li", {}, t("wrongItem", { n: q.number, topic: q.screen.topic || q.screen.title, question: versionOf(q.block).question }));
      })));
      var retry = el("button", { type: "button", id: "retry" }, t("retry"));
      retry.addEventListener("click", function () {
        state.round += 1;
        state.answers = {};
        state.index = 0;
        renderScreen();
      });
      box.appendChild(el("p", { class: "hint" }, t("retryNote")));
      box.appendChild(retry);
    }
    section.appendChild(box);
    if (passed && L.certificate) section.appendChild(renderCertificateForm());
    main.appendChild(section);
    heading.focus();
  }

  function renderCertificateForm() {
    var inputId = nextId("cert-name");
    var input = el("input", { type: "text", id: inputId, autocomplete: "name", maxlength: "60" });
    var status = el("p", { class: "feedback info", "aria-live": "polite" });
    var button = el("button", { type: "button", id: "certificate" }, t("certificateButton"));
    button.addEventListener("click", function () {
      var name = input.value.trim();
      if (!name) { status.textContent = t("certificateNeedName"); input.focus(); return; }
      makeCertificate(name, function () { status.textContent = t("certificateReady"); });
    });
    return el("section", { class: "result-box cert", "aria-labelledby": "cert-title" }, [
      el("h3", { id: "cert-title" }, t("certificateTitle")),
      el("label", { for: inputId }, t("certificateNameLabel")),
      input, button, status
    ]);
  }

  // The certificate is drawn and saved in the browser; nothing is sent anywhere.
  function makeCertificate(name, done) {
    var css = getComputedStyle(document.documentElement);
    function color(v) { return css.getPropertyValue("--" + v).trim(); }
    var canvas = document.createElement("canvas");
    canvas.width = 1600; canvas.height = 1130;
    var g = canvas.getContext("2d");
    g.fillStyle = color("background"); g.fillRect(0, 0, canvas.width, canvas.height);
    g.strokeStyle = color("primary"); g.lineWidth = 16; g.strokeRect(48, 48, canvas.width - 96, canvas.height - 96);
    g.direction = "rtl"; g.textAlign = "center"; g.fillStyle = color("text");
    var font = '"Segoe UI", Arial, sans-serif';
    var cx = canvas.width / 2;
    g.font = "bold 80px " + font; g.fillText(t("certificateTitle"), cx, 270);
    g.font = "44px " + font; g.fillText(t("certificateGrantedTo"), cx, 420);
    g.font = "bold 72px " + font; g.fillStyle = color("primary"); g.fillText(name, cx, 520);
    g.fillStyle = color("text"); g.font = "44px " + font; g.fillText(t("certificateFor"), cx, 640);
    g.font = "bold 56px " + font; g.fillText(L.title, cx, 730);
    g.font = "36px " + font; g.fillStyle = color("muted");
    g.fillText(t("certificateDate", { date: new Date().toLocaleDateString("he-IL") }), cx, 900);
    canvas.toBlob(function (blob) {
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = t("certificateFile");
      document.body.appendChild(a);
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
      if (done) done();
    }, "image/png");
  }

  renderScreen();
})();
