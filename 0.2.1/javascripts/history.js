import { layoutAxisLabels } from "./history-axis.mjs";
import { commandText } from "./history-text.mjs";

// The history section (docs/history/): the day strip and the four drawings of the
// story page, the owner's messages and the conversations. Material swaps the page
// content without running this file again, so every mount starts from its document$,
// which also emits for the first page.
(() => {
  const DATA = new URL("../history/data/", import.meta.url);
  const GH = "https://github.com/yaugenst/treams-rs/commit/";
  const ABOUT = "../about/#what-is-left-out";
  const TITLE = "history-title";
  const SEARCH = "search"; // not "q": Material opens its own search for ?q=
  const DAY = 86400;

  // One request per file and page load; a failed request is dropped, so "Try again" fetches it again.
  const cache = new Map();
  const load = (name) => {
    if (!cache.has(name)) {
      const p = fetch(new URL(name, DATA)).then((r) => {
        if (!r.ok) throw new Error(`${name}: ${r.status}`);
        return r.json();
      });
      p.catch(() => cache.delete(name));
      cache.set(name, p);
    }
    return cache.get(name);
  };
  let day0 = 0; // the start of day 1, the day of the first message
  const loadIndex = () => load("index.json").then((x) => ((day0 = x.day1), x));

  // put(el, ...kids) replaces the children: arrays are flattened, null, false and "" skipped.
  // h("p.a.b", {attribute: value, onclick: f, html: the build's sanitized agent HTML}, ...kids).
  const put = (e, ...kids) => (e.replaceChildren(...kids.flat(9).filter((k) => k != null && k !== false && k !== "")), e);
  const h = (tag, props, ...kids) => {
    const [name, ...cls] = tag.split(".");
    const e = document.createElement(name);
    if (cls.length) e.className = cls.join(" ");
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e[k] = v;
      else e.setAttribute(k, v === true ? "" : v);
    }
    return kids.length ? put(e, ...kids) : e;
  };
  const button = (props, ...kids) => h("button.link", { type: "button", ...props }, ...kids);

  // Days, numbers and durations.
  const dn = (s) => 1 + Math.floor((s - day0) / DAY);
  const when = (s) => `day ${dn(s)}`;
  const fromTo = (a, b) => (dn(a) === dn(b) ? when(a) : `${when(a)} to ${when(b)}`);
  const span = (a, b) => (a === b ? `day ${a}` : `days ${a}–${b}`);
  const num = (n) => n.toLocaleString("en-GB");
  const plural = (n, one, many = `${one}s`) => `${num(n)} ${n === 1 ? one : many}`;
  const dur = (minutes) => {
    const m = Math.round(minutes);
    return m < 1 ? "under a minute" : m < 60 ? `${m}\u00a0min` : m >= 600 ? `${Math.round(m / 60)}\u00a0h` // no break inside "1 h"
      : `${Math.floor(m / 60)}\u00a0h${m % 60 ? ` ${m % 60}\u00a0min` : ""}`;
  };
  const minutes = (a, b) => (b - a) / 60;
  const join = (parts) => parts.filter(Boolean).join(" · ");
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const effort = (m) => [m.model, m.effort && `effort level ${m.effort}`].filter(Boolean).join(", ");
  const tally = (x) => [x.commands && `${plural(x.commands, "command")} run`, x.files && `${plural(x.files, "file")} changed`,
    x.notes && plural(x.notes, "progress note"), x.helpers && plural(x.helpers, "helper agent"),
    x.groups && plural(x.groups, "planned group"), x.searches && plural(x.searches, "web search", "web searches"),
    x.changes && `${plural(x.changes, "change")} on GitHub`];
  const replies = (n) => n > 0 && (n === 1 ? "1 reply" : `${num(n)} replies and progress notes`);
  const helperName = (x) => (x.nick ? `Helper agent ${x.nick}` : cap(x.task));
  const quietText = (p) => plural(p.to - p.from + 1, "quiet day");

  // Days with work get wt(day) units of width (default one); a run of quiet days becomes one narrow gap.
  const compress = (list, wt = () => 1, gap = 0.6) => {
    const parts = [], of = {};
    let w = 0;
    for (const d of list) {
      const last = parts.at(-1);
      if (d.quiet && last?.gap) last.to = d.n;
      else {
        parts.push(d.quiet ? { gap: true, from: d.n, to: d.n, x: w, w: gap } : { ...d, x: w, w: wt(d) });
        w += parts.at(-1).w;
      }
      of[d.n] = parts.at(-1);
    }
    return { parts, of, W: w };
  };

  // Owner text: strings, and {"r": ...} for words replaced to protect privacy.
  const runs = (list) => [].concat(list).map((r) => (typeof r === "string" ? r : h("i.r", { title: "Replaced to protect privacy" }, r.r)));
  const plain = (list) => [].concat(list).map((r) => r.r ?? r).join("");

  // Clamps an element to `lines` lines. Once it is laid out (hidden and not yet rendered
  // parts wait), a button after it shows the rest if there is more; so does focus inside it.
  let observer;
  const clamp = (el, lines, label) => {
    el.classList.add("clamp");
    el.style.setProperty("--lines", lines);
    el.dataset.more = label;
    observer.observe(el);
    return el;
  };
  const measure = (entries) => {
    for (const { target: el } of entries) {
      if (!el.clientHeight) continue;
      observer.unobserve(el);
      if (el.scrollHeight <= el.clientHeight + 2) {
        el.classList.remove("clamp");
        continue;
      }
      const n = Math.round(el.scrollHeight / parseFloat(getComputedStyle(el).lineHeight));
      const open = () => (el.classList.remove("clamp"), b.remove());
      const b = button({ onclick: () => (open(), (el.tabIndex = -1), el.focus()) }, el.dataset.more.replace("{n}", num(n)));
      el.after(b);
      el.addEventListener("focusin", open, { once: true });
    }
  };

  // Shows `text` in `box` if the promise takes longer than 300 ms, and the error with
  // "Try again" if it fails (it then resolves to undefined).
  const wait = (box, text, promise, error, retry) => {
    const t = setTimeout(() => put(box, text), 300);
    return promise.finally(() => clearTimeout(t)).catch(() => void put(box, `${error} `, button({ onclick: retry }, "Try again")));
  };
  const debounce = (f, ms) => {
    let t;
    return () => (clearTimeout(t), (t = setTimeout(f, ms)));
  };
  // A new view gets its own history entry (push), so Back returns to it; filters replace theirs.
  const setQuery = (params, push) => {
    const qs = String(new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== ""))).replaceAll("%2F", "/");
    history[push ? "pushState" : "replaceState"](history.state, "", qs ? `?${qs}` : location.pathname);
  };
  const phoneWidth = () => matchMedia("(max-width: 44.984em)").matches;
  const coarse = matchMedia("(pointer: coarse)");

  // Story page: the day strip ---------------------------------------------------------------

  function days(root, signal) {
    const list = JSON.parse(root.dataset.days), phases = JSON.parse(root.dataset.phases);
    const { parts, of, W } = compress(list, undefined, phoneWidth() ? 1 : 0.6), cells = parts.filter((p) => !p.gap), n = cells.length;
    const max = Math.max(1, ...list.map((d) => d.owner)), most = list.find((d) => d.owner === max);
    const about = (d) => join([phases.find((p) => p.id === d.phase)?.title, `agents working ${num(d.hours)} of 24 hours`,
      `${plural(d.owner, "message")} from the human`, plural(d.conversations, "conversation")]);
    const pct = (x) => `${((x / W) * 100).toFixed(3)}%`;
    const bar = (f) => f > 0 && h("i", { style: `height:max(2px,${(f * 100).toFixed(1)}%)` });
    const cols = h("div.cols", { role: "group", "aria-label": "Days", style: `grid-template-columns:${parts.map((p) => `${p.w}fr`).join(" ")}` },
      parts.map((p) => (p.gap ? h("span.gap", null, h("span", null, quietText(p)))
        : h("button", { type: "button", "aria-pressed": "false", tabindex: p === cells[0] ? 0 : -1, "aria-label": `Day ${p.n}: ${about(p)}` },
          h("span.a", null, bar(p.hours / 24)), h("span.o", null, bar(p.owner / max))))));
    const rules = h("div.rules", { "aria-hidden": "true" }, phases.map((p, k) => {
      const a = of[p.day_from].x, b = of[p.day_to].x + of[p.day_to].w;
      return [h("b", { style: `left:calc(${pct(a)} + 1px);width:calc(${pct(b - a)} - 2px)` }), h(`span.l${k % 2}`, { style: `left:calc(${pct(a)} + 1px)` }, p.title)];
    }));
    // Day numbers at the start of each phase, under the bars.
    const nums = h("div.nums", { "aria-hidden": "true" }, phases.map((p) => h("span", { style: `left:${pct(of[p.day_from].x)}` }, `Day ${p.day_from}`)));
    const text = h("span"); // not live: it follows the pointer; the day buttons carry the same text in their names
    const buttons = [...cols.querySelectorAll("button")];
    let sel = -1;
    const show = (i) => {
      const d = cells[i];
      put(text, i < 0 ? "Select a day for its numbers." : [h("b", null, `Day ${d.n}`), " · ", about(d),
        " · ", h("a", { href: `conversations/?day=${d.n}` }, "Open this day")]);
    };
    const select = (i, focus) => {
      sel = Math.max(0, Math.min(n - 1, i));
      buttons.forEach((b, k) => {
        b.setAttribute("aria-pressed", k === sel);
        b.tabIndex = k === sel ? 0 : -1;
      });
      if (focus) buttons[sel].focus();
      show(sel);
    };
    const at = (e) => buttons.indexOf(e.target.closest("button"));
    const cur = () => (buttons.includes(document.activeElement) ? buttons.indexOf(document.activeElement) : sel); // keys start at the focused day
    const step = (k) => () => select(cur() < 0 ? (k > 0 ? 0 : n - 1) : cur() + k);
    cols.onclick = (e) => at(e) >= 0 && select(at(e));
    cols.addEventListener("focusin", (e) => at(e) >= 0 && show(at(e))); // there is no onfocusin property
    cols.onpointerover = (e) => {
      const g = e.target.closest(".gap"), p = g && parts[[...cols.children].indexOf(g)];
      if (p) put(text, [h("b", null, cap(span(p.from, p.to))), ` · ${quietText(p)}`]);
      else if (at(e) >= 0) show(at(e));
    };
    cols.onpointerleave = () => show(sel);
    cols.onkeydown = (e) => {
      const k = { ArrowLeft: cur() - 1, ArrowRight: cur() + 1, Home: 0, End: n - 1 }[e.key];
      if (k === undefined || e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      select(k, true);
    };
    put(root, h("p.keys", { "aria-hidden": "true" }, h("span", null, h("i"), "Hours with agents working (of 24)"),
      h("span", null, h("i.o"), `Messages from the human (most: ${most.owner} on day ${most.n})`)), cols, nums, rules,
    h("p.history-readout", null, text, h("span", null,
      h("button", { type: "button", "aria-label": "Previous day", onclick: step(-1) }, "‹"),
      h("button", { type: "button", "aria-label": "Next day", onclick: step(1) }, "›"))));
    show(-1);
    // A phase label starts at its rule. If that runs into the next label on its line or past
    // the row end, it ends with its rule instead; if that overlaps too, it moves to a third
    // line, and only if that is taken too, it is hidden.
    const labels = [...rules.querySelectorAll("span")];
    const fit = () => {
      for (const l of labels) (l.style.translate = ""), (l.style.visibility = ""), l.classList.remove("l2");
      rules.classList.remove("three");
      const end = [-1e9, -1e9, -1e9], width = rules.clientWidth;
      labels.forEach((l, k) => {
        const w = l.offsetWidth, r = l.previousElementSibling, next = labels[k + 2]?.offsetLeft ?? width + 8;
        let x = l.offsetLeft, line = k % 2;
        if (x + w + 8 > next) x = Math.max(0, Math.min(width, r.offsetLeft + r.offsetWidth) - w);
        if (x < end[line] + 8 || x + w + 8 > next) {
          (x = Math.max(0, Math.min(l.offsetLeft, width - w))), (line = 2);
          if (x < end[2] + 8) return void (l.style.visibility = "hidden");
          l.classList.add("l2");
          rules.classList.add("three");
        }
        l.style.translate = `${x - l.offsetLeft}px`;
        end[line] = x + w;
      });
    };
    requestAnimationFrame(fit);
    document.fonts?.ready.then(fit);
    addEventListener("resize", fit, { signal });
    declutter(nums);
  }

  // Story page: how the work was split (a schematic drawing above each list item) ------------
  // The marks are the ones of the maps: the owner (accent), an agent (ink), helper agents
  // (small grey dots), planned groups (stage columns of dots), other conversations (hollow).

  const SVG = "http://www.w3.org/2000/svg";
  const add = (parent, tag, attrs) => {
    const e = parent.appendChild(document.createElementNS(SVG, tag));
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };
  const DRAWINGS = [
    (s) => { s.l(80, 12, 80, 52, "own"); s.a(80, 52); },
    (s) => {
      s.l(80, 12, 80, 40, "own");
      for (let k = 0; k < 12; k++) {
        const x = 20 + (k * 120) / 11;
        s.l(80, 40, x, 72);
        s.d(x, 72);
        if (k % 4 === 1) for (const dx of [-4, 4]) { s.l(x, 72, x + dx, 96); s.d(x + dx, 96, 1.5); }
      }
      s.a(80, 40);
    },
    (s) => {
      s.l(80, 12, 80, 30, "own");
      s.l(80, 30, 80, 38);
      s.l(30, 38, 130, 38);
      ["make", "check", "fix"].forEach((t, c) => {
        const x = 30 + c * 50;
        s.l(x, 38, x, 44);
        add(s.dots, "text", { x, y: 58, "text-anchor": "middle" }).textContent = t;
        for (const y of [70, 82, 94]) s.d(x, y, 2.5);
      });
      s.a(80, 30);
    },
    (s) => {
      for (const x of [40, 80, 120]) {
        s.l(80, 12, x, 48, "own");
        for (let j = 0; j < 4; j++) { s.l(x, 48, x - 9 + j * 6, 84); s.d(x - 9 + j * 6, 84); }
      }
      s.l(80, 48, 40, 48, "dash");
      s.l(80, 48, 120, 48, "dash");
      s.a(80, 48);
      for (const x of [40, 120]) s.d(x, 48, 4, "ho");
    },
  ];

  function split(root) {
    // A key to the marks, once, before the drawings (the captions carry the meaning for screen readers).
    root.prepend(h("p.key", { "aria-hidden": "true" }, [["ow", "human"], ["ag", "agent"], ["hp", "helper agent"], ["ho", "another conversation"]]
      .map(([c, t]) => h("span", null, h(`i.${c}`), t))));
    root.querySelectorAll(":scope > ul > li").forEach((li, k) => {
      const svg = document.createElementNS(SVG, "svg");
      svg.setAttribute("viewBox", "0 0 160 104");
      svg.setAttribute("aria-hidden", "true");
      const lines = add(svg, "g"), dots = add(svg, "g");
      const s = {
        dots,
        l: (x1, y1, x2, y2, c = "") => add(lines, "line", { x1, y1, x2, y2, class: c }),
        d: (cx, cy, r = 2, c = "hp") => add(dots, "circle", { cx, cy, r, class: c }),
        a: (cx, cy) => s.d(cx, cy, 4.5, "ag"),
      };
      DRAWINGS[k]?.(s);
      s.d(80, 12, 5, "ow");
      li.prepend(svg);
    });
  }

  // The shape of a piece of work as a small glyph in the same marks: the agent, a fan of up
  // to 4 helper agents, a planned group as up to 4 stage columns of 3 dots (hollow: stopped
  // or failed), and last another conversation (hollow ink dot).
  const glyph = ({ fan = 0, stages, pair }) => {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("class", "g");
    s.setAttribute("aria-hidden", "true");
    let end = 7;
    const dot = (cx, cy, r, c) => ((end = Math.max(end, cx + r)), add(s, "circle", { cx, cy, r, class: c }));
    const n = Math.min(fan, 4);
    for (let k = 0; k < n; k++) {
      const cy = 8 + (k - (n - 1) / 2) * 4;
      add(s, "line", { x1: 4, y1: 8, x2: 15, y2: cy });
      dot(15, cy, 1.2, "hp");
    }
    const x0 = end + 4.5;
    (stages || []).slice(0, 4).forEach((col, c) => col.slice(0, 3).forEach((ok, r) =>
      dot(x0 + c * 4.6, 8 + (r - (Math.min(col.length, 3) - 1) / 2) * 4.6, 1.4, ok ? "hp" : "hx")));
    if (pair && end === 7) add(s, "line", { x1: 4, y1: 8, x2: 14, y2: 8 });
    if (pair) dot(end === 7 ? 15 : end + 5, 8, 2.3, "ho");
    dot(4, 8, 2.8, "ag");
    s.setAttribute("viewBox", `0 0 ${end + 1} 16`);
    s.style.width = `${((end + 1) * 0.05).toFixed(2)}rem`;
    return s;
  };
  const dot = (c) => h(`i.dot.${c}`); // hp: a helper agent; ho: another conversation
  const GRID = [[1, 1, 1], [1, 1, 1], [1, 1, 1]];

  // The owner's messages ----------------------------------------------------------------------

  const ownerText = (o, lines) => [
    lines ? clamp(h("p.text", null, runs(o.text)), lines, "Show the whole message ({n} lines)") : h("p.text", null, runs(o.text)),
    o.images > 0 && h("p.label", null, `${plural(o.images, "image")} attached, not published`),
  ];
  const typed = (o) => o.kind !== "answer";

  async function messages(root, signal, again) {
    const p = new URLSearchParams(location.search);
    const q = h("input", { type: "search", placeholder: "Search the messages", "aria-label": "Search the messages", value: p.get(SEARCH) });
    const phase = h("select", { "aria-label": "Phase" }, h("option", { value: "" }, "All phases"));
    const order = h("select", { "aria-label": "Order" }, h("option", { value: "" }, "In order"), h("option", { value: "longest" }, "Longest work first"));
    const count = h("p.count", { "aria-live": "polite" });
    const list = h("div");
    root.classList.add("history-app", "history-messages");
    put(root, h("form", { role: "search", onsubmit: (e) => e.preventDefault() }, q, phase, order), count, list);
    const index = await wait(list, "Loading the messages…", loadIndex(), "The messages could not be loaded.", () => messages(root, signal, true));
    if (!index || signal.aborted) return;
    for (const x of index.phases) phase.append(h("option", { value: x.id }, x.title));
    phase.value = p.get("phase") || "";
    if (phase.selectedIndex < 0) phase.value = "";
    order.value = p.get("order") === "longest" ? "longest" : "";
    if (again) q.focus();
    const titles = Object.fromEntries(index.conversations.map((c) => [c.id, c.title]));
    const phases = Object.fromEntries(index.phases.map((x) => [x.id, x]));
    const quiet = (a, b) => index.days.filter((d) => d.quiet && d.n > a && d.n < b).length;
    const total = index.figures.owner_messages;

    const reply = async (t, b, box) => {
      if (b.getAttribute("aria-disabled")) return;
      const open = box.hidden;
      b.setAttribute("aria-expanded", open);
      b.textContent = open ? "Hide the reply" : "Show the reply";
      box.hidden = !open;
      if (!open || box.dataset.ok) return;
      b.setAttribute("aria-disabled", "true");
      const file = await wait(box, "Loading the reply…", load(`c/${t.c}.json`), "The reply could not be loaded.",
        () => ((box.hidden = true), reply(t, b, box).then(() => b.focus())));
      b.removeAttribute("aria-disabled");
      if (!file) return;
      const it = file.items.find((i) => i.t === "agent" && i.reply && i.at === t.reply_at);
      box.dataset.ok = 1;
      const html = it && h("div.html", { html: it.html });
      put(box, it ? [h("p.label", null, dn(it.at) === dn(t.at) ? "Agent" : `Agent, ${when(it.at)}`), clamp(html, 8, `Show the whole reply (${num(html.textContent.split(/\s+/).filter(Boolean).length)} words)`)]
        : "The reply could not be loaded.");
    };
    const level = {}; // the last effort level per conversation: it is written only when it changes
    const turns = index.turns.map((t) => {
      const [first, ...more] = t.owner, effortText = t.effort && t.effort !== level[t.c] && `effort level ${t.effort}`;
      level[t.c] = t.effort;
      const box = h("div.reply", { hidden: true }), adds = [];
      const b = t.reply_at != null && button({ "aria-expanded": "false", onclick: () => reply(t, b, box) }, "Show the reply");
      const waited = t.reply_at != null ? minutes(t.at, t.reply_at) : -1;
      const node = h("article.turn", null, h("span.t"), h("div", null,
        h("div.msg", null, ownerText(first, 12)),
        more.map((o) => {
          if (o.kind === "answer") return [h("p.label", null, `Agent asked: ${o.question}`), h("div.msg.added", null, h("p.text", null, "Answered: ", runs(o.text)))];
          const el = h("div.msg.added", null, ownerText(o, 0), h("p.label", null, "While the agent worked"));
          adds.push([o, el]);
          return el;
        }),
        h("p.meta", null, join([waited < 0 ? "no reply before the next message" : waited < 1 ? "reply within a minute" : `reply after ${dur(waited)}`,
          ...tally(t.work || {}), effortText]), " · ", b, b && " · ",
        // Back from the conversation returns to this message: the current entry remembers it.
        h("a", { href: `../conversations/?c=${t.c}&at=${first.at}`, onclick: () => setQuery({ ...filters(), at: first.at }) }, "In the conversation")),
        box));
      return { t, node, adds, time: node.firstChild, text: t.owner.map((o) => plain(o.text)).join("\n").toLowerCase(), wait: waited };
    });

    const render = () => {
      if (signal.aborted) return;
      const raw = q.value.trim(), s = raw.toLowerCase(), ph = phase.value, longest = order.value;
      const hit = (o) => typed(o) && (!s || plain(o.text).toLowerCase().includes(s));
      const shown = turns.filter((x) => (!ph || x.t.phase === ph) && (!s || x.text.includes(s)));
      if (longest) shown.sort((a, b) => b.wait - a.wait);
      list.classList.toggle("longest", !!longest);
      put(list);
      let sec = list, lastPhase, lastDay, lastN, lastC;
      for (const { t, node, time, adds } of shown) {
        const n = dn(t.at);
        if (!longest) {
          const gap = lastN && n !== lastN ? quiet(lastN, n) : 0;
          if (gap) sec.append(h("p.sep.quiet", null, plural(gap, "quiet day")));
          if (t.phase !== lastPhase) {
            const x = phases[(lastPhase = t.phase)];
            list.append((sec = h("section", null, h("h2", null, `${x.title} · ${span(x.day_from, x.day_to)}`))));
            lastDay = null;
          }
          if (n !== lastDay) {
            sec.append(h("h3.sep", null, `Day ${(lastDay = n)}`));
            lastC = null;
          }
          lastN = n;
        }
        if (t.c !== lastC) sec.append(h("a.conv", { href: `../conversations/?c=${(lastC = t.c)}` }, titles[t.c]));
        time.textContent = longest ? when(t.at) : "";
        for (const [o, el] of adds) el.classList.toggle("miss", !!s && !hit(o));
        sec.append(node);
      }
      const n = shown.reduce((a, x) => a + x.t.owner.filter(hit).length, 0);
      put(count, s || ph ? [`${num(n)} of ${plural(total, "message")} · `, button({ onclick: () => ((q.value = phase.value = ""), render(), q.focus()) }, "Clear filters")]
        : `${plural(total, "message")} · about ${num(index.figures.owner_words)} words`);
      if (!shown.length && s) {
        list.append(h("p.status", null, ph ? [`No messages contain "${raw}" in this phase. `,
          button({ onclick: () => ((phase.value = ""), render(), q.focus()) }, "Search all phases")] : `No messages contain "${raw}".`));
      }
      setQuery(filters());
    };
    const filters = () => ({ phase: phase.value, [SEARCH]: q.value.trim(), order: order.value });
    q.oninput = debounce(render, 150);
    phase.onchange = order.onchange = render;
    render();
    const at = p.get("at");
    if (at) turns.find((x) => String(x.t.owner[0].at) === at)?.node.scrollIntoView({ block: "center" });
  }

  // Maps -------------------------------------------------------------------------------------
  // Rows of a label and a time track; marks are links placed in % of the track, and one SVG
  // overlay draws the hairlines (x in %, y in rows: every row has the same height). Under it a
  // readout line (no tooltips); one tab stop, arrows move within and between rows.

  // On touch screens the first tap on a mark shows its details, a second tap opens it.
  const hint = () => (coarse.matches ? "Tap a mark for its details, and again to open it." : "Point at a mark, or tab in and use the arrow keys, for its details.");
  const pos = (x, w) => `left:${(x * 100).toFixed(2)}%${w != null ? `;width:${(w * 100).toFixed(2)}%` : ""}`;
  const pin = (cls, href, label, x, w, style = "") => h(`a.${cls}`, { href, "aria-label": label, tabindex: -1, "data-x": x, style: pos(x, w) + style });
  const LEGEND = {
    tick: () => h("i.tick", { style: "left:50%" }), bar: () => h("i.bar", { style: "left:0;width:100%" }),
    hbar: () => h("i.hbar", { style: "left:0;width:100%" }), hd: () => h("i.hd", { style: "left:50%;top:50%" }),
    grp: () => h("i.grp", { style: "left:0;width:100%" }, h("i"), h("i"), h("i")),
    ring: () => h("i.ring", { style: "left:50%" }), link: () => h("i.curve", { style: "left:30%" }),
  };
  const legend = (keys) => h("p.legend", { "aria-hidden": "true" }, keys.map(([k, text]) => h("span", null, h("span.tr", null, LEGEND[k]()), text)));
  // Labels of an axis that would overlap one already placed are hidden (again on every resize);
  // break labels (.mid) are placed first, so a gap keeps its label.
  const declutter = (box) => new ResizeObserver(() => {
    const right = box.getBoundingClientRect().right + 2, kept = [];
    for (const l of [...box.children].sort((a, b) => b.classList.contains("mid") - a.classList.contains("mid"))) {
      l.style.visibility = "";
      const r = l.getBoundingClientRect();
      if (r.right > right || kept.some((k) => r.left < k.right + 4 && r.right > k.left - 4)) l.style.visibility = "hidden";
      else kept.push(r);
    }
  }).observe(box);

  const fitAxis = (box, axis, signal) => {
    const fit = () => {
      if (signal.aborted || !box.isConnected) return;
      const labels = [...box.children], bounds = box.getBoundingClientRect();
      for (const label of labels) label.style.removeProperty("--axis-shift");
      const measured = labels.map((label, i) => {
        const rect = label.firstElementChild.getBoundingClientRect();
        return { left: rect.left - bounds.left, width: rect.width,
          required: i === 0 || !!axis[i].required,
          priority: axis[i].mid ? 0 : i === axis.length - 1 ? 2 : 1 };
      });
      const placed = layoutAxisLabels(measured, bounds.width);
      labels.forEach((label, i) => {
        const place = placed[i];
        label.style.visibility = place ? "" : "hidden";
        label.style.setProperty("--axis-shift", `${place ? place.left - measured[i].left : 0}px`);
        label.style.setProperty("--axis-row", place?.row || 0);
      });
      box.closest(".map").style.setProperty("--axis-extra", `${Math.max(0, ...placed.map((p) => p?.row || 0))}rem`);
    };
    const resize = new ResizeObserver(fit);
    resize.observe(box);
    document.fonts?.ready.then(fit);
    document.fonts?.addEventListener("loadingdone", fit, { signal });
    signal.addEventListener("abort", () => resize.disconnect(), { once: true });
  };

  // The hairlines span rows `top` to `top + n` (all of the same height).
  const chart = (cls, { label, axis, bands = [], rows, paths = [], top = 0, n = rows.length }, signal) => {
    const readout = h("p.readout", null, hint());
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", `0 0 100 ${n}`);
    svg.setAttribute("preserveAspectRatio", "none");
    if (top || n < rows.length) svg.style.cssText = `top:calc(var(--rh) * ${top});height:calc(var(--rh) * ${n})`;
    for (const d of paths) add(svg, "path", { d });
    const ticks = h("span.tr", null, axis.map((l) => h(`span${l.mid ? ".mid" : ""}`, { style: pos(l.x) }, h("span", null, l.text))));
    const map = h(`div.map.${cls}${bands.some((b) => b.text) ? ".gaps" : ""}`, { role: "group", "aria-label": label }, h("div.mrow.axis", { "aria-hidden": "true" }, h("span"), ticks),
      h("div.body", null,
        h("div.under", { "aria-hidden": "true" }, bands.map((b) => h(`span.band${b.line ? ".line" : ""}`, { style: pos(b.x, b.w) }, b.text && h("span", null, b.text))), svg),
        rows.map((r) => h(`div.mrow${r.cls ? `.${r.cls}` : ""}`, null, typeof r.label === "string" ? h("span", null, r.label) : r.label, h("span.tr", null, r.marks)))));
    const grid = rows.map((r) => [r.label, ...r.marks.filter((m) => m.matches?.("a")).sort((a, b) => a.dataset.x - b.dataset.x)]
      .filter((e) => e?.matches?.("a, button")));
    const all = grid.flat();
    all.forEach((e, k) => (e.tabIndex = k ? -1 : 0));
    const show = (a) => {
      const t = a.getAttribute("aria-label") || a.textContent;
      put(readout, t.endsWith(" · Open") ? [t.slice(0, -4), h("a", { href: a.getAttribute("href") }, "Open")] : t);
    };
    map.move = (a) => {
      for (const e of all) e.tabIndex = -1;
      a.tabIndex = 0;
      a.focus();
    };
    map.addEventListener("focusin", (e) => all.includes(e.target) && show(e.target));
    map.onpointerover = (e) => {
      const a = e.target.closest("a, button");
      if (a && all.includes(a)) show(a);
    };
    let tapped;
    map.addEventListener("click", (e) => {
      const a = e.target.closest("a");
      if (!coarse.matches || !all.includes(a) || a === tapped) return;
      e.preventDefault();
      e.stopPropagation();
      show((tapped = a));
    });
    const near = (row, x) => row.reduce((a, b) => (Math.abs((b.dataset.x ?? -1) - x) < Math.abs((a.dataset.x ?? -1) - x) ? b : a), row[0]);
    map.onkeydown = (e) => {
      const r = grid.findIndex((row) => row.includes(document.activeElement));
      if (r < 0 || e.altKey || e.ctrlKey || e.metaKey) return;
      const row = grid[r], i = row.indexOf(document.activeElement), x = +(document.activeElement.dataset.x ?? -1);
      const moves = { ArrowLeft: row[i - 1], ArrowRight: row[i + 1], Home: row[0], End: row.at(-1) };
      let to = moves[e.key];
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const k = e.key === "ArrowUp" ? -1 : 1;
        for (let j = r + k; j >= 0 && j < grid.length && !to; j += k) to = grid[j].length ? near(grid[j], x) : null;
      } else if (!(e.key in moves)) return;
      e.preventDefault();
      if (to) map.move(to);
    };
    fitAxis(ticks, axis, signal);
    return [map, readout];
  };

  // The axis of one conversation: its active spans; gaps over 2 h become a narrow break.
  const timeAxis = (intervals, phaseStarts) => {
    const segs = [];
    for (const [a, b] of intervals.toSorted((p, q) => p[0] - q[0])) {
      const last = segs.at(-1);
      if (last && a - last[1] <= 7200) last[1] = Math.max(last[1], b);
      else segs.push([a, b]);
    }
    const total = segs.reduce((s, [a, b]) => s + b - a, 0) || 1;
    const ws = segs.map(([a, b]) => Math.max(b - a, total * 0.03)), gap = segs.length > 1 ? Math.min(0.04, 0.3 / (segs.length - 1)) : 0;
    const k = (1 - gap * (segs.length - 1)) / ws.reduce((s, w) => s + w, 0);
    const starts = ws.map((w, j) => ws.slice(0, j).reduce((s, v) => s + v * k + gap, 0));
    const x = (t) => {
      const T = t, j = segs.findIndex(([, b]) => T <= b), i = j < 0 ? segs.length - 1 : j, [a, b] = segs[i];
      return starts[i] + Math.min(Math.max((T - a) / (b - a || 1), 0), 1) * ws[i] * k;
    };
    const axis = [{ x: 0, text: `Day ${dn(segs[0][0])}` }], bands = [];
    segs.forEach(([a, b], j) => {
      if (j) {
        const p = segs[j - 1][1], bx = starts[j] - gap;
        bands.push({ x: bx, w: gap });
        axis.push({ x: bx + gap / 2, mid: 1, required: dn(a) > dn(p) && phaseStarts.has(dn(a)),
          text: dn(a) > dn(p) ? `Day ${dn(a)}` : `${Math.round((a - p) / 3600)} h later` });
      }
      for (let m = day0 + Math.ceil((a - day0) / DAY) * DAY; m < b; m += DAY) {
        if (m <= a) continue;
        axis.push({ x: x(m), text: `Day ${dn(m)}`, required: phaseStarts.has(dn(m)) });
        bands.push({ x: x(m), line: 1 });
      }
    });
    axis.sort((p, q) => p.x - q.x);
    return { x, axis, bands };
  };

  // Conversations ---------------------------------------------------------------------------

  const commits = (shas) => h("ul.commits", null, shas.map(([sha, subject]) => h("li", null, h("a.sha", { href: GH + sha }, sha), " ", subject)));
  const leftLine = (text) => h("p.left", null, h("a", { href: ABOUT }, "Left out"), text.replace(/^Left out/, ""));
  // A button that shows and hides a region, built by make() when first opened.
  const fold = (labels, make, cls = "link") => {
    const [show, hide = show] = [].concat(labels);
    const region = h("div", { hidden: true });
    const b = h(`button.${cls}`, { type: "button", "aria-expanded": "false" }, show);
    b.onclick = () => {
      const open = region.hidden;
      region.hidden = !open;
      b.setAttribute("aria-expanded", open);
      b.textContent = open ? hide : show;
      if (open && !region.firstChild) region.append(make());
    };
    return [b, region];
  };
  // At most n rows, then "Show all N"; the button hands focus to the first new row.
  const capped = (items, row, n = 20) => [items.slice(0, n).map(row), items.length > n && button({ onclick() {
    const rest = items.slice(n).map(row);
    this.replaceWith(...rest);
    rest[0].tabIndex = -1;
    rest[0].focus();
  } }, `Show all ${num(items.length)}`)];
  // "5 helper agents in 1 of 5 planned stages: Design": only the stages that had agents are named.
  const stagesRan = (g) => {
    const ran = g.stages.filter((s) => s.agents.length);
    const n = ran.length === g.stages.length ? plural(ran.length, "stage") : `${num(ran.length)} of ${num(g.stages.length)} planned stages`;
    return `${plural(g.agents, "helper agent")} in ${n}: ${ran.map((s) => s.name).join(", ")}`;
  };
  // A planned group: a box in the work, or (row: the group above, or true) a line of the list under the map.
  const group = (g, id, row) => {
    const plan = () => {
      const box = h("div.plan", { tabindex: 0, role: "region", "aria-label": `Plan of ${g.name}` }); // scrolls inside its box
      const fill = (again) => wait(box, "Loading…", load(`c/${id}.plans.json`), "The plan could not be loaded.", () => fill(true))
        .then((f) => f && (put(box, runs(f.plans[g.plan])), again && box.focus()));
      fill();
      return h("div", null, h("p.label", null, "The plan as the agent wrote it: the first lines name the stages, the quoted texts are what each helper agent was told, and the last lines set the order."), box);
    };
    const [pb, region] = g.plan != null ? fold(["Show the plan", "Hide the plan"], plan) : [];
    if (row) {
      return h("div.row", null, join([(row === true || dn(row.from) !== dn(g.from)) && when(g.from), g.name, stagesRan(g), dur(g.minutes), g.status]), pb && [" · ", pb], region);
    }
    const [ab, agents] = fold(["Show the agents", "Hide the agents"], () => h("div", null, g.stages.map((s) => s.agents.length > 0
      && h("p", null, `${s.name}: `, s.agents.map((a) => (a.status === "completed" ? a.label : `${a.label} (${a.status})`)).join(", ")))));
    return h("div.box.group", null, h("p", null, join([`Planned group ${g.name}`, dur(g.minutes), g.status])),
      h("div.stages", { role: "img", "aria-label": `${plural(g.agents, "helper agent")}: ${g.stages.map((s) => {
        const k = s.agents.filter((a) => a.status !== "completed").length;
        return `${s.name} ${s.agents.length}${k ? ` (${k} stopped or failed)` : ""}`;
      }).join(", ")}` },
        g.stages.map((s) => h("div", null, h("span", null, s.name), h("span.dots", null, s.agents.map((a) => h(`i.${a.status === "completed" ? "hp" : "hx"}`)))))),
      g.agents > 0 && [h("p", null, ab), agents], pb && [h("p", null, pb), region]);
  };
  const START = [
    ["port", (c) => `One agent alone at first, then ${plural(c.counts.helpers, "helper agent")} on day ${dn(c.helper_runs[0][0])}.`],
    ["improve", () => "Planned groups of helper agents."],
    ["release", (c) => `${plural(c.counts.helpers, "helper agent")} in one day, and the first release.`],
  ];
  const KEYS = [["Task from the agent that started it", "what a helper agent was told"], ["Report to the agent that started it", "what it sent back"],
    ["Message from …", "text another conversation sent here"], ["Pasted from …", "text copied in by hand from another conversation"],
    ["Scheduled check-in", "the agent woke itself at a set time"], ["Left out", ["a passage not published, see ", h("a", { href: ABOUT }, "About")]]];
  const SHAPES = [[{}, "one agent alone"], [{ fan: 4 }, "with helper agents"], [{ stages: GRID }, "with planned groups"], [{ pair: true }, "linked to another conversation"]];
  const FOLDED = new Set(["ran", "changed", "searched", "opened", "read", "sent"]);

  async function conversations(root, signal, again) {
    const p = new URLSearchParams(location.search);
    const state = { c: p.get("c"), at: p.get("at"), day: p.get("day") };
    const search = h("input", { type: "search", placeholder: "Search titles and messages", "aria-label": "Search titles and messages", value: p.get(SEARCH) });
    const phaseSel = h("select", { "aria-label": "Phase" }, h("option", { value: "" }, "All phases"));
    const dayLine = h("p.count", { hidden: true });
    const count = h("p.count", { "aria-live": "polite" });
    const rowsBox = h("div");
    const reader = h("section.reader", { "aria-labelledby": TITLE });
    const live = h("p.sr", { "aria-live": "polite" });
    root.classList.add("history-app", "history-conv");
    put(root, h("nav.list", { "aria-label": "Conversations" },
      h("form", { role: "search", onsubmit: (e) => e.preventDefault() }, search, phaseSel), dayLine, count, rowsBox), reader, live);
    const index = await wait(rowsBox, "Loading the conversations…", loadIndex(), "The conversations could not be loaded.",
      () => conversations(root, signal, true));
    if (!index || signal.aborted) return;
    for (const x of index.phases) phaseSel.append(h("option", { value: x.id }, x.title));
    phaseSel.value = p.get("phase") || "";
    if (phaseSel.selectedIndex < 0) phaseSel.value = "";
    const convs = new Map(index.conversations.map((c) => [c.id, c]));
    const titles = new Map(index.conversations.map((c) => [c.title, c.id]));
    const sentTo = (text) => [...titles.keys()].find((t) => text.endsWith(` to ${t}`)); // a message to another conversation
    const phases = new Map(index.phases.map((x) => [x.id, x]));
    const turns = Map.groupBy(index.turns, (t) => t.c);
    const link = (id, at) => h("a", { href: `?c=${id}${at != null ? `&at=${at}` : ""}` }, convs.get(id)?.title ?? id);
    // A helper's model and effort level, unless the lead agent worked with the same at that time.
    const otherEffort = (y) => {
      const list = convs.get(y.id.split("/")[0])?.models || [], lead = list.findLast((m) => m.from <= y.from) ?? list[0];
      return (!lead || effort(y) !== effort(lead)) && effort(y);
    };
    const phone = matchMedia("(max-width: 59.984em)");
    let seq = 0, listScroll = 0, shownCount;

    // The list, grouped by phase; the filters toggle `hidden`.
    const rows = index.conversations.map((c) => ({
      c,
      // A grid stands for its helper agents too, so a row shows the grid or the fan, then the link to other conversations.
      el: h("a.row", { href: `?c=${c.id}` }, h("b", null, h("span.gs", null, glyph({ stages: c.group_runs.length > 0 && GRID,
        fan: !c.group_runs.length && c.helper_runs.length, pair: c.related.length > 0 || !!c.started_by })), c.title), h("span", null,
        join([cap(span(dn(c.from), dn(c.to))), c.tool, c.counts.owner && `${plural(c.counts.owner, "message")} from the human`,
          c.counts.helpers + c.counts.group_agents && plural(c.counts.helpers + c.counts.group_agents, "helper agent")]),
        c.started_by && [" · started by ", h("em", null, convs.get(c.started_by.c)?.title)], !c.transcript && " · summary only")),
      text: [c.title, c.summary, ...(turns.get(c.id) || []).flatMap((t) => t.owner.map((o) => plain(o.text)))].join("\n").toLowerCase(),
      spans: c.spans,
    }));
    const groups = index.phases.map((x) => h("div", null, h("p.head", null, `${x.title} · ${span(x.day_from, x.day_to)}`),
      rows.filter((r) => r.c.phase === x.id).map((r) => r.el))).filter((g) => g.children.length > 1);
    const empty = h("p.status");
    put(rowsBox, groups, empty);
    const active = (r, d) => r.spans.some(([a, b]) => a < d + DAY && b > d);

    const filter = () => {
      const s = search.value.trim(), ph = phaseSel.value, d = state.day ? day0 + (state.day - 1) * DAY : null;
      let n = 0;
      for (const r of rows) n += !(r.el.hidden = !((!ph || r.c.phase === ph) && (!s || r.text.includes(s.toLowerCase())) && (d == null || active(r, d))));
      for (const g of groups) g.hidden = !g.querySelector("a.row:not([hidden])");
      dayLine.hidden = d == null;
      count.hidden = d != null && index.days[state.day - 1].quiet; // the day view says it: nothing was published
      if (d != null) put(dayLine, `Day ${state.day} · `, button({ onclick: () => (go({ day: null }, false, true), search.focus()) }, "Show all days"));
      const text = s || ph || d != null ? `${num(n)} of ${plural(rows.length, "conversation")} · ` : plural(n, "conversation");
      if (text !== shownCount) {
        put(count, (shownCount = text), (s || ph || d != null) && button({ onclick: () => ((search.value = phaseSel.value = ""), go({ day: null }), search.focus()) }, "Clear filters"));
      }
      put(empty, !n && s && [`No conversations match "${s}". The search covers titles, summaries and messages from the human. `,
        button({ onclick: () => ((search.value = ""), filter(), setURL(), search.focus()) }, "Clear the search")]);
    };
    const setURL = (push) => !signal.aborted && setQuery({ ...state, phase: phaseSel.value, [SEARCH]: search.value.trim() }, push);
    search.oninput = debounce(() => (filter(), setURL()), 150);
    phaseSel.onchange = () => (filter(), setURL(), root.dataset.view === "list" && start());

    // Shows a state: a conversation, a day or nothing; `focus` moves focus to the reader's heading,
    // `push` gives the new view its own history entry.
    const go = (next, focus, push) => {
      Object.assign(state, next);
      if (state.day && !(/^\d+$/.test(state.day) && index.days.some((d) => d.n === +state.day))) state.day = null;
      const top = state.c?.split("/")[0];
      for (const r of rows) r.c.id === top ? r.el.setAttribute("aria-current", "true") : r.el.removeAttribute("aria-current");
      root.dataset.view = state.c ? "read" : state.day ? "day" : "list";
      filter();
      setURL(push);
      // Desktop: the list scrolls by itself; keep the open conversation in view there.
      const cur = root.querySelector("a.row[aria-current]:not([hidden])"), box = cur?.closest(".list");
      if (cur && !phone.matches && (cur.offsetTop < box.scrollTop || cur.offsetTop + cur.offsetHeight > box.scrollTop + box.clientHeight)) {
        box.scrollTop = cur.offsetTop - box.clientHeight / 3;
      }
      if (state.c) return open(focus);
      state.day ? dayView() : start();
      if (focus) reader.querySelector(`#${TITLE}`)?.focus();
    };
    const back = () => {
      const row = root.querySelector("a.row[aria-current]");
      go({ c: null, at: null }, false, true);
      row?.focus({ preventScroll: true });
      scrollTo(0, listScroll);
    };
    const mark = (el, keep = true) => {
      reader.querySelector(".is-target")?.classList.remove("is-target");
      el.classList.toggle("is-target", keep);
      el.tabIndex = -1;
      // Lay out the items before it: their estimated heights would move it after the scroll.
      for (const it of reader.querySelectorAll(".it")) {
        it.style.contentVisibility = "visible";
        if (it.contains(el)) break;
      }
      el.scrollIntoView({ block: "start" });
      el.focus({ preventScroll: true });
    };

    // In-page links (?c=, ?day=, ?phase=) change the state instead of loading the page again.
    root.addEventListener("click", (e) => {
      const a = e.target.closest("a[href]");
      if (!a || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const href = a.getAttribute("href");
      // Leaving a conversation: its entry remembers the item in view, so Back returns there.
      if (state.c && !href.startsWith(`?c=${state.c}&`)) {
        const it = a.closest("[data-at],[data-from]") || [...reader.querySelectorAll(".items .it")].find((x) => x.getBoundingClientRect().bottom > 72);
        if (it) (state.at = it.dataset.at ?? it.dataset.from), setURL();
      }
      if (!href.startsWith("?")) return;
      e.preventDefault();
      e.stopPropagation(); // keeps Material's instant navigation out of it
      const u = new URLSearchParams(href);
      if (u.get("c") && u.get("c") === state.c && u.get("at")) {
        state.at = u.get("at");
        setURL();
        return target(reader.querySelector(".items"));
      }
      if (a.classList.contains("row") && phone.matches) listScroll = scrollY;
      if (u.has("phase")) phaseSel.value = u.get("phase");
      if (phaseSel.selectedIndex < 0) phaseSel.value = "";
      go(u.get("c") ? { c: u.get("c"), at: u.get("at") } : { c: null, at: null, day: u.get("day") }, !u.get("at"), true);
    });
    // Back and forward between views of this page. Material ignores a return to the address it
    // loaded last, so the page shows the entry's view itself (capture: before Material sees it).
    const path = location.pathname;
    addEventListener("popstate", (e) => {
      if (location.pathname !== path) return;
      e.stopImmediatePropagation();
      const u = new URLSearchParams(location.search);
      search.value = u.get(SEARCH) || "";
      phaseSel.value = u.get("phase") || "";
      if (phaseSel.selectedIndex < 0) phaseSel.value = "";
      go({ c: u.get("c"), at: u.get("at"), day: u.get("day") });
    }, { capture: true, signal });
    addEventListener("keydown", (e) => {
      if (!state.c || e.key !== "Escape" || e.target.closest("input, select, textarea")) return;
      phone.matches ? back() : root.querySelector("a.row[aria-current]")?.focus();
    }, { signal });

    // All conversations on the compressed day axis, grouped by phase (or one phase's days).
    const allMap = () => {
      const ph = phases.get(phaseSel.value);
      const ds = index.days.filter((d) => !ph || (d.n >= ph.day_from && d.n <= ph.day_to)), first = ds[0].n, last = ds.at(-1).n;
      // Busy days get more room (by the square root of their conversations); the strip keeps equal columns.
      const { parts, of, W } = compress(ds, (d) => Math.max(1, Math.sqrt(d.conversations))), start = (n) => day0 + (n - 1) * DAY;
      const X = (t) => {
        const n = Math.min(Math.max(dn(t), first), last), q = of[n];
        return (q.x + (q.gap ? 0 : Math.min(Math.max((t - start(n)) / DAY, 0), 1)) * q.w) / W;
      };
      const shown = index.conversations.filter((c) => !ph || c.spans.some(([a, b]) => a < start(last + 1) && b > start(first)));
      const row = (c) => {
        const t = c.title, owners = (turns.get(c.id) || []).flatMap((u) => u.owner.filter(typed));
        return {
          label: h("a", { href: `?c=${c.id}`, "data-x": -1, "aria-label": join([t, cap(span(dn(c.from), dn(c.to))), c.where,
            c.counts.owner && `${plural(c.counts.owner, "message")} from the human`, c.helper_runs.length && plural(c.helper_runs.length, "helper agent"),
            c.group_runs.length && plural(c.group_runs.length, "planned group")]) }, t),
          marks: [
            ...c.spans.map(([a, b]) => pin("bar", `?c=${c.id}&at=${a}`, `${t} · ${fromTo(a, b)}`, X(a), Math.max(X(b) - X(a), 0.002))),
            ...c.helper_runs.map(([f, m], k) => pin("hd", `?c=${c.id}&at=${f}`, `${t} · helper agent started ${when(f)} · ${dur(m)}`, X(f), null,
              `;top:${(8 + ((k * 0.618) % 1) * 22).toFixed(0)}%`)),
            ...owners.map((o) => pin("tick", `?c=${c.id}&at=${o.at}`, `${t} · human · ${when(o.at)}`, X(o.at))),
          ],
        };
      };
      const lines = [], at = new Map(), paths = [];
      for (const x of index.phases) {
        const cs = shown.filter((c) => c.phase === x.id);
        if (!cs.length) continue;
        lines.push({ cls: "phase", label: h("a", { href: `?phase=${x.id}`, "data-x": -1 }, `${x.title} · ${span(x.day_from, x.day_to)}`), marks: [] });
        for (const c of cs) at.set(c.id, lines.push(row(c)) - 1);
      }
      for (const c of shown) {
        for (const r of c.related) {
          if ((r.how !== "started" && r.how !== "sent") || !at.has(r.c)) continue;
          const x = (X(r.at[0]) * 100).toFixed(2), a = at.get(c.id) + 0.5, b = at.get(r.c) + 0.5;
          paths.push(`M${x} ${a}C${+x + 1.5} ${a},${+x + 1.5} ${b},${x} ${b}`);
        }
      }
      const starts = new Set(ph ? ds.filter((d) => !d.quiet).map((d) => d.n) : index.phases.map((x) => x.day_from));
      return [...chart("all", {
        label: "Map of all conversations, day by day; the arrow keys move between marks; the list of conversations has the same entries",
        axis: parts.filter((q) => !q.gap && starts.has(q.n)).map((q) => ({ x: q.x / W, text: `Day ${q.n}`, required: !ph })),
        bands: parts.flatMap((q, k) => (q.gap ? [{ x: q.x / W, w: q.w / W, text: quietText(q) }] : k && !parts[k - 1].gap ? [{ x: q.x / W, line: 1 }] : [])),
        rows: lines, paths,
      }, signal), legend([["tick", "human"], ["bar", "agents working"], ["hd", "helper agent started"], ["link", "a conversation started or messaged another"]])];
    };

    // Nothing open.
    // On phones the whole map is too narrow to read; it starts folded there unless one phase crops it.
    const start = () => {
      const [mb, map] = phoneWidth() && !phaseSel.value ? fold(["Show the map", "Hide the map"], () => h("div", null, allMap())) : [];
      put(reader,
        h("h2", { id: TITLE, tabindex: -1 }, "All conversations, day by day"),
        phaseSel.value && h("p.meta", null, h("a", { href: "?phase=" }, "Show all phases")), mb ? [h("p", null, mb), map] : allMap(),
        h("h2", null, "Start with"),
        h("ul.start", null, START.map(([id, text]) => convs.has(id) && h("li", null, h("a", { href: `?c=${id}` }, convs.get(id).title), ". ", text(convs.get(id))))),
        h("h2", null, "How to read a conversation"),
        h("div.owner", null, h("div.msg", null, h("p.text", null, "Messages from the human have a colored line on the left, like this one."))),
        h("p", null, "The agent's replies have no line. Between a message and its reply, a grey line sums up the work: time taken, commands run, files changed. \"Show the work\" opens it note by note."),
        h("p", null, "Where several agents worked, a conversation opens with a map of who worked when: the human, the agent, its helper agents, planned groups and other conversations. A small mark before each title and work line shows the shape of the work:"),
        h("dl.shapes", null, SHAPES.map(([shape, text]) => [h("dt", null, glyph(shape)), h("dd", null, text)])),
        h("dl.keys", null, KEYS.map(([term, text]) => [h("dt", null, term), h("dd", null, text)])));
    };

    // A day: the conversations active on it, then the owner's messages.
    const dayView = () => {
      const n = +state.day, d = index.days[n - 1], d0 = day0 + (n - 1) * DAY;
      const busy = index.days.filter((x) => !x.quiet), prev = busy.findLast((x) => x.n < n), next = busy.find((x) => x.n > n);
      const pct = (t) => `${(((Math.min(Math.max(t, d0), d0 + DAY) - d0) / DAY) * 100).toFixed(2)}%`;
      const nav = (x, label) => (x ? h("a", { href: `?day=${x.n}` }, label) : h("span"));
      const readout = h("p.readout", null, coarse.matches ? "Tap a conversation to open it." : "Point at or tab to a conversation for its details.");
      const owners = index.turns.flatMap((t) => t.owner.filter((o) => typed(o) && dn(o.at) === n).map((o) => ({ ...o, c: t.c })));
      const lines = rows.filter((r) => active(r, d0)).map(({ c, spans }) => {
        const mine = spans.filter(([a, b]) => a < d0 + DAY && b > d0), ticks = owners.filter((o) => o.c === c.id);
        const from = Math.max(mine[0][0], d0);
        const text = join([c.title, c.where, ticks.length && `${plural(ticks.length, "message")} from the human`]);
        const el = h("a.drow", { href: `?c=${c.id}&at=${from}`, "aria-label": text }, h("span", null, c.title), h("span.track", null,
          mine.map(([a, b]) => h("i", { style: `left:${pct(a)};width:calc(${pct(b)} - ${pct(a)})` })),
          ticks.map((o) => h("b", { style: `left:${pct(o.at)}` }))));
        el.onpointerenter = el.onfocus = () => put(readout, text);
        return el;
      });
      put(reader, h("div.daynav", null, nav(prev, `← Day ${prev?.n}`), h("h2", { id: TITLE, tabindex: -1 }, `Day ${n}`), nav(next, `Day ${next?.n} →`)),
        d.quiet ? h("p.status", null, "Nothing was published.") : [
          h("p.meta", null, join([phases.get(d.phase)?.title, `agents working ${num(d.hours)} of 24 hours`,
            `${plural(d.owner, "message")} from the human`, plural(lines.length, "conversation")])),
          h("div.chart", null, lines),
          readout,
          // The conversation's title once per run of its messages; each message's text opens it in the conversation.
          owners.length > 0 && [h("h3", null, "Messages from the human"), owners.map((o, k) => h("div.it.owner.compact", null,
            h("div", null, o.c !== owners[k - 1]?.c && h("p.label", null, h("a", { href: `?c=${o.c}` }, convs.get(o.c).title)),
              h("div.msg", null, h("p.text", null, h("a", { href: `?c=${o.c}&at=${o.at}` }, runs(o.text)))))))],
        ]);
    };

    // The facts under a reader title, as [term, ...value] pairs.
    const facts = (pairs) => h("dl", null, pairs.filter(Boolean).map(([term, ...value]) => [h("dt", null, term), h("dd", null, ...value)]));
    const record = (text) => ["Record", text, " ", h("a", { href: ABOUT }, "About the records")];
    const header = (c) => {
      const sb = c.started_by, n = c.counts, cd = c.condensed, lead = (s) => (c.changes.length ? s : cap(s));
      const [changes, list] = c.changes.length ? fold(`${plural(c.changes.length, "change")} on GitHub`, () => commits(c.changes)) : [];
      const sha = (s) => h("a.sha", { href: GH + s }, s);
      // One line per model: "gpt-6-astra, effort level high; xhigh from day 1".
      const models = [...Map.groupBy(c.models, (m) => m.model)].map(([m, list], k) => [k > 0 && h("br"), [m, list.some((x) => x.effort) && `effort level ${list
        .filter((x) => x.effort).map((x, k) => (k ? `${x.effort} from ${when(x.from)}` : x.effort)).join("; ")}`].filter(Boolean).join(", ")]);
      return h("header", null, h("h2", { id: TITLE, tabindex: -1 }, c.title), c.transcript && h("p.summary", null, c.summary), facts([
        ["When", `${fromTo(c.from, c.to)} · active for ${dur(c.spans.reduce((a, [x, y]) => a + minutes(x, y), 0))}`],
        ["Tool", c.where],
        ["Model", ...models],
        sb && ["Started by", link(sb.c, sb.at)],
        ["Human", n.owner ? join([plural(n.owner, "message"), n.answers && plural(n.answers, "answer")]) : "no messages"],
        ["Agents", join([replies(n.agent), ...tally({ commands: n.commands, files: n.files })]) || "none"],
        ["Helpers", join([n.helpers && plural(n.helpers, "helper agent"), n.groups && `${plural(n.groups, "planned group")} with ${plural(n.group_agents, "helper agent")}`]) || "none"],
        ["Changes", changes, cd.length > 0 && [changes && " · ", cd.length === 1 ? [lead("part of "), sha(cd[0]), `, condensed on day ${index.condensed_day}`]
          : [lead(`part of ${plural(cd.length, "change")} condensed on day ${index.condensed_day}: `), cd.map((s, k) => [k ? ", " : "", sha(s)])]],
        !changes && !cd.length && "none"],
        ["In the story", c.phases.map((id, k) => [k ? " · " : "", h("a", { href: `../#${id}` }, phases.get(id)?.title)])],
        record([{ complete: "Every message is shown.", partial: `Partly shown: ${c.record_note}.`, summary: `Summary only: ${c.record_note}.` }[c.record],
          c.encrypted_tasks && " Tasks and messages that agents sent each other were stored encrypted and are not shown.",
          n.groups > 0 && (c.place === "in the cloud" ? " The conversations of planned-group agents are not in the records."
            : " The conversations of planned-group agents are not shown.")]),
      ]), list);
    };

    // How the work of a conversation (c) or of one of its helper agents (x) was organized:
    // a map of the owner's messages, the agent, its helper agents, planned groups and the
    // other conversations it is linked to, with the same content as a list behind it.
    const organized = (id, c, file, x) => {
      const end = (y) => y.from + y.minutes * 60;
      const agent = x ? [[x.from, end(x)]] : c.agent_spans;
      const owners = x ? [] : file.items.filter((it) => it.t === "owner");
      const hs = file.helpers.toSorted((a, b) => a.from - b.from), gs = file.groups.toSorted((a, b) => a.from - b.from);
      const links = x ? [] : [...(c.started_by ? [{ c: c.started_by.c, how: "by", at: [c.started_by.at] }] : []), ...c.related];
      const others = [...Map.groupBy(links, (r) => r.c)];
      const relays = x ? file.items.filter((it) => it.how === "task" || it.how === "report") : []; // a helper's tasks and reports
      if (!hs.length && !gs.length && !links.length) return h("p.alone", null, "One agent worked alone.");
      const { x: X, axis, bands } = timeAxis([...(x ? agent : c.spans), ...hs.map((y) => [y.from, end(y)]),
        ...gs.map((g) => [g.from, end(g)]), ...[...owners, ...relays].map((o) => [o.at, o.at]), ...links.flatMap((r) => r.at.map((a, k) => (r.how === "sent" ? r.sent?.[k] ?? a : a)).map((a) => [a, a]))],
      new Set(index.phases.map((p) => p.day_from)));
      const lanes = (list, a, b, under) => { // first fit by start time; a helper sits below its parent
        const ends = [], lane = new Map();
        for (const y of list) {
          let r = under ? (lane.get(y.parent) ?? -1) + 1 : 0;
          while (ends[r] > a(y) - 0.003) r++;
          ends[r] = Math.max(b(y), a(y) + 0.004);
          lane.set(y.id ?? y, r);
        }
        return lane;
      };
      const hLane = lanes(hs, (y) => X(y.from), (y) => X(end(y)), true), gLane = lanes(gs, (g) => X(g.from), (g) => X(end(g)));
      const WORDS = { by: "started this conversation", started: "started", sent: "message sent", received: "message received", pasted: "text pasted in" };
      const RELAY = { task: "Task from the agent that started it", report: "Report to the agent that started it" };

      const draw = (full) => {
        const rows = [], paths = [], cap = full ? Infinity : 16, dense = [];
        if (owners.length) rows.push({ label: "Human", marks: owners.map((o) => pin("tick", `?c=${id}&at=${o.at}`, `Human · ${when(o.at)}`, X(o.at))) });
        if (relays.length) rows.push({ label: "Starting agent", marks: relays.map((it) => pin(it.how === "task" ? "tick.ink" : "hd", `?c=${id}&at=${it.at}`, `${RELAY[it.how]} · ${when(it.at)}`, X(it.at), null, it.how === "task" ? "" : ";top:50%")) });
        rows.push({ label: "Agent", marks: agent.map(([a, b]) => pin("bar", `?c=${id}&at=${a}`, `Agent · ${fromTo(a, b)}`, X(a), Math.max(X(b) - X(a), 0.002))) });
        const base = rows.length, hRows = Math.min(Math.max(-1, ...hLane.values()) + 1, cap);
        for (let r = 0; r < hRows; r++) rows.push({ label: r ? "" : "Helper agents", marks: [] });
        for (const y of hs) {
          const r = hLane.get(y.id), a = X(y.from), w = Math.max(X(end(y)) - a, 0.002), pr = hLane.get(y.parent);
          if (r >= cap) {
            dense.push(h("i.hbar", { style: pos(a, w) }));
            continue;
          }
          rows[base + r].marks.push(pin("hbar", `?c=${y.id}`, `${join([helperName(y), y.nick && y.task, otherEffort(y), fromTo(y.from, end(y))])} · Open`, a, w));
          if (pr != null) paths.push(`M${(a * 100).toFixed(2)} ${pr + 0.5}V${r + 0.5}`);
        }
        if (dense.length) rows.push({ cls: "dense", label: button({ "data-x": -1, onclick: () => expand() }, `and ${num(dense.length)} more`), marks: dense });
        const gBase = rows.length;
        for (const g of gs) {
          const r = gBase + gLane.get(g), a = X(g.from);
          rows[r] ||= { label: r > gBase ? "" : "Planned groups", marks: [] };
          rows[r].marks.push(put(pin("grp", `?c=${id}&at=${g.from}`, join([`Planned group ${g.name}`, stagesRan(g),
            when(g.from), dur(g.minutes), g.status]), a, Math.max(X(end(g)) - a, 0.006)), g.stages.map(() => h("i"))));
        }
        for (const [o, rs] of others) {
          const t = convs.get(o)?.title ?? o, n = (how) => rs.filter((r) => r.how === how).reduce((s, r) => s + r.at.length, 0);
          const sums = [n("sent") && `${plural(n("sent"), "message")} sent`, n("received") && `${plural(n("received"), "message")} received`,
            n("pasted") && `${plural(n("pasted"), "message")} pasted in`];
          rows.push({ cls: "other", label: h("a", { href: `?c=${o}`, "data-x": -1, "aria-label": join([t, ...sums]) }, t),
            marks: rs.flatMap((r) => r.at.map((at, k) => {
              const start = r.how === "by" || r.how === "started";
              const here = r.how === "received" || r.how === "pasted"; // the time is this conversation's: open its item here
              const t0 = r.how === "sent" ? r.sent?.[k] ?? at : at; // a sent message sits at its send time here; Open shows its arrival
              return pin(start ? "ring" : r.how === "pasted" ? "tick" : "tick.ink", `?c=${here ? id : o}&at=${at}`,
                join([t, `${WORDS[r.how]} on ${when(t0)}`, ...(start ? sums : [])]), X(t0));
            })) });
        }
        const [map, readout] = chart("conv", { axis, bands, rows, paths, top: base, n: hRows,
          label: "Map of this conversation's work; the arrow keys move between marks; the list after it has the same content" }, signal);
        put(box, map, readout, legend([owners.length && ["tick", "human"], relays.length && ["hd", "reports back"], ["bar", "agent"],
          hs.length && ["hbar", "helper agents"], gs.length && ["grp", "planned groups"], others.length && ["ring", "other conversations"]].filter(Boolean)));
        return map;
      };
      const expand = () => {
        const map = draw(true), first = map.querySelectorAll(".body > .mrow")[(owners.length || relays.length ? 2 : 1) + 16]?.querySelector("a");
        if (first) map.move(first);
      };
      const box = h("div");

      // The same content as a list, for reading it line by line.
      const hrow = (y, k, all) => h("div.row", { style: `margin-left:${Math.min(y.depth - 1, 2) * 0.8}rem` },
        join([(!k || dn(all[k - 1].from) !== dn(y.from)) && when(y.from), y.nick, y.task, otherEffort(y), dur(y.minutes)]), " · ", h("a", { href: `?c=${y.id}` }, "Open"));
      const SAY = { by: "Started by", started: "Started", sent: "Sent {n} to", received: "Received {n} from", pasted: "{n} pasted in from" };
      const rrow = (r) => h("div.row", null, `${SAY[r.how].replace("{n}", plural(r.at.length, "message"))} `, link(r.c, r.how === "received" ? r.sent?.[0] ?? r.at[0] : r.at[0]),
        r.how === "by" || r.how === "started" ? `, ${when(r.at[0])}` : "");
      const [lb, list] = fold(["Show as a list", "Hide the list"], () => h("div.split", null,
        hs.length > 0 && [h("p.head", null, "Helper agents"), capped(hs, hrow)],
        gs.length > 0 && [h("p.head", null, "Planned groups"), capped(gs, (g, k, all) => group(g, id, all[k - 1] ?? true))],
        links.length > 0 && [h("p.head", null, "Other conversations"), capped(links, rrow)]));
      // The list comes first for screen readers, before the many marks of the map.
      const [b, region] = fold("How this conversation's work was organized", () => {
        draw(false);
        return h("div", null, h("p", null, lb), list, box);
      }, "fold.title");
      b.click();
      return h("section.org", null, b, region);
    };

    // One transcript item; ctx = {id, nick, helpers: Map, groups}.
    const item = (it, ctx) => {
      const msg = (cls, label, ...body) => h(`div.it.${cls}`, { "data-at": it.at }, h("div.msg", null, h("p.who", null, label), ...body));
      if (it.t === "owner") {
        const [b, first] = it.kind === "edited" ? fold("First version", () => h("p.text", null, runs(it.first))) : [];
        return msg(`owner${it.added ? ".added" : ""}`, { answer: "Human · answer", edited: "Human · edited and sent again" }[it.kind]
          || (it.added ? "Human · while the agent worked" : "Human"), ownerText(it, 24), b, first);
      }
      if (it.t === "agent") {
        const body = h("div.html", { html: it.html }), level = effort(it), changed = level !== ctx.level && level; // written when it changes
        ctx.level = level;
        return msg("agent", it.question ? "Agent · question" : join([ctx.nick || "Agent", changed, it.reply && it.after > 10 && `reply after ${dur(it.after)}`]),
          clamp(body, 40, `Show the whole reply (${num(body.textContent.split(/\s+/).filter(Boolean).length)} words)`),
          it.question?.options && h("ul.opts", null, it.question.options.map((o) => h("li", null, o))));
      }
      if (it.t === "relay") {
        // The source where the message left it (its send line), if it was still running then.
        const at = it.sent ?? it.at, from = it.from && [dot("ho"), link(it.from, at <= convs.get(it.from)?.to ? at : null)];
        return msg("relay", { message: ["Message from ", from], task: "Task from the agent that started it", report: "Report to the agent that started it",
          pasted: ["Pasted", from && [" from ", from]], checkin: "Scheduled check-in: the agent resumed on its own" }[it.how],
        clamp(it.html ? h("div.html", { html: it.html }) : h("p.text", null, runs(it.text)), 8, "Show the whole message ({n} lines)"));
      }
      if (it.t === "work") {
        const rail = h("ol.rail", { hidden: true });
        const b = button({ "aria-expanded": "false" }, "Show the work");
        const stages = it.group_ix.length > 0 && ctx.groups[it.group_ix[0]]?.stages.map((s) => s.agents.map((a) => a.status === "completed"));
        const el = h("div.it.work", { "data-from": it.from, "data-to": it.to },
          h("p.sum", null, glyph({ fan: it.helper_ids.length, stages }), join([dur(minutes(it.from, it.to)), ...tally(it)]), " · ", b), rail);
        el.reveal = (T) => work(it, ctx, b, rail, T);
        b.onclick = () => el.reveal();
        return el;
      }
      if (it.t === "changes") return h("div.it.changes", { "data-at": it.at }, h("div", null, h("p.label", null, "Changes on GitHub"), commits(it.shas)));
      return h("div.it", { "data-at": it.at }, it.t === "left" ? leftLine(it.text) : h("p.line", null, it.text));
    };

    // The work between two messages: progress notes on a time rail, each with its activity folded.
    const work = async (it, ctx, b, rail, T, again) => {
      if (b.getAttribute("aria-disabled")) return;
      if (T == null && !again && !rail.hidden) {
        b.setAttribute("aria-expanded", "false");
        b.textContent = "Show the work";
        return (rail.hidden = true);
      }
      let hit = null; // a failed load leaves the rail open with "Try again", and the summary is the target
      if (!rail.dataset.ok || T != null) {
        b.setAttribute("aria-disabled", "true");
        b.textContent = "Loading…";
        const parts = await Promise.all(it.lines.map(([n]) => load(`c/${ctx.id}.w${n}.json`))).catch(() => null);
        b.removeAttribute("aria-disabled");
        if (parts) {
          rail.dataset.ok = 1;
          hit = steps(rail, it.lines.flatMap(([, a, z], k) => parts[k].lines.slice(a, z)), it.from, ctx, T);
        } else put(rail, h("li.status", null, "The work could not be loaded. ", button({ onclick: () => work(it, ctx, b, rail, T, true).then(() => b.focus()) }, "Try again")));
      }
      b.setAttribute("aria-expanded", "true");
      b.textContent = "Hide the work";
      rail.hidden = false;
      return hit;
    };
    const stamp = (at, prev) => dn(at) !== dn(prev) && h("span.t", null, when(at)); // the day where it changes
    const act = ([at, verb, text], t0) => {
      if (verb === "ran") text = commandText(text);
      const x = text.startsWith(`${verb} `) ? text.slice(verb.length + 1) : text, to = verb === "sent" && x.match(/^(.*? to )(.+)$/);
      return h("li", null, button({ class: "act", "aria-expanded": "false", onclick() {
        this.setAttribute("aria-expanded", this.classList.toggle("wrap"));
      } }, h("span", null, verb), h("span.x", null, to ? [to[1], dot(titles.has(to[2]) ? "ho" : "hp"), to[2]] : x)));
    };
    const reported = (x) => h("p.line", null, dot("hp"), `${x.nick || cap(x.task)} reported back`, " · ", h("a", { href: `?c=${x.id}` }, "Open"));
    // A message to another conversation, shown outside the fold; Open lands where it arrived.
    const sentLine = (ctx, at, text) => {
      const t = sentTo(text), id = titles.get(t), arrivals = (convs.get(ctx.id.split("/")[0])?.related || [])
        .filter((r) => r.c === id && (r.how === "sent" || r.how === "started")).flatMap((r) => r.at);
      const arrived = arrivals.reduce((a, b) => (Math.abs(b - at) < Math.abs(a - at) ? b : a), arrivals[0] ?? at);
      return h("p.line", null, dot("ho"), `Sent ${text.replace(/^sent /, "").slice(0, -t.length)}`, t, " · ", h("a", { href: `?c=${id}&at=${arrived}` }, "Open"));
    };
    const helperLine = (x) => [join([x.nick || cap(x.task), x.nick && x.task, otherEffort(x), dur(x.minutes)]), " · ", h("a", { href: `?c=${x.id}` }, "Open")];
    const helperBox = (xs) => xs.length > 0 && h("div.box", null, h("p", null, glyph({ fan: xs.length }),
      xs.length > 1 ? `Started ${plural(xs.length, "helper agent")}:` : ["Started helper agent ", helperLine(xs[0])]),
    xs.length > 1 && h("ul", null, xs.map((x) => h("li", null, helperLine(x)))));
    // Renders the rail; returns the element of the first line at or after T, if any.
    const steps = (rail, lines, t0, ctx, T) => {
      const list = [];
      let cur, hit, target;
      for (const [s, verb, text] of lines) {
        const at = t0 + s;
        if (verb === "note" || !cur) list.push((cur = { at, note: verb === "note" && text, acts: [], extra: [], left: [] }));
        const to = verb === "note" ? null : verb === "sent" && sentTo(text) ? cur.extra : FOLDED.has(verb) ? cur.acts
          : verb === "left" || verb === "event" ? cur.left : cur.extra;
        to?.push([at, verb, text]);
        if (T != null && !hit && at >= T) hit = { k: list.length - 1, i: to === cur.acts ? cur.acts.length - 1 : -1, e: to === cur.extra ? cur.extra.length - 1 : -1 };
        // A planned group or helper agent that started exactly at T wins (the maps link their starts).
        const start = verb === "group" ? ctx.groups[text]?.from : verb === "helper" && ctx.helpers.get(text)?.from;
        if (T != null && start === T && !hit?.exact) hit = { k: list.length - 1, i: -1, e: cur.extra.length - 1, exact: true };
      }
      const flat = list.length === 1 && !list[0].note; // no notes: the lines directly
      const els = list.map((st, k) => {
        const mine = hit?.k === k, all = mine && hit.i >= 50, n = (v) => st.acts.filter((x) => x[1] === v);
        const acts = h("ol.acts", { hidden: !flat && !(mine && hit.i >= 0) }, st.acts.slice(0, all ? undefined : 50).map((x) => act(x, t0)));
        if (st.acts.length > 50 && !all) {
          acts.append(h("li", null, button({ onclick() {
            const rest = st.acts.slice(50).map((x) => act(x, t0));
            this.parentNode.replaceWith(...rest);
            rest[0].querySelector("button").focus();
          } }, `Show all ${num(st.acts.length)}`)));
        }
        // Consecutive helper starts make one box; boxes[j] is the element of extra line j.
        const seqs = [], boxes = [];
        st.extra.forEach(([, verb, ref], j) => (verb === "helper" && seqs.at(-1)?.verb === "helper" ? seqs.at(-1).js.push(j) : seqs.push({ verb, ref, js: [j] })));
        const extras = seqs.map(({ verb, ref, js }) => {
          const x = ctx.helpers.get(ref), el = verb === "group" ? ctx.groups[ref] && group(ctx.groups[ref], ctx.id)
            : verb === "helper" ? helperBox(js.map((j) => ctx.helpers.get(st.extra[j][2])).filter(Boolean))
              : verb === "sent" ? sentLine(ctx, st.extra[js[0]][0], ref) : x && reported(x);
          for (const j of js) boxes[j] = el;
          return el;
        });
        const other = st.acts.length - n("ran").length - n("changed").length - n("sent").length;
        const el = h("li.step", null, stamp(st.at, k ? list[k - 1].at : t0), st.note && h("div.html", { html: st.note }), extras,
          st.acts.length > 0 && !flat && h("button.fold", { type: "button", "aria-expanded": String(!acts.hidden), onclick() {
            this.setAttribute("aria-expanded", (acts.hidden = !acts.hidden) === false);
          } }, join([n("ran").length && plural(n("ran").length, "command"),
            n("changed").length && `${plural(new Set(n("changed").map((x) => x[2])).size, "file")} changed`,
            n("sent").length && `${plural(n("sent").length, "message")} sent`, other && plural(other, "other step")])),
          st.acts.length > 0 && acts, st.left.map(([, verb, text]) => (verb === "event" ? h("p.line", null, text) : leftLine(text))));
        if (mine) target = hit.i >= 0 ? acts.children[hit.i] : (hit.e >= 0 && boxes[hit.e]) || el;
        return el;
      });
      const a = flat || list[0].note ? 8 : 9; // the first 8 notes and the last 4
      if (els.length > a + 4 && !(hit && hit.k >= a && hit.k < els.length - 4)) {
        const mid = els.slice(a, -4);
        for (const e of mid) e.hidden = true;
        const z = els.length - 4, s = !els[z].querySelector(":scope > .t") && stamp(list[z].at, list[a - 1].at); // the day across the gap
        if (s) els[z].prepend(s);
        const gap = h("li.gap", null, button({ onclick: () => {
          mid.forEach((e) => (e.hidden = false));
          gap.remove();
          if (s) s.remove(); // the steps in between show where the day changes
          mid[0].tabIndex = -1;
          mid[0].focus();
        } }, `Show the ${num(mid.length)} notes in between`));
        els.splice(a, 0, gap);
      }
      put(rail, els);
      return target;
    };

    // Scrolls to the item at or after state.at; inside the work, opens it first. An owner
    // message at exactly that second wins over a note at the same time.
    const target = async (box) => {
      if (!box || !state.at) return;
      const T = /^-?\d+$/.test(state.at) ? +state.at : NaN, exact = !isNaN(T) && box.querySelector(`.it.owner[data-at='${T}']`);
      if (exact) return mark(exact);
      for (const el of box.querySelectorAll("[data-at],[data-from]")) {
        if (el.dataset.from && T > +el.dataset.from && T <= +el.dataset.to) return mark((await el.reveal(T)) || el);
        if (+(el.dataset.at ?? el.dataset.from) >= T) return mark(el);
      }
      box.before(h("p.status", null, "The linked moment was not found; showing the start."));
    };

    // A conversation, or a helper agent's conversation ("port/averroes").
    const open = async (focus) => {
      const id = state.c, top = id.split("/")[0], c = convs.get(top), token = ++seq;
      const old = reader.querySelector(".items");
      old?.classList.add("stale");
      const status = h("p.status"), org = h("div");
      const backLink = h("p.back", null, button({ onclick: back }, "← All conversations"));
      const show = (head) => {
        put(reader, backLink, head, org, status, old);
        if (focus && head) reader.querySelector(`#${TITLE}`).focus();
      };
      const missing = () => put(reader, backLink, h("p", { id: TITLE, tabindex: -1 }, `There is no conversation called "${id}". It may have been renamed. `,
        button({ onclick: () => go({ c: null, at: null }, true, true) }, "All conversations")));
      if (!c) return missing();
      const fresh = () => token === seq && !signal.aborted;
      let head = top === id && header(c);
      show(head);
      if (head && !c.transcript) return reader.append(h("h3", null, "Summary"), h("p", null, c.summary));
      const get = (name) => wait(status, "Loading the conversation…", load(name), "The conversation could not be loaded.", () => go({}, true));
      const parent = await get(`c/${top}.json`);
      if (!parent || !fresh()) return;
      const x = top !== id && parent.helpers.find((y) => y.id === id);
      if (top !== id && !x) return missing();
      const own = x && (await get(`c/${id}.json`));
      if ((x && !own) || !fresh()) return;
      const file = own || parent;
      const helpers = new Map([...parent.helpers, ...file.helpers].map((y) => [y.id, y]));
      if (x) {
        const lead = x.parent && helpers.get(x.parent), sum = { commands: 0, files: 0, notes: 0 };
        for (const w of file.items) if (w.t === "work") for (const k in sum) sum[k] += w[k];
        const mine = file.helpers.filter((y) => y.parent === id), g = file.groups;
        head = h("header", null, h("p.crumb", null, link(top, x.from), ` › ${x.nick || cap(x.task)}`),
          h("h2", { id: TITLE, tabindex: -1 }, helperName(x)),
          h("p.summary", null, "Started by ", lead ? helperName(lead).replace(/^H/, "h") : ["the agent of ", link(top)], ".",
            x.nick && ` Task label: ${x.task}.`),
          facts([["When", `${when(x.from)} · ${dur(x.minutes)}`], ["Model", effort(x)],
            ["Agents", join([replies(file.items.filter((i) => i.t === "agent").length + sum.notes), ...tally({ commands: sum.commands, files: sum.files })]) || "none"],
            ["Helpers", join([mine.length && plural(mine.length, "helper agent"), g.length && `${plural(g.length, "planned group")} with ${plural(g.reduce((s, y) => s + y.agents, 0), "helper agent")}`]) || "none"],
            record(c.encrypted_tasks ? "The tasks and messages it received from other agents were stored encrypted and are not shown." : "Every message is shown.")]));
        show(head);
      }
      put(org, organized(id, c, file, x));
      put(status, !c.counts.owner && c.started_by && top === id
        && ["The first message came from another conversation, ", link(c.started_by.c), "."]);
      const ctx = { id, nick: x && helperName(x), helpers, groups: file.groups };
      const box = h("div.items");
      let day, dayBox;
      const add = (list) => {
        for (const it of list) {
          const n = dn(it.at ?? it.from);
          if (n !== day) box.append((dayBox = h("div.day", null, h("h3.sep", null, `Day ${(day = n)}`))));
          dayBox.append(item(it, ctx));
        }
      };
      // Without a linked moment, the first 40 items are drawn first and the rest after the first paint.
      const k = state.at ? Infinity : 40;
      add(file.items.slice(0, k));
      old?.remove();
      reader.append(box);
      live.textContent = `${ctx.nick || c.title} loaded`;
      if (file.items.length > k) requestAnimationFrame(() => setTimeout(() => fresh() && add(file.items.slice(k))));
      else target(box);
    };

    go({}, false);
    if (again) reader.querySelector(`#${TITLE}`)?.focus();
  }

  let abort;
  const mounts = { days, split, messages, conversations };
  document$.subscribe(() => {
    abort?.abort();
    observer?.disconnect();
    abort = new AbortController();
    observer = new ResizeObserver(measure);
    for (const root of document.querySelectorAll("[data-history]")) mounts[root.dataset.history]?.(root, abort.signal);
  });
})();
