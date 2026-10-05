---
description: "How AI coding agents wrote treams-rs in 16 working days."
---

# How treams-rs was built

<!-- Generated file. Do not edit by hand. -->

AI coding agents wrote treams-rs in 16 working days. The port itself took
29 hours.
{ .lead }

<div class="history-figures" markdown>

- **233** messages from the human, about 4,900 words
- **25** conversations
- **about 920** helper agents

</div>

<div class="history-strip" data-history="days" data-days='[{"n":1,"quiet":false,"phase":"port","hours":3.5,"owner":14,"conversations":1},{"n":2,"quiet":false,"phase":"port","hours":20.3,"owner":10,"conversations":2},{"n":3,"quiet":false,"phase":"beyond","hours":12.0,"owner":36,"conversations":1},{"n":4,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":5,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":6,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":7,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":8,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":9,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":10,"quiet":false,"phase":"core","hours":3.1,"owner":16,"conversations":2},{"n":11,"quiet":false,"phase":"core","hours":4.7,"owner":8,"conversations":2},{"n":12,"quiet":true,"phase":null,"hours":0.0,"owner":0,"conversations":0},{"n":13,"quiet":false,"phase":"review","hours":0.9,"owner":2,"conversations":1},{"n":14,"quiet":true,"phase":"review","hours":0.0,"owner":0,"conversations":0},{"n":15,"quiet":false,"phase":"review","hours":2.0,"owner":10,"conversations":1},{"n":16,"quiet":false,"phase":"review","hours":1.0,"owner":2,"conversations":1},{"n":17,"quiet":false,"phase":"improve","hours":12.8,"owner":5,"conversations":1},{"n":18,"quiet":false,"phase":"improve","hours":6.7,"owner":0,"conversations":1},{"n":19,"quiet":false,"phase":"improve","hours":16.6,"owner":1,"conversations":1},{"n":20,"quiet":false,"phase":"improve","hours":24.0,"owner":4,"conversations":1},{"n":21,"quiet":false,"phase":"improve","hours":18.9,"owner":8,"conversations":1},{"n":22,"quiet":false,"phase":"prepare","hours":24.0,"owner":14,"conversations":3},{"n":23,"quiet":false,"phase":"prepare","hours":24.0,"owner":15,"conversations":7},{"n":24,"quiet":false,"phase":"release","hours":22.9,"owner":78,"conversations":14},{"n":25,"quiet":false,"phase":"release","hours":3.3,"owner":10,"conversations":4}]' data-phases='[{"id":"port","title":"The port","day_from":1,"day_to":2},{"id":"beyond","title":"Beyond the port","day_from":3,"day_to":3},{"id":"core","title":"Finishing the core","day_from":10,"day_to":11},{"id":"review","title":"Second opinions and proofs","day_from":13,"day_to":16},{"id":"improve","title":"Improve, then trim","day_from":17,"day_to":21},{"id":"prepare","title":"Ready for other people","day_from":22,"day_to":23},{"id":"release","title":"Release","day_from":24,"day_to":25}]' markdown>

The chart of the work per day needs JavaScript. The table below has the same numbers.

</div>

One column per day; quiet days are compressed.
{ .history-figure-caption }

??? days "Show the days as a table"

    | Day | Phase | Hours with agents working | Messages from the human | Conversations |
    | --- | --- | ---: | ---: | ---: |
    | Day 1 | The port | 3.5 | 14 | 1 |
    | Day 2 | The port | 20.3 | 10 | 2 |
    | Day 3 | Beyond the port | 12.0 | 36 | 1 |
    | Days 4–9 | 6 quiet days | – | – | – |
    | Day 10 | Finishing the core | 3.1 | 16 | 2 |
    | Day 11 | Finishing the core | 4.7 | 8 | 2 |
    | Day 12 | 1 quiet day | – | – | – |
    | Day 13 | Second opinions and proofs | 0.9 | 2 | 1 |
    | Day 14 | 1 quiet day | – | – | – |
    | Day 15 | Second opinions and proofs | 2.0 | 10 | 1 |
    | Day 16 | Second opinions and proofs | 1.0 | 2 | 1 |
    | Day 17 | Improve, then trim | 12.8 | 5 | 1 |
    | Day 18 | Improve, then trim | 6.7 | 0 | 1 |
    | Day 19 | Improve, then trim | 16.6 | 1 | 1 |
    | Day 20 | Improve, then trim | 24.0 | 4 | 1 |
    | Day 21 | Improve, then trim | 18.9 | 8 | 1 |
    | Day 22 | Ready for other people | 24.0 | 14 | 3 |
    | Day 23 | Ready for other people | 24.0 | 15 | 7 |
    | Day 24 | Release | 22.9 | 78 | 14 |
    | Day 25 | Release | 3.3 | 10 | 4 |

## The port { #port }

Days 1–2
{ .history-dateline }

Codex first measured treams: about 22,700 lines of Python and Cython. Fifteen
minutes in, the brief became a full rewrite in Rust, soon extended to strong
tests and the best possible speed and memory use. One agent did the work alone.
Asked about slowdowns, it reported one case running 12% slower than treams.

<div class="history-quote" markdown>

ok that's unacceptable. treams-rs must make no compromises on performance. fix that, and then finish the rewrite

Human, day 2 ·
[In the conversation](conversations.md?c=port&at=54077)

</div>

The agent fixed the slowdown and
[had ported every treams feature on its list](conversations.md?c=port&at=104454).

2 conversations · Codex app · gpt-6-astra, effort level high, then xhigh · no helper agents · 50 changes
{ .history-facts }

[Conversation](conversations.md?c=port) ·
[Messages](messages.md?phase=port) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/commits/782d2ee)
{ .history-links }

## Beyond the port { #beyond }

Day 3
{ .history-dateline }

The brief grew to include a browser version, GPU calculations, connections to
JAX and PyTorch, reproductions of published results, and benchmark plots against
treams, for speed and for accuracy:

<div class="history-quote" markdown>

we also need some convincing benchmark plots regarding accuracy to completely overwhelm any "oh but this ai slop this cant possibly be correct"

Human, day 3 ·
[In the conversation](conversations.md?c=port&at=140564)

</div>

Checks against independent high-precision values showed treams-rs returning zero
Mie coefficients for a large metallic sphere. A helper agent's fix reached
treams-rs on day 10.

1 conversation · Codex app · gpt-6-astra, effort level ultra · 61 helper agents · 15 changes
{ .history-facts }

[Conversation](conversations.md?c=port&at=115873) ·
[Messages](messages.md?phase=beyond) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/782d2ee...5d7e06f)
{ .history-links }

## Finishing the core { #core }

Days 10–11
{ .history-dateline }

Two helper agents on gpt-5.6-luna ran the long correctness and speed checks,
on one condition:

<div class="history-quote" markdown>

make sure NOT to let the luna agent make any judgement calls though

Human, day 10 ·
[In the conversation](conversations.md?c=core&at=784875)

</div>

On Linux, all 615 correctness cases passed, and treams-rs was faster than
treams 0.4.5 in 664 timed comparisons, by a median 5.1× over 527 cases; four
other cases stayed slower.
After the Python interface was redesigned, fresh agents with only the installed
package passed 118 of 120 attempts at ten test tasks, up from 110.

3 conversations · Codex in a terminal · gpt-6-astra, effort level medium, high and xhigh · 9 helper agents, 2 of them on gpt-5.6-luna, effort level medium · 10 changes
{ .history-facts }

[Conversation](conversations.md?c=core) ·
[Messages](messages.md?phase=core) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/5d7e06f...f93276f)
{ .history-links }

## Second opinions and proofs { #review }

Days 13–16
{ .history-dateline }

Claude Code reviewed treams-rs and judged its numerical core and checks
strong. In a second conversation it wrote models of six routines in Lean, a
language for computer-checked proofs, and proved properties over exact
arithmetic; one proof exposed a bug that could drop diffraction orders exactly
on the cutoff.

<div class="history-quote" markdown>

ok so bottom line here? did you implement anything useful or was this just an experiment

Human, day 16 ·
[In the conversation](conversations.md?c=lean&at=1259155)

</div>

The agent's answer: the bug fix was the clear gain; the rest mostly added
confidence.

2 conversations · Claude Code · claude-opus-5-5, effort level xhigh · 1 helper agent · 4 changes
{ .history-facts }

[Conversation](conversations.md?c=lean) ·
[Messages](messages.md?phase=review) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/f93276f...76ed0e8)
{ .history-links }

## Improve, then trim { #improve }

Days 17–21
{ .history-dateline }

The task was to improve treams-rs while keeping every feature and keeping
results equal up to rounding. The agent later reported that the code had grown
by about 30,000 lines, mostly tests and reference data.

<div class="history-quote" markdown>

ok that seems like too much. without regressing performance or correctness, cut the code and make it more maintainable

Human, day 21 ·
[In the conversation](conversations.md?c=improve&at=1687742)

</div>

The cut made the code about 4,000 lines shorter; every numerical function gave
identical results at the same speed.

1 conversation · Claude Code in the cloud · claude-opus-5-5, effort level xhigh with ultracode · about 240 helper agents, about 230 of them in 34 planned groups · part of the 8 changes condensed on day 24
{ .history-facts }

[Conversation](conversations.md?c=improve) ·
[Messages](messages.md?phase=improve) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/76ed0e8...8251007)
{ .history-links }

## Ready for other people { #prepare }

Days 22–23
{ .history-dateline }

On day 21 the goal became publishing treams-rs, with code that
others can read and maintain. On day 22 the lead agent started
two conversations, for the Python package and the documentation.

<div class="history-quote" markdown>

i also want us to better utilize other sessions. if we are constrained by hardware then just start new sessions and coordinate with them

Human, day 22 ·
[In the conversation](conversations.md?c=improve&at=1811682)

</div>

Three more conversations in the cloud, started by hand on day
23, took on separate tasks; one rewrote a routine from published
formulas so that the MIT license alone applies.

7 conversations, 6 of them new · Claude Code in the cloud and the Codex app · claude-opus-5-5, effort level xhigh, also with ultracode; gpt-6-astra, effort level xhigh · about 380 helper agents, about 370 of them in 72 planned groups · part of the 8 changes condensed on day 24
{ .history-facts }

[Conversation](conversations.md?c=improve&at=1770664) ·
[Messages](messages.md?phase=prepare) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/76ed0e8...4068d6c)
{ .history-links }

## Release { #release }

Days 24–25
{ .history-dateline }

On day 24 the work in the cloud stopped on request; four
conversations left written handoff notes. Codex, running locally, brought the
remaining lines of work together, removed personal details and condensed the
history of changes. A plain-language rule, first set for the documentation on
day 22, was extended to all text:

<div class="history-quote" markdown>

also make sure all language is objective, concise, jargon-free (especially programming. physics is ok) approachable but not verbose. basically respect the readers’ time, no slop grenades

Human, day 24 ·
[In the conversation](conversations.md?c=release&at=1956748)

</div>

After final reviews, version 0.1.0 was published; version 0.1.1,
with faster calculations, followed on day 25.

17 conversations, 11 of them new · Codex and Claude Code · claude-opus-5-5, effort level xhigh with ultracode; gpt-6-astra, effort level xhigh, ultra and max · about 230 helper agents, about 100 of them in 18 planned groups · 30 changes, 8 of them condensed from days 17–24
{ .history-facts }

[Conversation](conversations.md?c=release) ·
[Messages](messages.md?phase=release) ·
[Changes on GitHub](https://github.com/yaugenst/treams-rs/compare/76ed0e8...1340631)
{ .history-links }

## How the work was split { #split }

<div class="history-split" data-history="split" markdown>

- **Alone.** One agent did the port.
  [See it in a conversation](conversations.md?c=port)
- **With helper agents.** An agent handed parts of the work to helper agents,
  some of which started their own.
  [See it in a conversation](conversations.md?c=port&at=115964)
- **Planned groups.** Helper agents worked in stages from a written plan,
  often: make a change, look for faults, fix them.
  [See it in a conversation](conversations.md?c=improve&at=1349468)
- **Parallel conversations.** One conversation started two others and briefed
  them in writing; they reported back through files in the repository.
  [See it on the map](conversations.md?phase=prepare)

</div>

[All conversations, day by day](conversations.md)

## Models and effort levels { #models }

The effort level sets how long the model may reason before it answers.

<div class="history-models" markdown>

| Tool | Model | Effort levels | Conversations | Phases |
| --- | --- | --- | ---: | --- |
| Codex | gpt-6-astra | medium, high, xhigh, ultra, max | 15 | The port; Beyond the port; Finishing the core; Ready for other people; Release |
| Claude Code | claude-opus-5-5 | xhigh, also with ultracode | 10 | Second opinions and proofs; Improve, then trim; Ready for other people; Release |
| Codex | gpt-5.6-luna | medium | – | Finishing the core (two helper agents) |

</div>

## Ways of working { #ways }

**Running and judging kept apart.** The helper agents that ran the long checks
made no judgements; the lead agent judged the results.
[In the conversation](conversations.md?c=core&at=787952)

**Faults looked for on purpose.** Separate agents looked for faults in the work
from day 3, and in most changes from day 17.
[In the conversation](conversations.md?c=strict&at=2035115)

**Evidence a skeptic would accept.** Results were checked against treams,
high-precision values, physical laws, published results and proofs; slower or
failed cases were reported, not dropped.
[In the conversation](conversations.md?c=lean&at=1214244)

**Written handoffs.** When work was paused or handed on, a written note of its
state went with it.
[In the conversation](conversations.md?c=release-readiness&at=1945508)

## What is left out { #left-out }

Command output, file contents, the agents' reasoning and personal details are
left out, as are six conversations that were not part of building
treams-rs.
[About the records](about.md#what-is-left-out) has the details.

[Messages](messages.md){ .md-button .md-button--primary }
[Conversations](conversations.md){ .md-button }
