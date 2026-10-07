# About the records

<!-- Generated file. Do not edit by hand. -->

The conversations in this section are condensed from the records that Codex and
Claude Code kept.
{ .lead }

## Days

Day 1 is the day of the first message. Each run of quiet days is
shown as one gap.

## Where the records come from

| Tool | Where | Conversations | Helper agents |
| --- | --- | ---: | ---: |
| Codex app | local | 11 | 182 |
| Claude Code | in the cloud | 6 | 652 |
| Codex in a terminal | local | 4 | 12 |
| Claude Code | local | 4 | 71 |

## What they contain

Each conversation shows:

- messages from the human and answers to the agents' questions;
- the agents' replies and progress notes;
- one line for each command an agent ran (except the agents of planned groups),
  each edit with the files it changed and each web search or page it opened;
- the helper agents an agent started, their tasks where readable, and what each
  reported back;
- for each planned group: its plan, its stages, its agents and how each run
  ended;
- messages passed on from other conversations, and the changes on GitHub that
  came from the conversation.

Changes of effort level, stops, restarts of the coding tool, usage limits and
scheduled check-ins appear as one-line notes; restarts of the computer in the
cloud show only in the agents' replies.

## What is left out { #what-is-left-out }

The conversations leave out what the commands printed, file contents, the agents' reasoning, images and the standing instructions the tools add on their own. Personal details are replaced by a short description in italics with a dotted underline (plain text in command lines), for example *[e-mail address]*{ .history-r }, or cut and marked *[…]*{ .history-r }. Folders outside the project are named by role, such as "scratch", or read *[a private folder]*{ .history-r }. Passages and lines about accounts, private computers, other projects, other people, the agents' own settings and notes, or other tasks are left out, and a line marks each place, for example "Left out: 2 commands that read the agent's own notes." Six [conversations that were not part of building treams-rs](#conversations-not-included) are not included. Messages are otherwise quoted as typed; both versions of an edited message are kept.

## How the figures were counted

| Figure | Definition | Value | Left out |
| --- | --- | ---: | ---: |
| Days | Days from day 1 to day 24, the day version 0.1.0 was published, counting both. | 24 | – |
| Working days | Days among those on which the published conversations show agents working. | 16 | – |
| Messages from the human | Messages typed into the published conversations and shown there. A message edited and sent again counts once. Answers to the agents' questions, pasted text and text passed on between conversations are not counted. | 233 | 21 |
| Words in those messages | Counted between spaces, rounded to the nearest hundred. | 4,900 | – |
| Days with messages | Days with at least one such message. | 16 | – |
| Answers to questions | Options chosen in reply to the agents' multiple-choice questions. | 28 | 1 |
| Conversations | Published conversations, including those started by an agent and those shown only as a summary. Helper agents are not counted. | 25 | 6 |
| Helper agents | Helper agents of the published conversations, including the agents of planned groups; a restarted agent counts once. Rounded to the nearest ten. | 920 | 4 |
| Planned groups | Planned groups that started. A group that was stopped and resumed counts once. | 124 | – |
| Agents of planned groups | A restarted agent counts once. Rounded to the nearest ten. | 700 | – |

## Words used in this section

| Word | Meaning |
| --- | --- |
| conversation | One exchange with an agent and everything the agent did in it. Some were started by another agent. |
| agent | An AI model at work in a coding tool. It reads files, runs commands and changes code. |
| helper agent | An agent that another agent started for part of the work. It reports back when it is done. |
| lead agent | The agent of a conversation that gave tasks to helper agents or other conversations. |
| planned group | Helper agents that an agent started from a written plan, in stages, for example: make a change, look for faults in it, fix them. |
| effort level | A setting of the coding tool: how long the model may reason before it answers. |
| ultracode | A Claude Code setting that has the agent run larger tasks as planned groups, at effort level xhigh. |
| progress note | A short message an agent wrote while it worked. |
| scheduled check-in | A time the agent set to wake itself and check on work in progress. |
| change | One entry in the public history of the code on GitHub. |
| passed on | Carried from one conversation to another, by hand or by an agent. |
| quiet day | A day with nothing published. |

## Changes on GitHub { #changes-on-github }

On day 24, before the first release, the history of changes was
rewritten: personal details were removed, the 79 changes up to
day 16 kept their dates and descriptions, and about 1,570 later
changes were condensed into 8. About 210 changes that only coordinated the
agents were not published.
[`benchmarks/history-provenance.json`](https://github.com/yaugenst/treams-rs/blob/63b181ab6de15d3b3a359c46e3698905391dea57/benchmarks/history-provenance.json)
maps the old identifiers of the kept changes to the new ones. Identifiers quoted
in replies carry their public counterpart, for example "552b672 (now 782d2ee)".

## Known gaps

- Codex stored the tasks and messages that agents sent each other in encrypted
  form; the helper agents' final reports are shown.
- Planned-group agents' own conversations are not shown; for runs in the cloud
  they are not in the records at all.
- The agents' reasoning was stored encrypted, apart from short headings and
  progress lines, which are not shown.

## Conversations not included

| Day | Tool | Reason |
| --- | --- | --- |
| Day 6 | Codex app | A copy of the first conversation, continued for another purpose. |
| Day 12 | Codex in a terminal | A question whether a code-rewriting tool would have helped the port. |
| Day 13 | Claude Code | Setting up a tool. |
| Day 21 | Claude Code | Another task. |
| Day 25 | Codex app | A fault in the coding tool. |
| Day 25 | Codex app | Unrelated work. |

## Corrections

AI agents wrote this section from the records. Corrections are welcome as an
[issue on GitHub](https://github.com/yaugenst/treams-rs/issues).
