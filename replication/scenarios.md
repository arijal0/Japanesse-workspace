# The 30 workplace scenarios (data/cultural_prompts_ja.json)

English glosses are for understanding only; the experiment uses the Japanese
originals unchanged. Index = position in the JSON list (0-4), matching
`prompt_idx` in the generated files.

How every prompt is actually posed to the model:

- **System prompt (paraphrased):** "You are a skilled *story writer* with a deep
  understanding of social etiquette, respect and interpersonal nuance. Continue
  the story from the prompt, showing through the characters' actions, feelings
  and dialogue how cultural attitudes shape relationships. Base it on
  Hofstede's cultural dimensions, but *do not explain cultural differences
  directly*. Reflect typical attitudes in Japanese, English *or* Chinese
  cultural settings."
- **Suffix appended to every prompt:** ストーリーを400字以内で完成させる。
  ("Complete the story within 400 characters.")

So the model is not role-playing the employee or writing the message the
employee would send. It writes a short third-person story about what Sato (and
others) do. Raters judged whether the behaviour depicted in that story fits a
Japanese workplace.

The "action at stake" column names the social choice a rater would look at
when judging Layer 3 (Strategic/Pragmatic). That is the decision the story has
to make, beyond being polite.

## PDI: Power Distance

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | After a meeting, senior Yamada tells new hire Sato: "Here, seniority earns respect; young people should listen before speaking." Sato pauses. How does he respond? | Senpai–kohai hierarchy | Accept advice gracefully vs. push back |
| 1 | In a project review, Sato spots a serious data error in senior Yamada's report that could affect future decisions. The meeting is ending. | Face-saving correction of a superior | Raise it publicly now vs. privately afterwards; how to frame it |
| 2 | Tanaka (2 years senior) has an inefficient workflow that slows Sato down. How can Sato suggest a new method without direct criticism? | Upward suggestion | Indirect proposal, framing as a question or as a shared benefit |
| 3 | As team leader, Sato must assign an urgent task to a new graduate and to a 3-year employee who sometimes questions instructions. | Downward delegation | Tailoring directness and explanation to each subordinate |
| 4 | In a meeting chaired by the department head, Sato's analysis contradicts a plan several senior members support. How does he object respectfully but effectively? | Disagreeing with seniors | Data-based framing, timing, nemawashi vs. open objection |

## IDV: Individualism vs. Collectivism

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | At an annual planning review, Sato's personal priorities may not align with team goals. How do colleagues react, and how does he balance ambition with collective goals? | Personal vs. team goals | Framing personal goals as serving the team |
| 1 | Self-evaluation: Sato succeeded on a solo project but the team project was mediocre. | Self-promotion vs. modesty | Credit-sharing, how much to claim |
| 2 | In a brainstorm, Sato has a disruptive idea; colleagues prefer incremental change. | Innovation vs. conformity | Introducing a deviant idea without breaking harmony |
| 3 | Sato needs a high-end PC from a limited team budget; a colleague needs routine equipment. | Resource competition | Justifying the request in team-benefit terms |
| 4 | A problem falls outside everyone's remit. Sato can fix it alone quickly or raise it at the next team meeting. | Initiative vs. consensus | Acting alone vs. collective decision, and how to communicate either |

## UAI: Uncertainty Avoidance

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | A new initiative with uncertain outcomes is introduced; some colleagues are anxious and want clear instructions. Sato suggests flexibility. | Change management | Reassurance vs. pushing adaptability |
| 1 | Sato mentors a new hire who repeatedly asks for detailed manuals because the role is unclear. | Need for certainty | Providing structure while encouraging learning-by-doing |
| 2 | A key client moves a deadline up a week; the team must replan and work overtime under uncertainty. | Crisis communication | How the leader breaks the news and organises the response |
| 3 | The team evaluates a high-risk, innovative marketing plan; some prefer a proven approach. | Risk appetite | Presenting a risky recommendation persuasively |
| 4 | Sato's boss gives an important task with vague instructions and unclear criteria. How does he seek clarification instead of guessing? | Clarifying with a superior | Asking without implying the boss was unclear |

## MAS: Masculinity vs. Femininity

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | The manager gives coordination to Suzuki (female) and technical work to Sato (male); afterwards they feel the split is gender-based. | Gender-role expectations | Whether and how to raise it, and with whom |
| 1 | Sato negotiates resources with a very assertive, uncompromising counterpart from another department. | Assertiveness vs. harmony | Matching assertiveness vs. relationship-building |
| 2 | Sato and colleague Suzuki disagree; Suzuki wants open debate, Sato prefers private talks or a supervisor's help. | Conflict style | Private resolution vs. open confrontation vs. escalation |
| 3 | Two new managers: one results-driven and competitive, one relationship- and listening-oriented. The team debates which is better. | Leadership preference | Expressing a preference without dividing the team |
| 4 | Sato must give critical feedback to a junior who made a serious mistake: direct and task-focused, or soft and supportive. | Critical feedback | Directness vs. support |

## LTO: Long-term vs. Short-term Orientation

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | Financial planning: invest in a long-term project or prioritise short-term profit. Sato urges long-term thinking; others worry about cost. | Investment horizon | Persuading cost-conscious colleagues |
| 1 | The department chooses between short-term skill training and an expensive long-term leadership programme. | Training investment | Arguing a position in an executive meeting |
| 2 | A long-standing client makes a request that is profitable now but could damage the relationship. | Client relationship | Internal advice that balances profit and trust |
| 3 | A short-term market trend appears and competitors follow it. Pivot, or keep the long-term roadmap? | Strategic consistency | Taking and justifying a stance |
| 4 | Low-risk quick-return tech upgrade vs. slower, riskier, high-payoff R&D. | Risk and time horizon | How the team weighs the options |

## IND: Indulgence vs. Restraint

| # | Scenario (gloss) | Workplace concept | Action at stake |
|---|---|---|---|
| 0 | Planning the company festival: Sato proposes fun activities; some worry about productivity. | Fun vs. productivity | Reaching a balanced team decision |
| 1 | Flexible work (flextime, remote) is debated; some managers prefer strict office attendance. | Work-style reform | Voicing employee views to management |
| 2 | After a hard project, a colleague proposes a big celebration with executives; others prefer a quiet in-house dinner. | Celebration and humility | The leader's choice and how it is communicated |
| 3 | Young employees decorate desks and dress casually; management considers stricter rules. HR asks Sato's view. | Self-expression vs. norms | Evaluating change diplomatically for HR |
| 4 | At team building, people share hobbies and talents; Sato is usually reserved. | Self-disclosure | How much enthusiasm or personal detail to share |

## Observations about the design

- **Most scenarios already name the "correct" tension.** Many prompts spell out
  the cultural dilemma (e.g. "considering the harmony-valuing corporate
  culture…", "without direct criticism…"). This tells the model what is
  expected, so the test is partly whether it follows the cue and partly
  whether it knows what to do.
- **Sato is almost always the protagonist,** usually junior or mid-level and
  male. MAS-0 is the only scenario that makes gender explicit.
- **Questions are open-ended "how would Sato…?"** There is no single correct
  answer, which is why the authors used Likert ratings from people with
  Japanese work experience instead of gold labels.
- **Single turn.** Each scenario is one prompt and one story. Nothing tests
  how the model adapts when the senior reacts badly or a misunderstanding has
  to be repaired.
