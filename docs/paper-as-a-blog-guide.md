# The Paper-as-a-Blog Style Guide

A template for turning research papers into blog posts that read like a private walkthrough from the team — the thinking, the stumbles, the surprises — rather than the official account. The rigor stays; the voice becomes human.

This guide distills patterns from two published examples in this style: a Microsoft Research blog on the *Memento* system (context management for LLMs) and a companion blog on the *Universal Verifier* (evaluating computer-use agents). Instructions are general; examples are drawn from both.

---

## 1. Open with a TL;DR that teases, not summarizes

Start the post with a numbered list of 3–5 findings. Phrase each to create curiosity rather than close it off. Tell the reader *what* you found while deliberately withholding *how*. This is the opposite of an abstract, which tries to be complete.

**Instruction:** Write each finding as a one-line claim that raises a question. If the reader doesn't feel the urge to keep reading after the TL;DR, it's too complete. Mix practical results with surprising or counterintuitive findings — the surprising ones are the hooks.

**Example A (from Memento — too informative):**
> 1. We trained models to compress their chain-of-thought into summaries, reducing KV cache by 2–3x.

**Example A (rewritten — creates tension):**
> 1. You can teach a model to segment its own chain-of-thought, compress each piece into a dense "memento," and reason forward from that alone.
> 2. Erased reasoning blocks don't fully disappear — their ghost persists in the KV cache and the model quietly relies on it.

The second bullet makes you *need* to understand what "ghost" means. That's the hook.

**Example B (from Universal Verifier):**
> 1. Good verifiers rely on rubric design — and good rubrics must have specific, non-overlapping criteria [...] Good rubric design alone accounts for roughly half the gains.
> 2. Auto-research agents can't fully replace human experts in verifier design yet — but they reach ~70% of expert quality in just 5% of the time.

Here, bullet 1 makes you wonder "what makes rubrics go wrong?" and bullet 2 sets up a human-vs-AI tension that pulls you through the entire second half of the post.

**Tip:** End the TL;DR list with a link to the paper and code/data. This signals openness and lets impatient readers jump straight to the artifact.

---

## 2. Use conversational first-person plural — like a lab thinking out loud

Write as "we" and let the team's thought process show. Use phrases that reveal deliberation, uncertainty, and surprise. The reader should feel like they're overhearing researchers talk through a whiteboard, not reading a polished report.

**Instruction:** Narrate the research as a discovery process. Include the internal reasoning that led to decisions — the concerns, the hypotheses, the "aha" moments. Avoid the passive voice of formal papers.

**Example A (from Memento):**
> We kept coming back to a concern: if the model is secretly relying on residual KV states from blocks we supposedly erased, how much is that actually doing? So we ran an ablation.

**Example B (from Universal Verifier):**
> We spent 96 experiments and several weeks building what we call the Universal Verifier. What we ended up with is less a single trick and more a set of learned design principles, each addressing a failure mode we discovered.

Both examples share a thought process, not just a result. The Memento version shows a worry leading to an experiment; the Verifier version shows the shape of the project's evolution.

**What to avoid:**
> An ablation study was conducted to determine the contribution of the implicit KV channel to overall performance.

This is paper voice. It's accurate but lifeless.

---

## 3. Make failure a structural beat, not a footnote

Walk the reader through what you tried first and why it failed, *before* presenting the approach that worked. Failed approaches become narrative turning points that make the eventual solution feel earned and trustworthy.

**Instruction:** For each major design decision, write a short paragraph about the approach you tried first and why it didn't work. Use direct, unhedged language — "This does not work" is better than "This yielded suboptimal results."

**Example A (from Memento):**
> We tried the obvious thing first: paste a reasoning trace into a frontier model and ask it to segment and summarize. This does not work — not even if you cut the trace into pieces first.
>
> What does work is decomposing the problem. An LLM scores each inter-sentence boundary from 0 to 3 (a local question LLMs handle well). The global optimization is then handled by dynamic programming. This is the kind of thing that's hard for an LLM to zero-shot, but where good old dynamic programming just works.

**Example B (from Universal Verifier):**
> Initially, we generated and scored rubrics in one pass, but this rarely caught subtle hallucinations. So, we separate rubric generation from scoring, and decomposed scoring the rubric into two stages: with and without screenshot evidence. Discrepancies between the two stages surface hallucinations that a single-pass scorer would miss.

Notice the blunt "this does not work" in A and "this rarely caught" in B. These are signals to the reader that the author is being straight with them. The failed approach also gives the reader context for why the working approach is shaped the way it is.

---

## 4. Frame section headers as rhetorical questions

Replace dry labels ("Methodology," "Experiments") with genuine questions the reader is likely already asking. This transforms the reading experience from skimming a document into following an argument.

**Instruction:** Before writing each section, ask: "What is the reader wondering at this point?" Use that as the header.

| Instead of this | Write this |
|:--|:--|
| Methodology | How do you build a rubric that doesn't lie to you? |
| Experiments | Does all this engineering actually show up in the numbers? |
| Ablation Study | What happens if you wipe the KV cache completely? |
| Discussion | What did this project actually teach us? |
| Context Management | What do you do when the trajectory is 50 screenshots long? |
| Automated Evaluation | Can an AI build a CUA verifier on its own? |

The reader's experience should feel like a conversation where each question is answered, then a new question naturally arises from the answer.

---

## 5. Use analogies that do real explanatory work

Good analogies in this style aren't decorative — they compress a concept into something the target audience already understands. Aim for precision, not color.

**Instruction:** For each core concept, find an analogy aimed at your audience's existing knowledge that genuinely reduces the explanation needed. If the analogy requires its own explanation, it's the wrong one.

**Example A (from Memento — precise, does real work):**
> Think of a memento as a lemma: a minimal, self-contained statement that captures what was established in a block of reasoning, so that future steps can build on it without re-deriving anything.

**Example B (from Universal Verifier — visual/intuitive):**
> Too many screenshots force the model into a needle-in-a-haystack problem that scales poorly with trajectory length.

The "lemma" analogy works because the ML audience knows exactly what a lemma is. The "needle-in-a-haystack" analogy works because it immediately conveys both the difficulty and the scaling problem without further explanation.

**What to avoid (decorative, doesn't help):**
> A memento is like a sticky note the model leaves for itself.

This is vague — it doesn't convey the precision or self-containedness that makes the concept interesting.

---

## 6. Set stakes before presenting results

Before showing any number, explain *why that number matters* — what concern it resolves, what hypothesis it tests, what would be worrying if it went the other way. The reader should already be wondering about the result before they see it.

**Instruction:** For each key result, write a paragraph that walks the reader into the question. Then deliver the number. The result should land as an answer to something the reader was already asking.

**Example A (from Memento):**
> The simplest approach — and the one that would make our lives much easier — is restarts: at each compaction step, wipe the KV cache completely and re-encode from mementos alone. If the model's reasoning is truly captured by the memento text, restarts should be fine.
>
> They're not. Restarting drops accuracy by 15 percentage points on MATH-500.

**Example B (from Universal Verifier):**
> You might wonder whether this is just a stronger backbone model doing the work. We tested that. Upgrading WebVoyager from GPT-4o to GPT-5.2 does drop its outcome false positive rate from 0.45 to 0.10 — but it also dramatically increases its false negative rate (0.24 to 0.44), and overall kappa improves only modestly. The UV's advantage is architectural, not model-driven.

Both set up the concern first ("Is this just restarts?" / "Is this just a better model?"), then deliver the answer. The numbers hit harder because the reader was already wondering the same thing.

---

## 7. Give concrete numbers, but wear them lightly

Weave key metrics into the narrative flow rather than isolating them in tables or dense results sections. The numbers should feel illustrative — evidence in service of the story — not exhaustive.

**Instruction:** Pick the 4–6 most important numbers. Embed each in a sentence that gives it context and meaning. Save comprehensive tables for the paper itself.

**Example A (from Memento — narrative):**
> On average, a full reasoning trace runs about 10,900 tokens; the mementos compress that to roughly 1,850 — about a 6x reduction. Standard SFT on around 30K of these examples was enough to teach the behavior.

**Example B (from Universal Verifier — comparative):**
> The UV achieves a Cohen's kappa of 0.64 on the internal set and 0.58 on Browserbase, compared to 0.44/0.26 for WebJudge and 0.31/0.13 for WebVoyager. More importantly, the UV's false positive rate is 0.01 on the internal set — essentially zero.

Both embed numbers in sentences that tell a story. The Memento version says "here's how much compression, here's how little training." The Verifier version uses comparison to make absolute numbers meaningful.

---

## 8. Anchor abstract ideas with concrete examples

When describing a general principle or failure mode, follow it immediately with a specific, named instance from your experiments. Concrete examples ground the reader and make abstract categories feel real.

**Instruction:** After stating any general claim, provide at least one "for example" or "for instance" with a specific task, specific numbers, or a specific screenshot. Name the real entities involved.

**Example A (from Universal Verifier — phantom criteria):**
> LLM-generated rubrics frequently introduce requirements that were never stated in the task. For example, given a multi-step task, our early rubric added criteria for the price and address of a hotel — neither of which the user requested for the primary intent of finding a coffee shop near the hotel. The agent completed the actual task but scored 2/8 because it "failed" those phantom criteria. After fixing the rubric to match only what was asked, the same trajectory scored 16/18 — a success.

**Example B (from Universal Verifier — hallucination):**
> The agent claimed a model exhibited "+6.2% CIDEr score" when the actual paper showed "+2.8% in CIDEr" — a discrepancy even human reviewers missed.

**Example C (from Universal Verifier — auto-research insights):**
> After observing the verifier failing trajectories over minor issues — things like "inferring most Coursera courses can be audited for free is unsubstantiated" or "not disambiguating apartment from rental-unit" — the expert deduced general scoring rules like "separate nitpicks from critical failures."

The 2/8 to 16/18 flip in Example A is devastating — it shows exactly how phantom criteria corrupt evaluation. The CIDEr numbers in B make the hallucination visceral. The Coursera and apartment examples in C make "minor issues" concrete rather than hand-wavy.

---

## 9. Use figures to show, not decorate

Every figure should carry explanatory weight — it should make a point that is harder to make in prose alone. Place each figure immediately after the paragraph that motivates it, and write a caption that tells the reader what to notice.

**Instruction:** For each figure, ask: "Could I delete this and lose nothing?" If yes, cut it. If no, place it right after the sentence that raises the question the figure answers. Write the caption as a complete sentence that states the takeaway, not just a label.

**Example A (from Universal Verifier — relevance matrix):**
> *Figure 6: A screenshot relevance matrix. Each cell scores how relevant a screenshot is to a specific rubric criterion, enabling targeted evidence retrieval rather than flooding the context window.*

**Example B (from Universal Verifier — error isolation):**
> *Figure 3: An example of error isolation in practice. The agent incorrectly identified "Timberlake" as the longest last name when "Kirkpatrick" is correct — but the error does not cascade to downstream criteria about reporting net worth.*

Both captions state what the reader should take away from the figure. Caption B in particular tells a micro-story: here's what went wrong, and here's what *didn't* go wrong because of the design choice.

**What to avoid:**
> *Figure 3: Ablation results.*

This forces the reader to decode the figure alone. The caption should do half the work.

---

## 10. Provide working links to artifacts

Include links to the paper, code, data, and any live demos or comparison tools. These signal openness, let curious readers go deeper, and differentiate the blog from a press release.

**Instruction:** Link to the full paper and code/data repository in the TL;DR section so they're visible immediately. Embed additional links inline wherever they add value — to specific datasets, to comparison visualizations, to related work.

**Example (from Universal Verifier):**
> Full paper is available [here](https://arxiv.org/pdf/2604.06240v1) and Code and Data are available at https://github.com/microsoft/fara

> You can see how our rubrics evolved on WebTailBench here: https://microsoft.github.io/fara/docs/webtailbench_rubric_comparison.html

The rubric comparison link is especially good — it's not just "here's our repo" but "here's a specific interactive artifact that lets you see the thing we just described." This rewards the engaged reader and builds trust.

---

## 11. Narrate design tradeoffs as genuine deliberation

When the team had to choose between competing approaches, don't just present the winner. Walk the reader through the tradeoff: the pros of the path not taken, why it was tempting, and what tipped the decision.

**Instruction:** For major architectural decisions, write a short "fork in the road" passage. Name both options, explain why the rejected one was attractive, and then explain the deciding factor.

**Example A (from Memento):**
> This distinguishes Memento from approaches like InftyThink and Accordion-Thinking, which discard original tokens and rebuild context from summary text alone. We went the other way: after a memento is generated, the preceding thinking block is masked from attention and its KV cache entries are flushed — but we don't restart from scratch. This was the harder engineering path, but it preserves the residual information channel that turned out to matter more than we expected.

**Example B (from Universal Verifier):**
> From a model training perspective, it doesn't make sense to penalize an agent for things outside its control, but from a metrics perspective, we still need to know if a task was completed.

Example A shows a fork between two architectures with a clear reason for the choice. Example B is more compact — a single sentence that captures the tension between two valid goals that shaped the process/outcome split.

---

## 12. Close with a reflection, not a conclusion

Don't end with the standard academic formula of "limitations, future work, and broader impacts." Step back and reflect on what the project *taught* you — elevate the most interesting conceptual insight above the practical contributions.

**Instruction:** In the final section, identify the single most surprising or perspective-shifting thing you learned. Write about *that*, not about what you built.

**Example A (from Memento):**
> The most interesting thing Memento taught us wasn't about efficiency — it was about what "erasing" means in a transformer. When you remove a block of reasoning from the token stream but leave its KV cache entries behind, the model keeps using them. The erased blocks aren't really gone.

**Example B (from Universal Verifier):**
> The thing that stays with us is how much of verification is judgment — and how poorly that judgment decomposes into simple rules. [...] Phantom criteria sound like an easy problem to fix until you realize how systematically LLMs hallucinate requirements. Separating process from outcome sounds like a clean abstraction until you're staring at a trajectory where the agent did everything right and the website just... didn't work.

Both reflections elevate a conceptual surprise over the engineering contribution. The Memento coda is about erasure being incomplete; the Verifier coda is about judgment resisting automation. These are the ideas readers will remember.

---

## 13. End on a punchy, quotable line

The last sentence should be a callback to the post's most surprising or important finding, distilled into something crisp and memorable. Closer to a journalist's kicker than anything in a paper.

**Instruction:** Write a final sentence that is: (a) no more than ~20 words, (b) a direct callback to a specific finding, and (c) slightly provocative or reframing. If it could work as a tweet, it's the right length.

**Example A (from Memento):**
> Stop flushing your KV cache. Your model remembers more than you think.

**Example B (from Universal Verifier):**
> The verifier doesn't just tell you whether the agent succeeded. It tells you how it failed — and whether the failure was even the agent's fault.

Example A is an imperative that reframes a technical finding as practical advice. Example B reframes what a verifier *is* — not a binary judge, but a diagnostic tool. Both are quotable and memorable.

---

## 14. Draw practical lessons from automated experiments

If you ran any automated optimization, search, or AI-assisted experimentation, extract the operational insights — the specific, transferable things you learned about how to do this kind of work, not just the final result.

**Instruction:** Rather than just reporting "the AI reached X% of expert quality," describe the specific patterns that emerged — what kinds of changes worked, what didn't, and what surprised you about the process.

**Example (from Universal Verifier):**
> A few things stood out watching the auto-research agent iterate. First, code changes consistently beat prompt additions when prompts were already long — the single most impactful change was injecting rubric scores directly into context, since it provided quantitative calibration without adding more text for the model to parse. Second, forcing explicit rule-checking helped: by naming rules in a mandatory output field, the LLM was far more likely to actually apply them rather than silently ignore instructions buried in a long prompt. Third, concrete tests beat abstract principles — "would the user say this is useful?" proved more actionable than vague guidance like "be reasonable about minor issues."

This paragraph is arguably the most useful part of the post for practitioners. Each lesson is specific enough to apply directly.

---

## Summary checklist

Before publishing, verify that your post:

- [ ] Opens with findings that create curiosity, not closure
- [ ] Links to the paper and code/data in or near the TL;DR
- [ ] Uses "we" and narrates the research as a thinking process
- [ ] Includes at least one explicit dead end or failed approach
- [ ] Frames every section header as a question the reader is asking
- [ ] Contains at least one analogy aimed precisely at the target audience
- [ ] Sets up stakes before every major result
- [ ] Embeds numbers in narrative sentences, not tables
- [ ] Grounds every general claim with a specific, named example
- [ ] Places figures right after the paragraph that motivates them, with takeaway captions
- [ ] Includes inline links to artifacts, tools, or comparisons where they add value
- [ ] Shows at least one design tradeoff as genuine deliberation
- [ ] Ends with a reflection on what was learned, not a list of future work
- [ ] Closes on a single punchy, quotable sentence
- [ ] Extracts practical, transferable lessons from any automated experiments
