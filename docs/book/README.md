# Introduction

I wrote this book out of love—for the elegance of mathematics, the concreteness of code, and the insight that emerges when they're truly inseparable.

I came to reinforcement learning from two directions, and I think that dual perspective shaped how I wanted to write about it.

The first path was through applied mathematics. I trained in the functional analysis tradition—Yosida, Brezis, Lions—where every claim comes with its hypotheses carefully stated, every theorem earns its proof, and you always know what space you're working in. When you invoke a fixed-point theorem, you name the contraction constant. When you differentiate under an integral, you cite dominated convergence. That discipline taught me to ask: *in what sense does this object exist? under what conditions does this algorithm converge?*

The second path was through industry. I spent years building personalized search and recommendation systems at one of Europe's largest e-commerce platforms. There, the questions were different but equally demanding: *how do we rank millions of products in real-time? how do we balance short-term conversions with long-term customer satisfaction? how do we deploy safely when every change affects revenue?* The gap between textbook algorithms and production systems was... substantial.

When I started reading RL literature seriously, I found myself caught between two worlds. Sutton and Barto would present a policy gradient, and I'd reach for my pen: *what function space is this policy in? when can we exchange the gradient and expectation?* The intuition was beautiful, but I needed to see the mathematical scaffolding. Meanwhile, the rigorous analysis texts I grew up with would prove exquisite convergence theorems for abstract operators—but rarely connected them to anything I could implement, let alone deploy.

Neither tradition was wrong. They were solving different problems, speaking to different audiences. But I kept wishing for a book that held both standards simultaneously: mathematical precision *and* production-quality implementations, rigorous proofs *and* honest assessments of when theory's assumptions break in practice.

This book is my attempt to write what I wished I'd had—a text that treats mathematics and implementation as inseparable parts of the same understanding.

---

## What I'm trying to do

The books that shaped me most were those that managed to be both rigorous and alive. Brezis's *Functional Analysis, Sobolev Spaces and Partial Differential Equations* showed that you could prove theorems with full precision while explaining *why* the hypotheses mattered and what would break without them. Lions and Magenes demonstrated that abstract operator theory could illuminate concrete physical problems. These books didn't simplify—they clarified.

I wanted to write an RL book in that spirit, though I'm not sure I've fully succeeded. It's an attempt to hold both standards at once:

**Mathematics with full precision.** When I claim a value function exists and is unique, I try to prove it—with explicit assumptions, named theorems, and complete arguments. When I say an algorithm converges, I specify in what norm, at what rate, under what conditions. You shouldn't have to guess whether something is a theorem or a heuristic.

**Code you could actually run.** Not pseudocode that gestures at an implementation, but working PyTorch and JAX that demonstrates the concepts. Every theorem gets a numerical verification. Every algorithm gets tested against edge cases. The code might not be production-ready in every detail, but it's honest about what actually works.

**Honesty about the gaps.** Theory makes assumptions that practice violates. Rewards aren't always bounded; state spaces aren't always finite; we rarely know the transition kernel exactly. I try not to pretend these gaps don't exist. Instead, I name them, explore when they matter, and point toward research that's working to close them. Sometimes the most valuable lesson is understanding *why* something works despite violating its theoretical assumptions.

---

## How the book is organized

The book is organized by chapters: `ch00/` through `ch10/` and beyond. Each chapter folder contains the main content, exercises, and lab solutions.

The chapters follow a progression:

- **Part I (Ch 1–3):** Foundations — measure theory, probability, operators, the Bellman equation as a contraction
- **Part II (Ch 4–5):** The simulator — how to generate realistic catalogs, users, queries, and model position bias
- **Part III (Ch 6–8):** Policies — from discrete bandits to continuous actions to constrained optimization
- **Part IV (Ch 9–11):** Evaluation and deployment — off-policy estimation, robustness, multi-session dynamics

`outline.md` maps chapters to code modules. `syllabus.md` provides a detailed internal course-style reading order; `syllabus_github.md` is a GitHub-facing syllabus for self-study and teaching.

---

## A note on the style

I use equations with anchors like `{#EQ-1.2}` and reference them in prose. I include runnable code blocks with explicit outputs so you can verify your environment matches mine. When bridging theory to implementation, I point to concrete modules rather than waving vaguely at "the code."

The precision here is practical, not pedantic. I've spent too many hours staring at discrepancies between my results and a textbook, unsure whether the problem was a bug in my code, a typo in the text, or a gap in my understanding. I want you to be able to check—to trace any claim back to its source and verify it yourself.

---

## Who this book is for

This book assumes you're comfortable with measure theory—or at least willing to work through it—and that you want to understand RL with mathematical precision while also seeing working implementations.

If you prefer intuition-first approaches and want to start shipping quickly, this might feel heavier than you need. Sutton and Barto's book is excellent for building intuition, and you can always return to the proofs when curiosity strikes.

If you want pure mathematics without implementation details, dedicated functional analysis texts like Rudin or Conway will serve you better.

But if you're the kind of person who reads a policy gradient theorem and finds yourself wondering about the dominated convergence conditions that justify swapping the derivative and expectation—*and* you want to see that gradient computed on actual hardware—then I hope this book will resonate.

I wrote it for the version of myself who was searching library shelves for exactly this: a text that takes both the mathematics and the engineering seriously, and doesn't treat them as separate concerns.

Welcome. Let's begin.
