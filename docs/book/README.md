# Introduction

I wrote this book out of love to a pure beauty of math, the applicability of code, their inseparability and out of frustration.

Not frustration with reinforcement learning itself—RL is beautiful, and the problems it tackles are genuinely important. My frustration was with how we teach it, and how I learned it.

I trained as an applied mathematician. Yosida and Brezis were my table books—the functional analysis tradition where every claim comes with hypotheses, every theorem earns its proof, and you always know what space you're working in. When you prove a fixed point exists, you name the contraction constant. When you differentiate under an integral, you cite dominated convergence. That's how I learned to think.

Then I started reading reinforcement learning.

Sutton and Barto would present a gradient update, and I'd stop: *Is this really a minimum? In what sense? What if the function isn't differentiable there?* Policy gradient theorems appeared, but the function spaces stayed implicit. I found myself scribbling in margins, trying to reconstruct the measure theory behind the expectations, the operator theory behind the Bellman equations. The books weren't wrong—they just weren't written for someone who needed to see the scaffolding.

And the rigorous texts I grew up on had the opposite problem. I knew contraction mappings. I could prove Banach fixed-point theorems in my sleep. But those books never said: *here's how you use this to make a robot walk, or rank search results, or allocate ad budgets.* The connection to anything computable, anything deployable—that was left as an exercise. Usually an impossible one.

This book is my attempt to bridge that gap.

---

## What I'm trying to do

The books that left the deepest impression on me during my university years were those that managed to be both rigorous and alive. Brezis's *Functional Analysis, Sobolev Spaces and Partial Differential Equations* taught me that you could prove theorems with full mathematical precision while still explaining *why* the hypotheses mattered, what would break without them, and where the result fit into a larger story. Lions and Magenes showed me that even the most abstract operator theory could connect to physical problems you cared about.

I wanted to write an RL book in that spirit. Not a watered-down version of either tradition, but an honest attempt to hold both standards simultaneously:

**The mathematics should be real mathematics.** When I claim a value function exists and is unique, I prove it—with explicit assumptions, named theorems, and complete arguments. When I say an algorithm converges, I specify: in what norm, at what rate, under what conditions. The reader should never have to wonder whether a claim is a theorem or a heuristic.

**The code should be real code.** Not pseudocode that waves at an implementation, but production-quality PyTorch and JAX that you could deploy tomorrow. Every mathematical concept gets a numerical verification. Every theorem gets tested against its computational counterpart.

**The gap between them should be acknowledged honestly.** Theory assumes things that practice violates. Rewards aren't bounded; state spaces aren't finite; we don't actually know the transition kernel. A good textbook doesn't pretend these gaps don't exist. It names them, explains when they matter, and points to the research trying to close them.

---

## How the book is organized

The book is organized by chapters: `ch00/` through `ch10/` and beyond. Each chapter folder contains the main content, exercises, lab solutions, and an `archive/` of earlier drafts. The structure is simple because the content is what matters.

The chapters follow a progression:

- **Part I (Ch 1–3):** Foundations — measure theory, probability, operators, the Bellman equation as a contraction
- **Part II (Ch 4–5):** The simulator — how to generate realistic catalogs, users, queries, and model position bias
- **Part III (Ch 6–8):** Policies — from discrete bandits to continuous actions to constrained optimization
- **Part IV (Ch 9–11):** Evaluation and deployment — off-policy estimation, robustness, multi-session dynamics

`outline.md` maps chapters to code modules. `syllabus.md` provides a course-style reading order if you want structure.

---

## A note on the style

I use equations with anchors like `{#EQ-1.2}` and reference them in prose. I include runnable code blocks with explicit outputs so you can verify your environment matches mine. When bridging theory to implementation, I point to concrete modules rather than waving vaguely at "the code."

This isn't showing off. It's defensive. I've read too many books where I couldn't figure out whether a discrepancy between my results and the text was a bug in my code, a typo in the book, or a fundamental misunderstanding on my part. I want you to be able to check.

---

## Who this book is for

If you're comfortable with measure theory and want to understand RL with full mathematical precision—while also building systems that work—this book is for you.

If you prefer intuition over formalism and just want to ship, you'll probably find this heavier than necessary.

If you want pure mathematics without the messy application details, there are better functional analysis texts.

But if you're the kind of person who reads a policy gradient theorem and immediately wonders about the dominated convergence conditions that justify interchanging the derivative and integral—and also wants to see that gradient computed on a GPU—then I wrote this for you and me :) .

Welcome. Let's begin.
