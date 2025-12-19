# Appendix 7.A: The "Fair Fight" Principle in RL Benchmarking

**By Vlad Prytula**

In scientific computing, we often claim to compare **Algorithm A** (e.g., Discrete LinUCB) against **Algorithm B** (e.g., Continuous Q-Learner). But usually, we are actually comparing **Algorithm A + Configuration X** against **Algorithm B + Configuration Y**.

During the development of Chapter 7, we encountered a textbook example of this trap. Early runs of our benchmark showed the "sophisticated" Continuous Q-Learner underperforming the "simple" Discrete LinUCB by nearly 50%.

The culprit wasn't the neural network, the CEM optimizer, or the exploration strategy. It was a single scalar: `a_max`.

## The "Hidden Handcuffs" Incident

### 1. The Symptom
Our discrete templates (Chapter 6) were defined with aggressive boost logic:
```python
# zoosim/policies/templates.py
lambda p: a_max if p.cm2 > 0.4 else 0.0
```
where `a_max` defaulted to **5.0** in the template factory. This allowed the discrete agent to apply massive score adjustments (+5.0) to sweep huge swathes of products to the top of the ranking.

Our continuous agent, however, initially inherited a conservative default from `zoosim/core/config.py`:
```python
@dataclass
class ActionConfig:
    a_max: float = 0.5  # Conservative default (early experiments)
```

### 2. The Physics of Failure
The Continuous agent was trying to optimize a ranking with **1/10th the authority** of the Discrete agent.
- **Discrete Agent**: Could assert "This is a high-margin item, boost it by +5.0!" (Overriding base relevance).
- **Continuous Agent**: Could only whisper "Boost this by +0.5." (Drowned out by base relevance noise).

Mathematically, the continuous agent was optimizing over a hypercube $\mathcal{A}_{\text{cont}} = [-0.5, 0.5]^{10}$, while the discrete agent selected from corners of a much larger hypercube $\mathcal{A}_{\text{disc}} \subset [-5.0, 5.0]^{10}$.

### 3. Design Space and Standardization

**What is the "right" action magnitude?** This is a hyperparameter trade-off, not a universal constant. The choice depends on the signal-to-noise ratio of the learning problem.

**Base relevance scale in our simulator**: Lexical and embedding relevance scores typically range from 10 to 30 for relevant products (see §5.2). Position bias and click propensities add multiplicative effects.

**Action magnitude relative to base signal:**
- `a_max = 0.5`: Boosts are **subtle interventions** (±2-5% of base relevance). Agents must learn fine-grained adjustments, but learning signals are weak.
- `a_max = 5.0`: Boosts are **visible interventions** (±15-50% of base relevance). Agents can override poor base rankings, making learning dynamics observable in experiments.

**Our standardized choice for Chapters 6-8:**
```python
# zoosim/core/config.py (current)
@dataclass
class ActionConfig:
    a_max: float = 5.0  # Default for RL training (Ch6-8); explore smaller values for conservative control
```

**Rationale:**
1. **Pedagogical visibility**: With `a_max=5.0`, we can observe agents discovering high-margin products, correcting position bias, and exploring-exploiting in readable experiment outputs. At `a_max=0.5`, these effects are statistically present but visually obscured by noise.
2. **Fair comparison**: All RL methods (discrete templates, continuous Q-learning, policy gradients) now use the same action space geometry, enabling ceteris paribus benchmarking.
3. **Exploration of conservative control**: Readers interested in gradual, low-risk interventions can experiment with smaller values (0.5, 1.0) to study the regime where agents must learn precise, subtle adjustments.

**The Result:**
- **Before Fix**: Continuous Agent **-49.8%** vs LinUCB.
- **After Fix**: Continuous Agent **+101.2%** vs LinUCB.

!!! note "Pedagogical Takeaway"
    **Ceteris Paribus (All Else Being Equal)** is the hardest part of RL research. When comparing a new method to a baseline, you must ensure they have access to the same **information** (context features) and the same **actuation authority** (action magnitude).
    
    If your complex RL agent is failing, don't just tune the learning rate. Check the physics. Are you fighting with one hand tied behind your back?

--- 
*See `scripts/ch07/compare_discrete_continuous.py` for the corrected implementation.*
