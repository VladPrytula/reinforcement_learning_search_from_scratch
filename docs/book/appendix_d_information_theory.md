# Appendix D --- Information-Theoretic Lower Bounds

**Vlad Prytula**

---

## Motivation

Chapter 1 previewed a fundamental limit on learning: any bandit algorithm must incur expected regret $\Omega(\sqrt{KT})$ over $T$ rounds with $K$ arms. Chapter 6 states this bound formally as [THM-6.0] and shows that Thompson Sampling and LinUCB achieve matching upper bounds (up to logarithmic factors). But why is this lower bound true? What prevents a clever algorithm from learning faster?

The answer lies in **information theory**. Learning requires distinguishing between bandit instances, and information theory quantifies the fundamental cost of discrimination. This appendix develops the key tools---KL divergence, Fano's inequality, and hypothesis testing---and proves the minimax lower bound for stochastic bandits.

**Prerequisites.** This appendix assumes familiarity with probability theory at the level of Chapter 2 (expectations, probability measures) and basic linear algebra. No prior information theory background is required.

**Why this matters for RL.** Lower bounds tell us when to stop searching for better algorithms. If Thompson Sampling achieves $O(\sqrt{KT \log K})$ regret and the lower bound is $\Omega(\sqrt{KT})$, then TS is optimal up to logarithmic factors---no algorithm can do fundamentally better. This is why the bandit community considers these algorithms "solved" (for the stochastic setting) and focuses research on harder problems: adversarial bandits, contextual bandits with rich function classes, and continuous action spaces.

---

## D.1 KL Divergence and Mutual Information

### D.1.1 Kullback-Leibler Divergence

**Definition D.1.1** (KL Divergence) {#DEF-D.1.1}

Let $P$ and $Q$ be probability distributions on a measurable space $(\Omega, \mathcal{F})$ with $P$ absolutely continuous with respect to $Q$ (written $P \ll Q$). The **Kullback-Leibler divergence** (or relative entropy) from $Q$ to $P$ is:

$$
\mathrm{KL}(P \| Q) = \mathbb{E}_P\left[\log \frac{dP}{dQ}\right] = \int \log\left(\frac{dP}{dQ}\right) dP
\tag{D.1}
$$
{#EQ-D.1}

where $\frac{dP}{dQ}$ is the Radon--Nikodym derivative (see Chapter 2, Definition 2.6.1).

For discrete distributions with $P = (p_1, \ldots, p_n)$ and $Q = (q_1, \ldots, q_n)$:

$$
\mathrm{KL}(P \| Q) = \sum_{i=1}^{n} p_i \log \frac{p_i}{q_i}
\tag{D.2}
$$
{#EQ-D.2}

with the convention $0 \log 0 = 0$ and $p \log(p/0) = +\infty$ if $p > 0$.

**Remark D.1.1** (Interpretation). KL divergence measures the "statistical distinguishability" between distributions. If $\mathrm{KL}(P \| Q)$ is small, observations drawn from $P$ look similar to observations from $Q$, making it hard to determine which distribution generated the data. This is precisely the challenge in bandit learning: we must distinguish between arms with similar (but not identical) reward distributions.

**Proposition D.1.1** (Properties of KL Divergence) {#PROP-D.1.1}

KL divergence satisfies:

1. **Non-negativity**: $\mathrm{KL}(P \| Q) \geq 0$, with equality if and only if $P = Q$ a.e.
2. **Not symmetric**: In general, $\mathrm{KL}(P \| Q) \neq \mathrm{KL}(Q \| P)$
3. **Chain rule**: For joint distributions, $\mathrm{KL}(P_{XY} \| Q_{XY}) = \mathrm{KL}(P_X \| Q_X) + \mathbb{E}_{P_X}[\mathrm{KL}(P_{Y|X} \| Q_{Y|X})]$
4. **Additivity for products**: If $P = P_1 \otimes \cdots \otimes P_n$ and $Q = Q_1 \otimes \cdots \otimes Q_n$ are product measures, then $\mathrm{KL}(P \| Q) = \sum_{i=1}^n \mathrm{KL}(P_i \| Q_i)$

*Proof.* Property 1 follows from Jensen's inequality applied to the convex function $-\log$. Properties 3--4 follow from the chain rule for Radon--Nikodym derivatives. See [@cover:information_theory:2006, Chapter 2] for details. $\square$

### D.1.2 Mutual Information

**Definition D.1.2** (Mutual Information) {#DEF-D.1.2}

For random variables $X$ and $Y$ with joint distribution $P_{XY}$ and marginals $P_X$, $P_Y$, the **mutual information** is:

$$
I(X; Y) = \mathrm{KL}(P_{XY} \| P_X \otimes P_Y) = \mathbb{E}\left[\log \frac{P_{XY}(X, Y)}{P_X(X) P_Y(Y)}\right]
\tag{D.3}
$$
{#EQ-D.3}

Mutual information measures the reduction in uncertainty about $X$ from observing $Y$ (and vice versa, since $I(X; Y) = I(Y; X)$).

### D.1.3 Data Processing Inequality

The data processing inequality is the key tool connecting information measures to statistical estimation.

**Theorem D.1.1** (Data Processing Inequality) {#THM-D.1.1}

If $X \to Y \to Z$ form a Markov chain (i.e., $Z$ is conditionally independent of $X$ given $Y$), then:

$$
I(X; Z) \leq I(X; Y)
\tag{D.4}
$$
{#EQ-D.4}

*Proof.* By the chain rule for mutual information:

$$
I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y) = I(X; Z) + I(X; Y \mid Z)
$$

Since $X \to Y \to Z$ is Markov, $I(X; Z \mid Y) = 0$. Thus $I(X; Y) = I(X; Z) + I(X; Y \mid Z) \geq I(X; Z)$, where the inequality uses non-negativity of mutual information. $\square$

**Corollary D.1.1** (Processing cannot create information). {#COR-D.1.1}

No deterministic or stochastic function of data can increase information about a parameter. If $\theta$ is a parameter, $X$ is data, and $\hat{\theta} = f(X)$ is an estimator, then $I(\theta; \hat{\theta}) \leq I(\theta; X)$.

This is the information-theoretic foundation for statistical lower bounds: the information in the data places a hard limit on estimation accuracy, regardless of how clever the estimator.

---

## D.2 Fano's Inequality

Fano's inequality translates information-theoretic quantities into bounds on error probability for hypothesis testing.

### D.2.1 Classical Fano's Inequality

**Theorem D.2.1** (Fano's Inequality) {#THM-D.2.1}

Let $\Theta = \{1, 2, \ldots, M\}$ be a finite set of hypotheses with prior $\pi$ (e.g., uniform). Let $X$ be an observation with distribution $P_\theta$ under hypothesis $\theta$. For any estimator $\hat{\theta}(X)$, the probability of error satisfies:

$$
\mathbb{P}(\hat{\theta}(X) \neq \theta) \geq 1 - \frac{I(\theta; X) + \log 2}{\log M}
\tag{D.5}
$$
{#EQ-D.5}

where $I(\theta; X)$ is the mutual information between the hypothesis and observation.

*Proof sketch.* Define the error indicator $E = \mathbf{1}\{\hat{\theta} \neq \theta\}$. By the data processing inequality applied to the chain $\theta \to X \to \hat{\theta}$:

$$
H(\theta \mid \hat{\theta}) \leq H(\theta \mid X) + H(E) \leq H(\theta) - I(\theta; X) + 1
$$

where $H(\cdot)$ denotes entropy and we used $H(E) \leq \log 2 \leq 1$ for a binary variable. Now apply Fano's lemma: $H(\theta \mid \hat{\theta}) \geq \mathbb{P}(E = 1) \cdot \log(M-1)$. Combining and rearranging yields the result. See [@cover:information_theory:2006, Theorem 2.10.1] for the complete proof. $\square$

**Remark D.2.1** (Interpretation). Fano's inequality says: if the mutual information $I(\theta; X)$ between the parameter and observations is small compared to $\log M$ (the entropy of a uniform prior over $M$ hypotheses), then no estimator can reliably identify the true hypothesis. This is the quantitative version of "you cannot learn what you cannot distinguish."

### D.2.2 Fano for Bandit Lower Bounds

For bandit lower bounds, we use a slightly different form that bounds the number of times a suboptimal arm must be pulled.

**Lemma D.2.1** (Change of Measure for Bandits) {#LEM-D.2.1}

Consider two bandit instances $\nu$ and $\nu'$ that differ only in the reward distribution of arm $a$. Let $\mathbb{E}_\nu$ and $\mathbb{E}_{\nu'}$ denote expectations under the two instances. For any event $\mathcal{E}$ and any algorithm:

$$
\mathrm{KL}(\mathbb{P}_\nu(\mathcal{E}) \| \mathbb{P}_{\nu'}(\mathcal{E})) \leq \mathbb{E}_\nu[N_a(T)] \cdot \mathrm{KL}(\nu_a \| \nu'_a)
\tag{D.6}
$$
{#EQ-D.6}

where $N_a(T)$ is the number of times arm $a$ is pulled in $T$ rounds, and $\nu_a$, $\nu'_a$ are the reward distributions for arm $a$ under the two instances.

*Proof.* This follows from the chain rule for KL divergence and the fact that rewards from arm $a$ contribute $\mathrm{KL}(\nu_a \| \nu'_a)$ to the total divergence each time the arm is pulled. See [@lattimore:bandits:2020, Lemma 15.1] for the complete argument. $\square$

---

## D.3 Bandit Lower Bound via Hypothesis Testing

We now prove the minimax lower bound for stochastic bandits. The strategy is to construct a family of "hard" bandit instances such that any algorithm must incur high regret on at least one of them.

### D.3.1 Problem Setup

Consider a $K$-armed stochastic bandit with reward distributions $\nu = (\nu_1, \ldots, \nu_K)$. Each arm $a$ has mean reward $\mu_a = \mathbb{E}_{r \sim \nu_a}[r]$. The optimal arm is $a^* = \arg\max_a \mu_a$ with gap $\Delta_a = \mu_{a^*} - \mu_a$ for suboptimal arms.

Expected regret over $T$ rounds is:

$$
\mathbb{E}[\mathrm{Regret}(T)] = \sum_{a \neq a^*} \Delta_a \cdot \mathbb{E}[N_a(T)]
\tag{D.7}
$$
{#EQ-D.7}

where $N_a(T)$ is the number of pulls of arm $a$.

### D.3.2 Construction of Hard Instances

**Theorem D.3.1** (Minimax Lower Bound for Stochastic Bandits) {#THM-D.3.1}

For any learning algorithm and any time horizon $T \geq K$, there exists a $K$-armed bandit instance with Bernoulli rewards such that:

$$
\mathbb{E}[\mathrm{Regret}(T)] \geq \frac{1}{20} \sqrt{KT}
\tag{D.8}
$$
{#EQ-D.8}

*Proof.* We construct $K$ bandit instances $\nu^{(1)}, \ldots, \nu^{(K)}$ and show that any algorithm has high regret on at least one.

**Step 1: Instance construction.**

Let $\Delta = \sqrt{K/(4T)}$ (to be optimized). Define:
- **Base instance** $\nu^{(0)}$: All arms have Bernoulli$(1/2)$ rewards
- **Instance** $\nu^{(i)}$ for $i = 1, \ldots, K$: Arm $i$ has Bernoulli$(1/2 + \Delta)$ rewards; all other arms have Bernoulli$(1/2)$ rewards

Under $\nu^{(i)}$, arm $i$ is optimal with gap $\Delta$ for all other arms.

**Step 2: Information bound.**

By Lemma D.2.1, the KL divergence between observations under $\nu^{(0)}$ and $\nu^{(i)}$ is:

$$
\mathrm{KL}(\mathbb{P}_{\nu^{(0)}} \| \mathbb{P}_{\nu^{(i)}}) \leq \mathbb{E}_{\nu^{(0)}}[N_i(T)] \cdot \mathrm{KL}(\mathrm{Ber}(1/2) \| \mathrm{Ber}(1/2 + \Delta))
$$

For small $\Delta$, $\mathrm{KL}(\mathrm{Ber}(1/2) \| \mathrm{Ber}(1/2 + \Delta)) \leq 4\Delta^2$ (using Taylor expansion of binary KL).

**Step 3: Lower bound on pulls.**

Summing over arms:

$$
\sum_{i=1}^K \mathbb{E}_{\nu^{(0)}}[N_i(T)] \cdot 4\Delta^2 \geq \sum_{i=1}^K \mathrm{KL}(\mathbb{P}_{\nu^{(0)}} \| \mathbb{P}_{\nu^{(i)}})
$$

Since $\sum_i N_i(T) = T$, there exists some arm $j$ with $\mathbb{E}_{\nu^{(0)}}[N_j(T)] \leq T/K$.

**Step 4: Regret bound.**

Under instance $\nu^{(j)}$, the regret is:

$$
\mathbb{E}_{\nu^{(j)}}[\mathrm{Regret}(T)] = \sum_{a \neq j} \Delta \cdot \mathbb{E}_{\nu^{(j)}}[N_a(T)] = \Delta \cdot (T - \mathbb{E}_{\nu^{(j)}}[N_j(T)])
$$

Using a change-of-measure argument (Pinsker's inequality), we can show that $\mathbb{E}_{\nu^{(j)}}[N_j(T)]$ cannot be much larger than $\mathbb{E}_{\nu^{(0)}}[N_j(T)] \leq T/K$ when $\Delta$ is small.

Substituting $\Delta = \sqrt{K/(4T)}$:

$$
\mathbb{E}_{\nu^{(j)}}[\mathrm{Regret}(T)] \geq \Delta \cdot \frac{T}{2} = \frac{1}{2}\sqrt{\frac{K}{4T}} \cdot T = \frac{1}{4}\sqrt{KT}
$$

Accounting for constants in the change-of-measure step yields the stated bound $\frac{1}{20}\sqrt{KT}$. $\square$

**Remark D.3.1** (Tightness). The constant $1/20$ is not optimal; tighter analyses achieve $\frac{1}{2}\sqrt{KT}$ or better. The key message is the **rate** $\Omega(\sqrt{KT})$, which is matched by UCB and Thompson Sampling up to logarithmic factors.

---

## D.4 Extensions and Connections

### D.4.1 Instance-Dependent Lower Bounds

Theorem D.3.1 gives a **minimax** (worst-case) lower bound. For specific instances with known gaps $\Delta_a$, tighter **instance-dependent** bounds exist.

**Theorem D.4.1** (Lai-Robbins Lower Bound, informal) {#THM-D.4.1}

For any consistent algorithm (one with $o(T^p)$ regret for all $p > 0$ on all instances), and any bandit instance with gaps $\Delta_a > 0$:

$$
\liminf_{T \to \infty} \frac{\mathbb{E}[\mathrm{Regret}(T)]}{\log T} \geq \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\mathrm{KL}(\nu_a \| \nu_{a^*})}
\tag{D.9}
$$
{#EQ-D.9}

This shows that regret must grow at least logarithmically, with a coefficient determined by the KL divergence between suboptimal and optimal arm distributions.

### D.4.2 Contextual Bandits

For contextual bandits with feature dimension $d$, the lower bound becomes:

$$
\mathbb{E}[\mathrm{Regret}(T)] = \Omega(d\sqrt{T})
\tag{D.10}
$$
{#EQ-D.10}

The dimension $d$ replaces the number of arms $K$ because, with linear structure, the effective number of "directions" to explore is $d$. This is why LinUCB and Linear Thompson Sampling achieve $O(d\sqrt{T \log T})$ regret---they match the lower bound up to logarithmic factors.

### D.4.3 Continuous Actions and Eluder Dimension

For continuous action spaces $\mathcal{A} \subseteq \mathbb{R}^d$, the situation is more subtle. The relevant complexity measure is the **eluder dimension** of the function class, which captures how many "independent directions" of exploration are needed.

**Definition D.4.1** (Eluder Dimension, informal) {#DEF-D.4.1}

The $\varepsilon$-eluder dimension of a function class $\mathcal{F}$ on action space $\mathcal{A}$ measures the length of the longest sequence of actions such that each action is "$\varepsilon$-independent" of the previous actions with respect to functions in $\mathcal{F}$.

For linear functions on $\mathbb{R}^d$, the eluder dimension is $O(d)$, recovering the contextual bandit bound. For more complex function classes (e.g., neural networks), the eluder dimension can be much larger or even infinite, explaining why deep RL requires more samples.

### D.4.4 Connection to Minimax Theory

The bandit lower bound is an instance of **minimax theory** from statistics. The general pattern is:

1. Construct a family of "hard" instances (hypotheses)
2. Show that distinguishing between them requires many samples (via Fano/Le Cam)
3. Conclude that any estimator/algorithm must have high error/regret on some instance

This methodology extends to supervised learning (minimax rates for regression), density estimation, and other statistical problems. The information-theoretic tools developed here---KL divergence, Fano's inequality, data processing---are universal.

---

## Summary

This appendix established the information-theoretic foundation for bandit lower bounds:

1. **KL divergence** [DEF-D.1.1] measures statistical distinguishability between distributions
2. **Data processing inequality** [THM-D.1.1] shows that processing cannot create information
3. **Fano's inequality** [THM-D.2.1] converts information bounds to error probability bounds
4. **Minimax lower bound** [THM-D.3.1] proves $\Omega(\sqrt{KT})$ regret is unavoidable

The key insight: learning is fundamentally limited by the information content of observations. When bandit instances are statistically similar (low KL divergence), no algorithm can distinguish them quickly, and regret is unavoidable. Thompson Sampling and UCB are optimal because they match these information-theoretic limits.

---

## References

The treatment in this appendix follows:

- [@cover:information_theory:2006] --- Standard reference for information theory fundamentals
- [@lattimore:bandits:2020, Chapters 15--16] --- Modern treatment of bandit lower bounds
- [@tsybakov:nonparametric:2009] --- Minimax theory for statistical estimation

The original Lai-Robbins lower bound appears in [@lai:asymptotically:1985]. Fano's inequality is due to [@fano:transmission:1961].
