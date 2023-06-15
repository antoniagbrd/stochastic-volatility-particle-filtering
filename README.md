# Stochastic Volatility with particle filtering

### Project Description
Nous vous proposons d'étudier quelques méthodes SMC (principalement développés par Arnaud Doucet et Geir Storvik) :

**Table of contents**
1. [Technologies](#technologies)
2. [Eexamples](#examples)
3. [Sources](#sources)


### Technologies 

### Sereval examples 

We have discussed about three benchmark models : a linear model, the Kitagawa's model and the stochstic volatility model (SV). Here, we will only present the SV model. Let $T \in \mathbb{R}$, for instance $T=100$, and $(X_t)_{t \in [0:T]}$ the hidden process, and $Y_{1:T}=(Y_1,\dots,Y_T)$ the observations, we write the Hidden Markov model as :

$$x_t = \alpha x_{t-1} + \beta \frac{x_{t-1}}{1+x^2_{t-1}} + \gamma \cos(1.2(t-1)) + \omega_t$$
Avec, $\omega_t \sim \mathcal{N}(0,W)$

$$y_t = \frac{x^2_t}{20} + \nu_t$$
Avec, $\nu_t \sim \mathcal{N}(0,V)$


### Sources
