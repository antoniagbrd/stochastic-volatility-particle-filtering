# Stochastic Volatility with particle filtering

## Project Description
Nous vous proposons d'étudier quelques méthodes SMC (principalement développés par Arnaud Doucet et Geir Storvik) :

**Table of contents**
1. [Technologies](#technologies)
2. [Examples](#examples)
3. [Sources](#sources)


## 1. Technologies 

## 2. Sereval examples 

We have discussed about three benchmark models : a linear model, the Kitagawa's model and the stochstic volatility model (SV). Here, we will only present the SV model. Let $T \in \mathbb{R}$, for instance $T=100$, and $(X_t)_{t \in [0:T]}$ the hidden process, and $Y_{1:T}=\left(Y_1,...,Y_T\right)$ the observations, we write the Hidden Markov model as :

$$x_t = \alpha x_{t-1} + \beta \frac{x_{t-1}}{1+x^2_{t-1}} + \gamma \cos(1.2(t-1)) + \omega_t$$
Avec, $\omega_t \sim \mathcal{N}(0,W)$

$$y_t = \frac{x^2_t}{20} + \nu_t$$
Avec, $\nu_t \sim \mathcal{N}(0,V)$

### First approach, assuming the parameters are known

### Then, assuming the parameters are unknown

![estimation_par_SV](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/feb1269e-76c9-4c9c-bf2f-efcce9a7175f)

### Comparaison between PLS, SIR and Storvik's filter

![comparaison-PLS-SIR-STORVIK](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/6e573136-db0a-439f-9645-98915e31b394)

### 3. Sources
