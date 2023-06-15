# Stochastic Volatility with particle filtering

## Project Description
Nous vous proposons d'étudier quelques méthodes SMC (principalement développés par Arnaud Doucet et Geir Storvik) : SIR, filre de Storvik and PLS.

**Table of contents**
1. [Technologies](#technologies)
2. [Examples](#examples)
3. [Sources](#sources)


## 1. Technologies 

Simple but not very effective : Python, we would should have used C++, R or Julia.

## 2. Sereval examples 

We have discussed about three benchmark models : a linear model, the Kitagawa's model and the stochstic volatility model (SV). Here, we will only present the SV model. Let $T \in \mathbb{R}$, for instance $T=100$, and $(X_t)_{t \in [0:T]}$ the hidden process, and $Y_{1:T}=\left(Y_1,...,Y_T\right)$ the observations, we write the Hidden Markov model as :

$$x_t = \alpha x_{t-1} + \beta \frac{x_{t-1}}{1+x^2_{t-1}} + \gamma \cos(1.2(t-1)) + \omega_t$$
Avec, $\omega_t \sim \mathcal{N}(0,W)$

$$y_t = \frac{x^2_t}{20} + \nu_t$$
Avec, $\nu_t \sim \mathcal{N}(0,V)$

### 2.1. First approach, assuming the parameters are known

We apply a simple SIR with $\alpha$=0.5, $\beta$=25, $\gamma$=8, V=5 et $W=1$ supposed known.
![stocha-param_connus](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/c61e594c-379f-4cf4-9906-5412fec14a56)

### 2.2. Then, assuming the parameters are unknown

However, in real life, model parameters are usually unkonwn. That is why, whe should first, estimate them. 
![SV_storvik](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/172931f6-f8e7-438b-a0da-80f0916e6774)

![estimation_par_SV](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/feb1269e-76c9-4c9c-bf2f-efcce9a7175f)

### 2.3. Comparaison between PLS, SIR and Storvik's filter

![comparaison-PLS-SIR-STORVIK](https://github.com/SarcasticMatrix/Stochastic-Volatility-with-particle-filtering/assets/94806199/6e573136-db0a-439f-9645-98915e31b394)

### 3. Sources
