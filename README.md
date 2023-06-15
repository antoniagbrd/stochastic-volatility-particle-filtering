# Stochastic Volatility with particle filtering

### Project Description
Nous vous proposons d'étudier quelques méthodes SMC (principalement développés par Arnaud Doucet et Geir Storvik) :

**Table of contents**
1. [Technologies](#technologies)
2. [Eexamples](#examples)
3. [Sources](#sources)


### Technologies 

$a = b^2$
### Sereval examples 

We have discussed about three benchmark models : a linear model, the Kitagawa's model and the stochstic volatility model (SV). There is a 

$$ x_{t} = \alpha x_{t-1} + \beta \frac{x_{t-1}}{1+x^{2}_{t-1}} + \gamma \cos(1.2(t-1)) + \omega_{t}, \mbox{ avec } \omega_{t} \sim \mathcal{N}(0,W) $$
$$ y_{t} = \frac{x^{2}_{t}}{20} + \nu_{t}, \mbox{ avec } \nu_{t} \sim \mathcal{N}(0,V).$$

### Sources
