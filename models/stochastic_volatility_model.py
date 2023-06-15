import numpy as np
from random import gauss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
import seaborn as sns

from scipy.stats import norm, invgamma

from tqdm import tqdm


#######################################################################
# Trajectories fonctions
#######################################################################

" -> generates a random trajectory and its observations <- "
" -> return : hidden states, observations "
def generate_trajectory_stochastic_volatility_model(T) : 
    X_0 = 1
    alpha_cst= 0
    beta_cst = 0.9702
    mu_cst = 0
    W_cst = 1

    res = np.zeros((T,2))
    res[0,0] = X_0
    res[0,1] = mu_cst + np.exp(res[0,0]/2)*gauss(0,1)

    for k in range(1,T):
      
        res[k,0] = alpha_cst + beta_cst*res[k-1,0]+ gauss(0,W_cst)
        res[k,1] = mu_cst + np.exp(res[k,0]/2)*gauss(0,1)

    return res[:,0],res[:,1]

" -> plots a trajectory and its observations <-"
def show_a_trajectory_stochastic_volatility_model(T,x,y):
    L_t=[t for t in range(T)]

    plt.figure(figsize=(20,8))
    plt.plot(L_t,x,'-o',color='red',label='trajectoire')
    plt.plot(L_t,y,'-o',color='b',label='observations y')
    plt.xlabel('Temps')
    plt.title('Hidden states and observations generate from stochastic volatility model')
    plt.legend()
    plt.show()

" -> plots and generates a random trajectory and its observations  <-"
def show_random_trajectory_stochastic_volatility_model(T):

    x,y=generate_trajectory_stochastic_volatility_model(T)
    L_t=[t for t in range(T)]

    show_a_trajectory_stochastic_volatility_model(T,x,y)

#######################################################################
# Filters fonctions
#######################################################################

" -> runs SIR filter on a stochastic volatility model (parameters are known here) <-"
" -> return : weights, hidden states estimated "
def SIR_stochastic_volatility_model(T,Y,N,W_cst,alpha_cst,beta_cst,mu_cst):
    # N est le nombre de particules

    X_0 = np.random.randn(N)
    X = np.zeros((T,N))

    w_0 = norm.pdf(Y[0],0,np.exp(X_0/2))
    
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0
    
    # for t in range(1,T-1):
    for t in range(1,T):
        A = np.random.choice(range(N),N,p=W[t-1,:])
        
        X[t,:] = np.random.normal(alpha_cst+beta_cst*X[t-1][A],W_cst,N) 
        
        W[t,:] = norm.pdf(Y[t],mu_cst,np.exp(X[t,:]/2))
        W[t,:] = W[t,:]/(W[t,:].sum()) 
    return W,X

" -> runs Storvik's filter on a stochastic volatility model (parameters are unknownw here)"
" -> return : weights, hidden states, estimated a, estimated sigma "
def storvik_SIR_stochastic_volatility_model(T,Y,N):
    # N est le nombre de particules
    # T est le nombre d'itération

    # Statistique 
    n = np.zeros((T,N))
    d = np.zeros((T,N))
    m = np.zeros((T,N,2)) # m[t,j] = vecteur m de la jème particule à la tème itération 
    C = np.zeros((T,N,2,2)) # C[t,j] = matrice C de la jème particule à la tème itération 

    # Initialisation des paramètres
    #a[0,:] = 1
    #b[0,:] = 1
    m[0,:] = np.array([0,0.9])
    C[0,:] = np.diag([1, 1])
    n[0,:] = 2
    d[0,:] = 2
    
    W_estimated = np.zeros((T,N)) # W_cstametre[i,j] correspond à W au ième temps de la jème particule
    alpha_estimated = np.zeros((T,N)) # pareil
    beta_estimated = np.zeros((T,N)) # pareil
    mu_estimated = np.zeros((T,N)) # pareil

    #mu_estimated[0,:] = np.random.normal(a[0,:],b[0,:])
    
    for j in range(N):
        W_estimated[0,j] = invgamma.rvs(a=np.sqrt(n[0,j]/2), scale=np.sqrt(d[0,j]/2))
        alpha_estimated[0,j], beta_estimated[0,j] = np.random.multivariate_normal(mean=m[0,j] ,cov=np.sqrt(W_estimated[0,j])*np.linalg.inv(C[0,j])).tolist()

    # Initialisation de l'hiddent state et des poids
    X_0 = np.random.randn(N)
    X = np.zeros((T,N))
    X[0,:] = X_0
    
    w_0 = norm.pdf(Y[0],mu_estimated[0,:],np.sqrt(np.exp(X_0/2)))
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0  

    for t in range(1,T):

        A = np.random.choice(range(N),N,p=W[t-1,:])
        
        X[t,:] = np.random.normal(alpha_estimated[t-1,:]+beta_estimated[t-1,:]*X[t-1,:],np.sqrt(W_estimated[t-1,:])) 
        
        W[t,:] = norm.pdf(Y[t],mu_estimated[t-1,:],np.sqrt(np.exp(X[t,:]/2)))
        W[t,:] = W[t,:]/(W[t,:].sum())         

        for i in range(N):
               
            x_t_prec = X[t-1,i]
            x_t = X[t,i]
            y_t = Y[t]
            
            F_t = np.array([1,x_t_prec])
            
            C[t,i] = C[t-1,i] + np.outer(F_t,F_t.T)
            n[t,i] = n[t-1,i] + 1
            m[t,i] = np.linalg.inv(C[t,i]) @ (C[t-1,i] @ m[t-1,i] + F_t*x_t)
            d[t,i] = d[t-1,i] + (x_t - F_t@m[t-1,i])**2

            W_estimated[t,i] = invgamma.rvs(a=np.sqrt(n[t,i]/2), scale=np.sqrt(d[t,i]/2))
            alpha_estimated[t,i], beta_estimated[t,i] = np.random.multivariate_normal(mean=m[t,i] ,cov=np.sqrt(W_estimated[t,i])*np.linalg.inv(C[t,i])).tolist()

        # Resample x_t and s_t
        A = np.random.choice(range(N),N,p=W[t,:])
        X[t,:] = X[t,A]

        C[t,:] = C[t,A]
        n[t,:] = n[t,A]
        m[t,:] = m[t,A]
        d[t,:] = d[t,A]

    return X, W, np.sqrt(W_estimated), alpha_estimated, beta_estimated, mu_estimated

" -> estimation of parameters <- "
" -> return : the estimated parameters"
def parameters_estimations(n_W,n_W_estimated, n_alpha_estimated, n_beta_estimated, n_mu_estimated):

    nbr_iteration = np.shape(n_W)[0]
    
    estimated_alpha = (n_W[:,-1,:]*n_alpha_estimated[:,-1,:]).sum()/nbr_iteration
    estimated_beta = (n_W[:,-1,:]*n_beta_estimated[:,-1,:]).sum()/nbr_iteration
    mu_estimated = (n_W[:,-1,:]*n_mu_estimated[:,-1,:]).sum()/nbr_iteration
    estimated_W = (n_W[:,-1,:]*n_W_estimated[:,-1,:]).sum()/nbr_iteration

    return estimated_W, estimated_alpha, estimated_beta, mu_estimated

" -> smoother PLS "
" -> arguments : needs the estimated parameters, weights and hidden states estimations "
" -> return a hidden states estimations smoothed with PLS"
def backward_filter_stochastic_volatility_model(T,N,W,X,W_par,alpha_par,beta_par,mu_par) :
    X_smooth = np.zeros((T, N))

    # Select a pair (x_T(i),theta(i)) from step 1
    index = np.random.choice(range(N), p=W[T-1,:]) 
    X_smooth[T - 1,:] = X[T-1, index] 

    if np.shape(alpha_par) == ():
        alpha_smooth = alpha_par
        beta_smooth = beta_par
        mu_smooth = mu_par
        W_smooth = W_par
        
    else : 
        alpha_smooth = alpha_par[T-1,index]
        beta_smooth = beta_par[T-1,index]
        mu_smooth = mu_par[T-1,index]
        W_smooth = W_par[T-1,index]

    for t in tqdm(range(T - 2, -1, -1)): 
        #construct weight for the resampling
        w_t = norm.pdf(X_smooth[t+1,:], loc = beta_smooth*X[t,:], scale=W_smooth)
        w_t = w_t/(w_t.sum())

        #resampling particles
        index = np.random.choice(range(N),N, p=w_t)
        X_smooth[t,:] = X[t,index]

    return w_t,X_smooth


def FBS_stochastic_volatility_model(T,N,weights,X,W_par,alpha_par,beta_par):

    new_weights = np.zeros((T,N))
    new_weights[-1,:] = weights[-1,:]

    for t in tqdm(range(T-2,-1,-1)):
        
        for i in range(N):
            somme = 0
            for j in range(N):
           
                # for k in range(N):
                #    somme_k = weights[t,k] * norm.pdf(X[t+1,j], loc=beta_par*X[t,k], scale=W_par)

                somme += new_weights[t+1,j] * norm.pdf(X[t+1,j], loc=beta_par*X[t,i], scale=W_par)
                somme_k = np.sum(weights[t,:] * norm.pdf(X[t+1,j], loc=beta_par*X[t,:], scale=W_par))

                somme = somme/(somme_k)

            new_weights[t,i] += weights[t,i]*somme
            
        new_weights[t,:] = new_weights[t,:]/np.sum(new_weights[t,:]) 

    return new_weights 

#######################################################################
# Intermediate functions (not very usefull)
#######################################################################
" -> generates a random trajectory and runs Storvik's filter (and estimates unkonwn parameters) <- "
def run_storvik_SIR_stochastic_volatility_model(T,N):

    _,Y = generate_trajectory_stochastic_volatility_model(T)
    return storvik_SIR_stochastic_volatility_model(T,Y,N)

" -> run Storvik's filter on many random trajectories <- "
def run_n_storvik_SIR_stochastic_volatility_model(T,N,nbr_iteration):
    # on run nbr_iteration de fois storvik_SIR et on stocke theta et les poids
    
    n_W = np.zeros((nbr_iteration,T,N))

    n_W_parametre = np.zeros((nbr_iteration,T,N))
    n_alpha = np.zeros((nbr_iteration,T,N))
    n_beta = np.zeros((nbr_iteration,T,N))
    n_mu = np.zeros((nbr_iteration,T,N))
    
    for i in tqdm(range(nbr_iteration)):
        _, W, W_parametre, alpha, beta, mu = run_storvik_SIR_stochastic_volatility_model(T,N)
    
        n_W[i,:,:] = W
        n_W_parametre[i,:,:] = W_parametre
        n_alpha[i,:,:] = alpha
        n_beta[i,:,:] = beta
        n_mu[i,:,:] = mu
    
    return n_W,n_alpha,n_beta,n_mu,n_W_parametre

#######################################################################
# Show a run functions
#######################################################################

# SIR 
" -> show a run of SIR on a chosen trajectory <- "
def show_SIR_trajectory_stochastic_volatility_model(T,N,hidden_states,observations,alpha_cst,W_cst,beta_cst,mu_cst):
    L_t = np.arange(T)

    W,X = SIR_stochastic_volatility_model(T,observations,N,W_cst,alpha_cst,beta_cst,mu_cst)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hidden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("SIR filter with "+str(N)+' particles on stochastic volatility model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show() 

" -> show a run of SIR on a random trajectory <- "
def show_SIR_random_trajectory_stochastic_volatility_model(T,N,alpha_cst,W_cst,beta_cst,mu_cst):
    hidden_states,observations = generate_trajectory_stochastic_volatility_model(T)
    show_SIR_trajectory_stochastic_volatility_model(T,N,hidden_states,observations,alpha_cst,W_cst,beta_cst,mu_cst)

# Storvik's filter
" -> show a run of Storvik's filter on a chosen trajectory <- "
def show_storvik_SIR_stochastic_volatility_model(T,N,hiden_states,observations):
    L_t = np.arange(T)

    X, W, W_parametre, alpha, beta, mu = storvik_SIR_stochastic_volatility_model(T,observations,N)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    alpha_estimated = (W[-1,:]*alpha[-1,:]).sum()
    beta_estimated = (W[-1,:]*beta[-1,:]).sum()
    mu_estimated = (W[-1,:]*mu[-1,:]).sum()
    W_estimated = (W[-1,:]*W_parametre[-1,:]).sum()

    print('alpha =',round(alpha_estimated,3),'(=0.)')
    print('beta =',round(beta_estimated,3),'(=0.9702)')
    print('mu =',round(mu_estimated,3),'(=0)')
    print('W =', round(W_estimated,3),'(=1)')

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hiden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("Storvik's filter with "+str(N)+' particles stochastic volatility model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show()

" -> show a run of Storvik's filter on a random trajectory <- "
def show_random_storvik_SIR_stochastic_volatility_model(T,N):

    hiden_states,observations = generate_trajectory_stochastic_volatility_model(T)
    show_storvik_SIR_stochastic_volatility_model(T,N,hiden_states,observations)

#######################################################################
# Show estimated parameters functionns
####################################################################### 
" -> run Storvik's filter on a random trajectory and shows the estimated parameters <- "
def show_random_stovik_SIR_and_parameters_stochastic_volatility_model(T,N):

    x,y=generate_trajectory_stochastic_volatility_model(T)
    L_t = np.arange(T)

    X, W, W_parametre, alpha, beta, mu = storvik_SIR_stochastic_volatility_model(T,y,N)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    alpha_estimated = (W[-1,:]*alpha[-1,:]).sum()
    beta_estimated = (W[-1,:]*beta[-1,:]).sum()
    W_estimated = (W[-1,:]*W_parametre[-1,:]).sum()

    print('alpha =',round(alpha_estimated,3),'(=0)')
    print('beta =',round(beta_estimated,3),'(=0.9702)')
    print('W =', round(W_estimated,3),'(=1)')

    data_mean = [np.sum(alpha*W, axis=1),np.sum(beta*W, axis=1),np.sum(W_parametre*W, axis=1)]
    data_max = [np.max(alpha, axis=1),np.max(beta, axis=1),np.max(W_parametre, axis=1)]
    data_min = [np.min(alpha, axis=1),np.min(beta, axis=1),np.min(W_parametre, axis=1)]

    hist_data = [alpha[-1,:],beta[-1,:],W_parametre[-1,:]]

    true_data = [[0]*T,[0.9702]*T,[1]*T]

    labels = ['$\\alpha$','$\\beta$','$W$']

    top_limit = [2,2,3]
    bot_limit = [-2,0,0.5]

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(20,9))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])

    # Plot pour la ligne 1
    ax1 = fig.add_subplot(gs[0, :])
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='Storvik\'s filter' ,marker="o",ax=ax1)
    sns.lineplot(x=L_t,y=x,ax=ax1,label='hidden state',marker="o",color='black')
    ax1.set_xlabel('Temps')
    ax1.legend()

    # Plots pour la ligne 2
    for i in range(3):
        ax2 = fig.add_subplot(gs[1, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i],ax=ax2)
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=top_limit[i],bottom=bot_limit[i])

    # Plots pour la ligne 3
    # Last filter step 
    for i in range(3):
        ax3 = fig.add_subplot(gs[2, i])
        sns.histplot(hist_data[i], ax=ax3)
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    fig.suptitle("Storvik's SIR filter on stochastic volatility model with "+str(N)+' particles', fontsize=30,fontname='Times New Roman')  

    # Affichage de la figure
    plt.show()

" -> show Storvik's filter on many random trajectories and shows the estimated parameters <- "
def show_n_runs_stochastic_volatility_model(n_W,n_W_parametre,n_alpha,n_beta,n_mu):
    
    # nbr_iteration = np.shape(n_W)[0]
    # N = np.shape(n_W[0,0])[0]
    # T = np.shape(n_W[0])[0]
    nbr_iteration, N, T = np.shape(n_W)
    L_t = np.arange(T)
    
    # n_alpha[i,t,j] contient l'alpha de la ième itération au tème temps et de la jème particule
    
    alpha_mean = [np.sum(n_alpha[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    beta_mean = [np.sum(n_beta[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    W_parametre_mean = [np.sum(n_W_parametre[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]

    data_mean = [np.mean(alpha_mean,axis=0),np.mean(beta_mean,axis=0),np.mean(W_parametre_mean,axis=0)] 
    data_max = [np.max(alpha_mean,axis=0),np.max(beta_mean,axis=0),np.max(W_parametre_mean,axis=0)] 
    data_min = [np.min(alpha_mean,axis=0),np.min(beta_mean,axis=0),np.min(W_parametre_mean,axis=0)] 
    
    hist_data = [n_alpha[-1,:].flatten(), n_beta[-1,:].flatten(), n_W_parametre[-1,:].flatten()]

    labels = ['$\\alpha$','$\\beta$','$W$']
    
    true_data = [[0]*T,[0.9702]*T,[1]*T]
    y_limit_top = [2,2,3]
    y_limit_bot = [-2,0,0.5]
    x_limit_left = y_limit_bot
    x_limit_right = y_limit_top

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(20,9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Plots pour la ligne 1
    for i in range(3):
        ax2 = fig.add_subplot(gs[0, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i])
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=y_limit_top[i],bottom=y_limit_bot[i])

    # Plots pour la ligne 2
    # Last filter step 
    for i in range(3):
        ax3 = fig.add_subplot(gs[1, i])
        sns.histplot(hist_data[i], ax=ax3)
        ax3.axvline(x = true_data[i][0], color = 'b')
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))
        ax3.set_xlim(left=x_limit_left[i],right=x_limit_right[i])

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.1, 1, 0.90])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    str_title = "Estimated parameters with Storvik's SIR filter \n with "+str(N)+' particles and '+str(nbr_iteration)+' iterations on stochastic volatility model'
    fig.suptitle(str_title, fontsize=20,fontname='Times New Roman')

    # Affichage de la figure
    plt.show()

    estimated_W, estimated_alpha, estimated_beta, mu_estimated = parameters_estimations(n_W,n_W_parametre, n_alpha, n_beta, n_mu)
    print('W =', estimated_W)
    print('alpha =', estimated_alpha)
    print('beta =', estimated_beta)

    return estimated_W, estimated_alpha, estimated_beta, 0