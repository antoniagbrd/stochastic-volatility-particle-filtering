import numpy as np
from random import gauss
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import matplotlib.gridspec as gridspec


from scipy.stats import norm, invgauss, invgamma, multivariate_normal

import statistics

import time 
from tqdm import tqdm


#######################################################################
# Trajectories fonctions
#######################################################################

" -> generates a random trajectory and its observations <- "
" -> return : hidden states, observations "
def generate_trajectory_kitagawa_model(T) : 
    X_0=1
    alpha_par=0.5
    beta_par=25
    gamma_par=8
    V_par=5
    W_par=1

    observations = np.zeros((T,2))
    observations[0,0] = X_0
    observations[0,1] = observations[0,0]**2/20 + gauss(0,V_par)

    for k in range(1,T):
        observations[k,0] = alpha_par*observations[k-1,0]+beta_par*observations[k-1,0]/(1+observations[k-1,0]**2)+gamma_par*np.cos(1.2*k) + gauss(0,W_par) # X_t
        observations[k,1] = observations[k,0]**2/20 + gauss(0,V_par) # Y_t

    return observations[:,0],observations[:,1]

" -> plots a trajectory and its observations <-"
def show_a_trajectory_kitagawa_model(T,x,y):
    L_t=[t for t in range(T)]

    plt.figure(figsize=(20,8))
    plt.plot(L_t,x,'-o',color='red',label='trajectoire')
    plt.plot(L_t,y,'-o',color='b',label='observations y')
    plt.xlabel('Temps')
    plt.title('Hidden states and observations generate from kitagawa model')
    plt.legend()
    plt.show()

" -> plots and generates a random trajectory and its observations  <-"
def show_random_trajectory_kitagawa_model(T):

    x,y=generate_trajectory_kitagawa_model(T)
    L_t=[t for t in range(T)]

    plt.figure(figsize=(20,8))
    plt.plot(L_t,x,'-o',color='red',label='trajectoire')
    plt.plot(L_t,y,'-o',color='b',label='observations y')
    plt.xlabel('Temps')
    plt.title('Hidden states and observations generate from Kitawa model')
    plt.legend()
    plt.show()

#######################################################################
# Filters fonctions
#######################################################################

" -> runs SIR filter on a kitagawa model (parameters are known here) <-"
" -> return : weights, hidden states estimated "
def SIR_kitagawa_model(T,Y,N,alpha_par,beta_par,gamma_par,W_par,V_par):
    # N est le nombre de particules

    X_0 = np.random.randn(N)
    X = np.zeros((T,N))
    X[0,:] = X_0

    w_0 = norm.pdf(Y[0],X_0**2/20,V_par)
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0
    
    for t in range(1,T):
        A = np.random.choice(range(N),N,p=W[t-1,:])
           
        X[t,:] = np.random.normal(loc=alpha_par*X[t-1][A]+beta_par*X[t-1][A]/(1+X[t-1][A]**2)+gamma_par*np.cos(1.2*t),scale=W_par,size=N)
            
        W[t,:] = norm.pdf(Y[t],X[t,:]**2/20,V_par)
        W[t,:] = W[t,:]/(W[t,:].sum()) 

    return W,X

" -> runs Storvik's filter on a kitagawa model (parameters are unknownw here)"
" -> return : weights, hidden states, estimated a, estimated sigma "
def storvik_SIR_kitagawa_model(T,Y,N):
    # N est le nombre de particules
    # T est le nombre d'itération
    
    # Initialisations des paramètres  
    n_0 = 2 # Degrés de liberté
    d_0 = 2 # Paramètre de forme
    b_0 = np.array([0.5, 25, 8]) # Vecteur de moyenne
    B_0 = np.diag([1/0.25**2,1/10**2,1/4**2]) # Matrice de covariance
    nu_0 = 2
    delta_0 = 2  
    
    W_parametre = np.empty((T,N)) # W_parametre[i,j] correspond à W au ième temps de la jème particule
    V_parametre = np.empty((T,N)) # V_parametre[i,j] correspond à V au ième temps de la jème particule 
    alpha = np.zeros((T,N)) # pareil
    beta = np.zeros((T,N)) # pareil
    gamma = np.zeros((T,N)) # pareil
    
    # Statistique S
    S = np.zeros((T,N,6),dtype=object) # Statistique
    S[0,:,:] = np.array([[B_0,n_0,nu_0,b_0,d_0,delta_0]]*N,dtype=object)
    
    for i in range(N):
            V_parametre[0,i] = invgamma.rvs(a=S[0][i][2], scale=np.sqrt(S[0][i][5]))
            W_parametre[0,i] = invgamma.rvs(a=S[0][i][1], scale=np.sqrt(S[0][i][4]))
            alpha[0,i],beta[0,i],gamma[0,i] = np.random.multivariate_normal(mean=S[0,i,3], cov=np.sqrt(W_parametre[0,i])*np.linalg.inv(S[0,i,0])).tolist()

    # Initialisation de l'hiddent state et des poids
    X_0 = np.random.randn(N)
    X = np.zeros((T,N))
    X[0,:] = X_0
    
    w_0 = norm.pdf(Y[0],X_0**2/20,np.sqrt(V_parametre[0,:]))
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0
        
    for t in tqdm(range(1,T)):   
        
        # Etape 2 : Propagate x_t ~ p(x_t | x_{t-1}, theta)
        X[t,:] = np.random.normal(alpha[t-1,:]*X[t-1] + beta[t-1,:]*X[t-1]/(1+X[t-1]**2) + gamma[t-1,:]*np.cos(1.2*t-1), np.sqrt(W_parametre[t-1,:]), N)

        # Etape 3 : Compute weights w_t = p(y_t | x_t, theta)
        W[t,:] = norm.pdf(Y[t], X[t,:]**2/20, np.sqrt(V_parametre[t-1,:]))
        W[t,:] = W[t,:]/(W[t,:].sum())

        # Sample theta ~ p(theta | s_{t-1})
        # Update sufficient statistics
        for i in range(N):
            B_t_prec = S[t-1,i,0]
            n_t_prec = S[t-1,i,1]
            nu_t_prec = S[t-1,i,2]
            b_t_prec = S[t-1,i,3]
            d_t_prec = S[t-1,i,4]
            delta_t_prec = S[t-1,i,5]
                
            x_t_prec = X[t-1,i]
            x_t = X[t,i]
            y_t = Y[t]
            
            F_t = np.array([x_t_prec,x_t_prec/(1+x_t_prec**2),np.cos(1.2*(t-1))])
            
            B_t = B_t_prec + np.outer(F_t,F_t.T)
            n_t = n_t_prec + 1/2
            nu_t = nu_t_prec + 1/2
            b_t = np.linalg.inv(B_t) @ (B_t_prec @ b_t_prec + F_t*x_t)
            d_t = d_t_prec + (np.transpose(b_t_prec) @ B_t_prec @ b_t_prec + x_t**2 - np.transpose(b_t) @ B_t @ b_t).sum()/2
            delta_t = delta_t_prec + (y_t-x_t**2/20)**2/2

            S[t,i,0] = B_t
            S[t,i,1] = n_t
            S[t,i,2] = nu_t
            S[t,i,3] = b_t
            S[t,i,4] = d_t
            S[t,i,5] = delta_t
            
            V_parametre[t,i] = invgamma.rvs(a=np.sqrt(nu_t), scale=np.sqrt(delta_t))            
            W_parametre[t,i] = invgamma.rvs(a=np.sqrt(n_t), scale=np.sqrt(d_t))
            
            alpha[t,i],beta[t,i],gamma[t,i] = np.random.multivariate_normal(mean=S[t-1,i,3], cov=np.sqrt(W_parametre[t,i])*np.linalg.inv(S[t-1,i,0])).tolist()
            
        # Etape 5 : Resample x_t and s_t
        A = np.random.choice(range(N),N,p=W[t,:])
        X[t,:] = X[t,A]
        S[t,:,:] = S[t,A,:]
        # V_parametre[t,:] = V_parametre[t,A]
        # W_parametre[t,:] = W_parametre[t,A]
        # alpha[t,:] = alpha[t,A]
        # beta[t,:] = beta[t,A]
        # gamma[t,:] = gamma[t,A]
        
        
    return W,X,alpha,beta,gamma,W_parametre,V_parametre

" -> estimation of parameters <- "
" -> return : the estimated parameters"
def parameters_estimations(n_W,n_alpha,n_beta,n_gamma,n_W_parametre,n_V_parametre):

    nbr_iteration = np.shape(n_W)[0]
    
    estimated_alpha = (n_W[:,-1,:]*n_alpha[:,-1,:]).sum()/nbr_iteration
    estimated_beta = (n_W[:,-1,:]*n_beta[:,-1,:]).sum()/nbr_iteration
    estimated_gamma = (n_W[:,-1,:]*n_gamma[:,-1,:]).sum()/nbr_iteration
    estimated_W = (n_W[:,-1,:]*n_W_parametre[:,-1,:]).sum()/nbr_iteration
    estimated_V =(n_W[:,-1,:]*n_V_parametre[:,-1,:]).sum()/nbr_iteration

    return estimated_alpha, estimated_beta, estimated_gamma, estimated_W, estimated_V

" -> smoother PLS "
" -> arguments : needs the estimated parameters, weights and hidden states estimations "
" -> return a hidden states estimations smoothed with PLS"
def backward_filter_kitagawa_model(T,N,W,X,alpha,beta,gamma,W_parametre) :
    X_smooth = np.zeros((T, N))

    # Select a pair (x_T(i),theta(i)) from step 1
    index = np.random.choice(range(N), p=W[T-1,:]) 
    X_smooth[T - 1,:] = X[T-1, index] 

    if np.shape(alpha) == ():
        alpha_smooth = alpha
        beta_smooth = beta
        gamma_smooth = gamma
        W_smooth = W_parametre
        
    else : 
        alpha_smooth = alpha[T-1,index]
        beta_smooth = beta[T-1,index]
        gamma_smooth = gamma[T-1,index]
        W_smooth = W_parametre[T-1,index]

    for t in tqdm(range(T - 2, -1, -1)): 
        #construct weight for the resampling
        w_t = norm.pdf(X_smooth[t+1,:], loc=alpha_smooth*X[t]+beta_smooth*X[t]/(1+X[t]**2)+gamma_smooth*np.cos(1.2*t), scale=W_smooth)
        w_t = w_t/(w_t.sum())
        #resampling particles
        index = np.random.choice(range(N),N, p=w_t)
        X_smooth[t,:] = X[t,index]

    return X_smooth

def FBS_stochastic_kitagawa_model(T,N,weights,X,alpha,beta,gamma,W_parametre):

    new_weights = np.zeros((T,N))
    new_weights[-1,:] = weights[-1,:]

    for t in tqdm(range(T-2,-1,-1)):
        
        for i in range(N):
            somme_j = 0
            for j in range(N):
                
                density = norm.pdf(x=X[t+1,j], loc=alpha*X[t,i]+beta*X[t,i]/(1+X[t,i]**2)+gamma*np.cos(1.2*t), scale=W_parametre)
                denominator = np.sum(weights[t,:] * norm.pdf(x=X[t+1,j], loc=alpha*X[t,:]+beta*X[t,:]/(1+X[t,:]**2)+gamma*np.cos(1.2*t), scale=W_parametre))
                somme_j = new_weights[t+1,j] * density / denominator
                
            new_weights[t,i] += weights[t,i]*somme_j
          
        new_weights[t,:] = new_weights[t,:]/np.sum(new_weights[t,:]) 

    return new_weights 

#######################################################################
# Intermediate functions (not very usefull)
#######################################################################
" -> generates a random trajectory and runs Storvik's filter (and estimates unkonwn parameters) <- "
def run_storvik_SIR_kitagawa_model(T,N):

    _,Y = generate_trajectory_kitagawa_model(T)
    return storvik_SIR_kitagawa_model(T,Y,N)

" -> run Storvik's filter on many random trajectories <- "
def run_n_storvik_SIR_kitagawa_model(T,N,nbr_iteration):
    # on run nbr_iteration de fois storvik_SIR et on stocke theta et les poids
    
    n_W = np.zeros((nbr_iteration,T,N))

    n_W_parametre = np.zeros((nbr_iteration,T,N))
    n_V_parametre = np.zeros((nbr_iteration,T,N)) # n_a[i,t,j] contient l'a de la ième itération au tème temps et de la jème particule
    n_alpha = np.zeros((nbr_iteration,T,N))
    n_beta = np.zeros((nbr_iteration,T,N))
    n_gamma = np.zeros((nbr_iteration,T,N))
    
    for i in tqdm(range(nbr_iteration)):
        W,_,alpha,beta,gamma,W_parametre,V_parametre = run_storvik_SIR_kitagawa_model(T,N)
    
        n_W[i,:,:] = W
        n_W_parametre[i,:,:] = W_parametre
        n_V_parametre[i,:,:] = V_parametre
        n_alpha[i,:,:] = alpha
        n_beta[i,:,:] = beta
        n_gamma[i,:,:] = gamma
    
    return n_W,n_alpha,n_beta,n_gamma,n_W_parametre,n_V_parametre


#######################################################################
# Show a run functions
#######################################################################

# SIR 
" -> show a run of SIR on a chosen trajectory <- "
def show_SIR_trajectory_kitagawa_model(T,N,hidden_states,observations,alpha_par,beta_par,gamma_par,W_par,V_par):
    L_t = np.arange(T)

    W,X = SIR_kitagawa_model(T,observations,N,alpha_par,beta_par,gamma_par,W_par,V_par)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hidden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("SIR filter with "+str(N)+' particles on Kitagawa model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show() 

" -> show a run of SIR on a random trajectory <- "
def show_SIR_random_trajectory_kitagawa_model(T,N,alpha_par,beta_par,gamma_par,W_par,V_par):
    hidden_states,observations = generate_trajectory_kitagawa_model(T)
    show_SIR_trajectory_kitagawa_model(T,N,hidden_states,observations,alpha_par,beta_par,gamma_par,W_par,V_par)

# Storvik's filter
" -> show a run of Storvik's filter on a chosen trajectory <- "
def show_storvik_SIR_kitagawa_model(T,N,hiden_states,observations):
    L_t = np.arange(T)

    W,X,alpha,beta,gamma,W_parametre,V_parametre = storvik_SIR_kitagawa_model(T,observations,N)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    alpha_estimated = (W[-1,:]*alpha[-1,:]).sum()
    beta_estimated = (W[-1,:]*beta[-1,:]).sum()
    gamma_estimated = (W[-1,:]*gamma[-1,:]).sum()
    W_estimated = (W[-1,:]*W_parametre[-1,:]).sum()
    V_estimated = (W[-1,:]*V_parametre[-1,:]).sum()

    print('alpha =',round(alpha_estimated,3),'(=0.5)')
    print('beta =',round(beta_estimated,3),'(=25)')
    print('gamma =',round(gamma_estimated,3),'(=8)')
    print('W =', round(W_estimated,3),'(=1)')
    print('V =', round(V_estimated,3),'(=5)')

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hiden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("Storvik's filter with "+str(N)+' particles Kitagawa model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show()

" -> show a run of Storvik's filter on a random trajectory <- "
def show_random_storvik_SIR_kitagawa_model(T,N):

    hiden_states,observations = generate_trajectory_kitagawa_model(T)
    show_storvik_SIR_kitagawa_model(T,N,hiden_states,observations)

#######################################################################
# Show estimated parameters functionns
####################################################################### 
" -> run Storvik's filter on a random trajectory and shows the estimated parameters <- "
def show_random_stovik_SIR_and_parameters_kitagawa_model(T,N):

    x,y=generate_trajectory_kitagawa_model(T)
    L_t = np.arange(T)

    W,X,alpha,beta,gamma,W_parametre,V_parametre = storvik_SIR_kitagawa_model(T,y,N)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    alpha_estimated = (W[-1,:]*alpha[-1,:]).sum()
    beta_estimated = (W[-1,:]*beta[-1,:]).sum()
    gamma_estimated = (W[-1,:]*gamma[-1,:]).sum()
    W_estimated = (W[-1,:]*W_parametre[-1,:]).sum()
    V_estimated = (W[-1,:]*V_parametre[-1,:]).sum()

    print('alpha =',round(alpha_estimated,3),'(=0.5)')
    print('beta =',round(beta_estimated,3),'(=25)')
    print('gamma =',round(gamma_estimated,3),'(=8)')
    print('W =', round(W_estimated,3),'(=1)')
    print('V =', round(V_estimated,3),'(=5)')

    data_mean = [np.sum(alpha*W, axis=1),np.sum(beta*W, axis=1),np.sum(gamma*W, axis=1),np.sum(W_parametre*W, axis=1),np.sum(V_parametre*W, axis=1)]
    data_max = [np.max(alpha, axis=1),np.max(beta, axis=1),np.max(gamma, axis=1),np.max(W_parametre, axis=1),np.max(V_parametre, axis=1)]
    data_min = [np.min(alpha, axis=1),np.min(beta, axis=1),np.min(gamma, axis=1),np.min(W_parametre, axis=1),np.min(V_parametre, axis=1)]

    hist_data = [alpha[-1,:],beta[-1,:],gamma[-1,:],W_parametre[-1,:],V_parametre[-1,:]]

    true_data = [[0.5]*T,[25]*T,[8]*T,[1]*T,[5]*T]

    labels = ['$\\alpha$','$\\beta$','$\\gamma$','$W$','$V$']

    top_limit = [1,40,14,10,15]
    bot_limit = [-0.5,0,-1,-1,-1]

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(20,9))
    gs = gridspec.GridSpec(3, 5, height_ratios=[2, 1, 1])

    # Plot pour la ligne 1
    ax1 = fig.add_subplot(gs[0, :])
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='Storvik\'s filter' ,marker="o",ax=ax1)
    sns.lineplot(x=L_t,y=x,ax=ax1,label='hidden state',marker="o",color='black')
    ax1.set_xlabel('Temps')
    ax1.legend()

    # Plots pour la ligne 2
    for i in range(5):
        ax2 = fig.add_subplot(gs[1, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i],ax=ax2)
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=top_limit[i],bottom=bot_limit[i])

    # Plots pour la ligne 3
    # Last filter step 
    for i in range(5):
        ax3 = fig.add_subplot(gs[2, i])
        sns.histplot(hist_data[i], ax=ax3)
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    fig.suptitle("Storvik's SIR filter on Kitagawa model with "+str(N)+' particles', fontsize=30,fontname='Times New Roman')  

    # Affichage de la figure
    plt.show()

" -> show Storvik's filter on many random trajectories and shows the estimated parameters <- "
def show_n_runs_kitagawa_model(n_W,n_alpha,n_beta,n_gamma,n_W_parametre,n_V_parametre):
    
    nbr_iteration = np.shape(n_W)[0]
    N = np.shape(n_W[0,0])[0]
    T = np.shape(n_W[0])[0]
    L_t = np.arange(T)
    
    # n_alpha[i,t,j] contient l'alpha de la ième itération au tème temps et de la jème particule
    
    alpha_mean = [np.sum(n_alpha[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    beta_mean = [np.sum(n_beta[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    gamma_mean = [np.sum(n_gamma[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    W_parametre_mean = [np.sum(n_W_parametre[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    V_parametre_mean = [np.sum(n_V_parametre[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]

    data_mean = [np.mean(alpha_mean,axis=0),np.mean(beta_mean,axis=0),np.mean(gamma_mean,axis=0),np.mean(W_parametre_mean,axis=0),np.mean(V_parametre_mean,axis=0)] 
    data_max = [np.max(alpha_mean,axis=0),np.max(beta_mean,axis=0),np.max(gamma_mean,axis=0),np.max(W_parametre_mean,axis=0),np.max(V_parametre_mean,axis=0)] 
    data_min = [np.min(alpha_mean,axis=0),np.min(beta_mean,axis=0),np.min(gamma_mean,axis=0),np.min(W_parametre_mean,axis=0),np.min(V_parametre_mean,axis=0)] 
    
    hist_data = [n_alpha[-1,:].flatten(), n_beta[-1,:].flatten(), n_gamma[-1,:].flatten(), n_W_parametre[-1,:].flatten(), n_V_parametre[-1,:].flatten()]

    labels = ['$\\alpha$','$\\beta$','$\\gamma$','$W$','$V$']
    
    true_data = [[0.5]*T,[25]*T,[8]*T,[1]*T,[5]*T]
    y_limit_top = [1,40,14,5,15]
    y_limit_bot = [-1,-1,-1,-1,-1]
    x_limit_left = [-1,20,0,0,0]
    x_limit_right = [1,30,15,10,15]

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(20,9))
    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1])

    # Plots pour la ligne 1
    for i in range(5):
        ax2 = fig.add_subplot(gs[0, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i])
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=y_limit_top[i],bottom=y_limit_bot[i])

    # Plots pour la ligne 2
    # Last filter step 
    for i in range(5):
        ax3 = fig.add_subplot(gs[1, i])
        sns.histplot(hist_data[i], ax=ax3)
        ax3.axvline(x = true_data[i][0], color = 'b')
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))
        ax3.set_xlim(left=x_limit_left[i],right=x_limit_right[i])

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.1, 1, 0.90])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    str_title = "Estimated parameters with Storvik's SIR filter \n with "+str(N)+' particles and '+str(nbr_iteration)+' iterations on Kitagawa model'
    fig.suptitle(str_title, fontsize=20,fontname='Times New Roman')

    # Affichage de la figure
    plt.show()

    return parameters_estimations(n_W,n_alpha,n_beta,n_gamma,n_W_parametre,n_V_parametre)