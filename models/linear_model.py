from random import gauss
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
def generate_trajectory_linear_model(T,X_0,a,sigma) : 
    obs = np.zeros((T,2))
    obs[0,0] = X_0
    obs[0,1] = obs[0,0] + gauss(0,1)

    for k in range(1,T):
        obs[k,0] = a*obs[k-1,0] + sigma*gauss(0,sigma) # x_k
        obs[k,1] = obs[k,0] + gauss(0,1) # z_k
    return obs[:,0],obs[:,1]

" -> plots a trajectory and its observations <-"
def show_a_trajectory_linear_model(T,x,y):

    L_t=[t for t in range(T)]

    plt.figure(figsize=(20,8))
    plt.plot(L_t,x,'-o',color='red',label='trajectoire')
    plt.plot(L_t,y,'-o',color='b',label='observations y')
    plt.xlabel('Temps')
    plt.title('Hidden states and observations generate from linear model')
    plt.legend()
    plt.show()

" -> plots and generates a random trajectory and its observations  <-"
def show_random_trajectory_linear_model(T,X_0):

    x,y=generate_trajectory_linear_model(T,X_0,a=0.9,sigma=1)
    L_t=[t for t in range(T)]

    plt.figure(figsize=(20,8))
    plt.plot(L_t,x,'-o',color='red',label='trajectoire')
    plt.plot(L_t,y,'-o',color='b',label='observations y')
    plt.xlabel('Temps')
    plt.title('Hidden states and observations generate from linear model')
    plt.legend()
    plt.show()

#######################################################################
# Filters fonctions
#######################################################################

" -> runs SIR filter on a linear model (parameters are known here) <-"
" -> return : weights, hidden states estimated "
def SIR_linear_model(T,Y,N,a,sigma):

    # N est le nombre de particules
    # T est le nombre d'itération

    # Initialisation --------------------------------------
    X_0 = np.random.randn(N)
    X = np.zeros((T,N))
    X[0,:] = X_0

    w_0 = norm.pdf(x=Y[0],loc=X[0,:],scale=1)
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0
    
    # Paramètres ------------------------------------------
    
    for t in range(1,T):      
        A = np.random.choice(range(N),N,p=W[t-1,:])
        X[t-1,:] = X[t-1,A]
        
        X[t,:] = np.random.normal(a*X[t-1,:],1,N)
        
        W[t,:] = W[t-1,:]*norm.pdf(x=Y[t],loc=X[t,:],scale=sigma)
        W[t,:] = W[t,:]/(W[t,:].sum()) 
    
    return W,X

" -> runs Storvik's filter on a linear model (parameters are unknownw here)"
" -> return : weights, hidden states, estimated a, estimated sigma "
def storvik_SIR_linear_model(T,Y,N):
    # N est le nombre de particules
    # T est le nombre d'itération
    
    # Initialisations des paramètres  

    sigma = np.zeros((T,N)) #
    a = np.zeros((T,N)) # 
    
    # Statistique S
    B_stat = np.zeros((T,N))
    n_stat = np.zeros((T,N))
    nu_stat = np.zeros((T,N))
    b_stat = np.zeros((T,N))
    d_stat = np.zeros((T,N))
    delta_stat = np.zeros((T,N))
    
    B_stat[0,:] = 1
    n_stat[0,:] = 2
    nu_stat[0,:] = 2
    b_stat[0,:] = 0.5
    d_stat[0,:] = 2
    delta_stat[0,:] = 2

    # Initialisation --------------------------------------
    X_0 = np.random.randn(N)
    X = np.zeros((T,N))
    X[0,:] = X_0

    # Générer un échantillon NIG
    sigma[0,:] = invgamma.rvs(a=d_stat[0,:], scale=n_stat[0,:])
    a[0,:] = norm.rvs(loc=b_stat[0,:], scale=np.sqrt(sigma[0,:]) / B_stat[0,:])
    
    w_0 = norm.pdf(Y[0],X_0,1)
    w_0 = w_0/(w_0.sum())
    W = np.zeros((T,N))
    W[0,:] += w_0
    
    # Paramètres ------------------------------------------
    
    for t in range(1,T):       
        
        X[t,:] = np.random.normal(loc=a[t-1,:]*X[t-1,:],scale=np.sqrt(sigma[t-1,:]))
        
        W[t,:] = W[t-1,:]*norm.pdf(x=Y[t],loc=X[t,:],scale=1)
        W[t,:] = W[t,:]/(W[t,:].sum()) 
        
        # Update s_t(i) :
      
        x_t_prec = X[t-1,:]
        x_t = X[t,:]
        y_t = Y[t]

        B_stat[t,:] = B_stat[t-1,:] + x_t_prec**2
        n_stat[t,:] = n_stat[t-1,:] + 1/2
        nu_stat[t,:] = nu_stat[t-1,:] + 1/2
        b_stat[t,:] = (B_stat[t-1,:]*b_stat[t-1,:]+x_t_prec*x_t)/B_stat[t,:]
        d_stat[t,:] = d_stat[t-1,:] + (b_stat[t-1,:]**2*B_stat[t-1,:]+x_t**2-b_stat[t,:]**2*B_stat[t,:])/2
        delta_stat[t,:] = delta_stat[t-1,:] + (y_t - x_t)**2/2
        
        # Générer un échantillon NIG
        sigma[t,:] = invgamma.rvs(a=d_stat[t,:], scale=n_stat[t,:])
        a[t,:] = norm.rvs(loc=b_stat[t,:], scale=np.sqrt(sigma[t,:]) / B_stat[t,:])
        
        A = np.random.choice(range(N),N,p=W[t,:])
        X[t,:] = X[t,A]
        
        a[t,:] = a[t,A]
        sigma[t,:] = sigma[t,A]
        
        B_stat[t,:] = B_stat[t,A]
        n_stat[t,:] = n_stat[t,A]
        nu_stat[t,:] = nu_stat[t,A]
        b_stat[t,:] = b_stat[t,A]
        d_stat[t,:] = d_stat[t,A]
        delta_stat[t,:] = delta_stat[t,A]
        
    return W,X,a,np.sqrt(sigma)

" -> estimation of parameters <- "
" -> return : the estimated parameters"
def parameters_estimations(n_W,n_a,n_sigma):

    nbr_iteration = np.shape(n_W)[0]
    
    a_estimated = (n_W[:,-1,:]*n_a[:,-1,:]).sum()/nbr_iteration
    sigma_estimated = (n_W[:,-1,:]*n_sigma[:,-1,:]).sum()/nbr_iteration

    return a_estimated,sigma_estimated

#######################################################################
# Intermediate functions (not very usefull)
#######################################################################
" -> generates a random trajectory and runs Storvik's filter (and estimates unkonwn parameters) <- "
def run_storvik_SIR_linear_model(T,N):
    
    _,y=generate_trajectory_linear_model(T,1,a=0.9,sigma=1)
    W,X,a,sigma = storvik_SIR_linear_model(T,y,N)
    return W,X,a,sigma

" -> run Storvik's filter on many random trajectories <- "
def run_n_storvik_SIR_linear_model(T,N,nbr_iteration):
    # on run nbr_iteration de fois storvik_SIR et on stocke theta et les poids
    
    n_W = np.zeros((nbr_iteration,T,N))
    n_a = np.zeros((nbr_iteration,T,N)) # n_a[i,t,j] contient l'a de la ième itération au tème temps et de la jème particule
    n_sigma = np.zeros((nbr_iteration,T,N))
    
    for i in tqdm(range(nbr_iteration)):
        W,_,a,sigma = run_storvik_SIR_linear_model(T,N)
    
        n_W[i,:,:] = W
        n_a[i,:,:] = a
        n_sigma[i,:,:] = sigma
    
    return n_W,n_a,n_sigma

#######################################################################
# Show a run functions
#######################################################################

# SIR 
" -> show a run of SIR on a chosen trajectory <- "
def show_SIR_trajectory_linear_model(T,N,hidden_states,observations,a,sigma):
    L_t = np.arange(T)

    W,X = SIR_linear_model(T,observations,N,a,sigma)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hidden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("SIR filter with "+str(N)+' particles on linear model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show() 

" -> show a run of SIR on a random trajectory <- "
def show_SIR_random_trajectory_linear_model(T,N,a,sigma):
    x,y=generate_trajectory_linear_model(T,1,a=0.9,sigma=1)
    L_t = np.arange(T)

    W,X = SIR_linear_model(T,y,N,a,sigma)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=x,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("SIR filter with "+str(N)+' particles on linear model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show()

# Storvik's filter
" -> show a run of Storvik's filter on a chosen trajectory <- "
def show_storvik_SIR_linear_model(T,N,hiden_states,observations,a,sigma):
    L_t = np.arange(T)

    W,X,a,sigma = storvik_SIR_linear_model(T,observations,N)
    hidden_state_estimation = np.sum(W*X,axis=1) 

    print('estimated a =',round((W[-1,:]*a[-1,:]).sum(),3),'(= 0.9 ?)')
    print('estimated sigma =',round((W[-1,:]*sigma[-1,:]).sum(),3),'(= 1 ?)')

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=hiden_states,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("Storvik's filter with "+str(N)+' particles on linear model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show()

" -> show a run of Storvik's filter on a random trajectory <- "
def show_random_storvik_SIR_linear_model(T,N,a,sigma):

    x,y=generate_trajectory_linear_model(T,1,a,sigma)
    L_t = np.arange(T)

    W,X,a,sigma = storvik_SIR_linear_model(T,y,N)
    hidden_state_estimation = np.sum(W*X,axis=1) 

    print('estimated a =',round((W[-1,:]*a[-1,:]).sum(),3),'(= 0.9 ?)')
    print('estimated sigma =',round((W[-1,:]*sigma[-1,:]).sum(),3),'(= 1 ?)')

    plt.figure(figsize=(20,8))
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='filtre particulaire',marker="o")
    sns.lineplot(x=L_t,y=x,label='hidden state',marker="o",color='black')
    plt.xlabel('Temps')
    plt.title("Storvik's filter with "+str(N)+' particles on linear model', fontsize=30,fontname='Times New Roman')  
    plt.legend()
    plt.show()

#######################################################################
# Show estimated parameters functionns
####################################################################### 
" -> run Storvik's filter on a random trajectory and shows the estimated parameters <- "
def show_random_stovik_SIR_and_parameters_linear_model(T,N):
    x,y=generate_trajectory_linear_model(T,1,0.9,1)
    L_t = np.arange(T)

    W,X,a,sigma = storvik_SIR_linear_model(T,y,N)

    hidden_state_estimation = np.sum(W*X,axis=1) 

    print('estimated a =',round((W[-1,:]*a[-1,:]).sum(),3),'(= 0.9 ?)')
    print('estimated sigma =',round((W[-1,:]*sigma[-1,:]).sum(),3),'(= 1 ?)')

    data_mean = [np.sum(a*W, axis=1),np.sum(sigma*W, axis=1)]
    data_max = [np.max(a, axis=1),np.max(sigma, axis=1)]
    data_min = [np.min(a, axis=1),np.min(sigma, axis=1)]

    hist_data = [a[-1,:],sigma[-1,:]]

    labels = ['$a$','$\\sigma$']
    true_data = [[0.9]*T,[1]*T]
    top_limit = [6,5]
    bot_limit = [-1,-1]

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(20,9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # Plot pour la ligne 1
    ax1 = fig.add_subplot(gs[0, :])
    sns.lineplot(x=L_t,y=hidden_state_estimation,label='Storvik\'s filter' ,marker="o",ax=ax1)
    sns.lineplot(x=L_t,y=x,ax=ax1,label='hidden state',marker="o",color='black')
    ax1.set_xlabel('Temps')
    ax1.legend()

    # Plots pour la ligne 2
    for i in range(2):
        ax2 = fig.add_subplot(gs[1, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i],ax=ax2)
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=top_limit[i],bottom=bot_limit[i])

    # Plots pour la ligne 3
    # Last filter step 
    for i in range(2):
        ax3 = fig.add_subplot(gs[2, i])
        sns.histplot(hist_data[i], ax=ax3,kde=True)
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    fig.suptitle("Storvik's SIR filter on linear model with "+str(N)+' particles', fontsize=30,fontname='Times New Roman')  

    # Affichage de la figure
    plt.show()

" -> show Storvik's filter on many random trajectories and shows the estimated parameters <- "
def show_n_runs_linear_model(n_W,n_a,n_sigma):

    nbr_iteration = np.shape(n_W)[0]
    N = np.shape(n_W[0,0])[0]
    T = np.shape(n_W[0])[0]
    L_t = np.arange(T)
    
    # n_a[i,t,j] contient l'a de la ième itération au tème temps et de la jème particule
    
    a_mean = [np.sum(n_a[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]
    sigma_mean = [np.sum(n_sigma[i,:,:]*n_W[i,:,:], axis=1) for i in range(nbr_iteration)]

    data_mean = [np.mean(a_mean,axis=0),np.mean(sigma_mean,axis=0)] 
    data_max = [np.max(a_mean,axis=0),np.max(sigma_mean,axis=0)] 
    data_min = [np.min(a_mean,axis=0),np.min(sigma_mean,axis=0)] 
    
    n_a = n_a[:,-1,:]
    n_sigma = n_sigma[:,-1,:]
    hist_data = [n_a.flatten(), n_sigma.flatten()]
    labels = ['a','$\\sigma$']
    y_limit_top = [6,5]
    y_limit_bot = [-1,-1]

    true_data = [[0.9]*T,[1]*T]

    # Création de la figure et des sous-graphiques avec GridSpec
    fig = plt.figure(tight_layout=True,figsize=(10,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Plots pour la ligne 1
    for i in range(2):
        ax2 = fig.add_subplot(gs[0, i])
        sns.lineplot(x=L_t,y=data_mean[i],ax=ax2)
        ax2.fill_between(L_t, data_max[i], data_min[i],color='grey',alpha=0.3)
        sns.lineplot(x=L_t,y=true_data[i])
        ax2.set_title(labels[i])
        ax2.set_xlabel('Temps')
        ax2.set_ylim(top=y_limit_top[i],bottom=y_limit_bot[i])

    # Plots pour la ligne 2
    # Last filter step 
    for i in range(2):
        ax3 = fig.add_subplot(gs[1, i])
        sns.histplot(hist_data[i], ax=ax3)
        ax3.set_title(labels[i])
        ax3.set_xlabel('Valeur de '+ str(labels[i]))

    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout(rect=[0, 0.1, 1, 0.90])  # Ajustez les valeurs ([left, bottom, right, top]) pour définir l'espacement souhaité
    str_title = "Estimated parameters with Storvik's SIR filter \n with "+str(N)+' particles and '+str(nbr_iteration)+' iterations on linear model'
    fig.suptitle(str_title, fontsize=20,fontname='Times New Roman')

    # Affichage de la figure
    plt.show()

    return parameters_estimations(n_W,n_a,n_sigma)
