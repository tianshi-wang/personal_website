"""Module for calculate customer lifetime value (CLV)
This module is still in test
Currently this module is independent, not implemented in machine learning module or webapp
"""
import pickle

import pandas as pd
import numpy as np
import pystan
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from hashlib import md5

from connLocalDB import connDB

engine, conn = connDB()
"""
Read collections table and write new table collectiongroupby
:return: format as "userId, moduleName, year, month, count"
"""
# query:
sql_query = """
SELECT "userId", created_date, "Amount"
FROM orders
WHERE "userId" IS NOT NULL 
AND created_date>'2017-07-12'::date
;
"""
transactions = pd.read_sql_query(sql_query, conn)
transactions['week'] = transactions['created_date'].apply(lambda x: (x.year-2017)*56+x.isocalendar()[1])
transactions = transactions.drop('created_date',axis=1)
amount = transactions.pop('Amount')
transactions['Amount'] = amount

print(transactions.head())
ts_transactions = transactions.groupby(['week']).size()


def shift_date(x):
    x['shifted_week'] = x['week'].shift(-1)
    return x


transactions_tmp = transactions.sort_values(['week']).\
                        groupby(['userId'], as_index=True).apply(shift_date)

transactions_tmp.sort_values(['userId','week'], ascending=True, inplace=True)
transactions_tmp.dropna(inplace=True)

# Compute the IPT in days :
transactions_tmp['IPT'] = (transactions_tmp['shifted_week'] - transactions_tmp['week'])

print(transactions_tmp.head(5))

print("transactions_tmp IPT mean is: "+str(transactions_tmp['IPT'].mean()))
transactions_tmp['IPT'].hist(bins=40)
plt.yscale('log')
plt.xlabel('IPT (weeks)')
plt.ylabel('Number of Purchases')
plt.show()

n_purchases = transactions.groupby(['userId']).size()
print("Distribution of the number of purchases per customer : ")
print(n_purchases.min(axis=0), n_purchases.max(axis=0))
n_purchases.hist(bins=(n_purchases.max(axis=0) - n_purchases.min(axis=0)) + 1)

print("Plot number_of_customers vs number_of_purchases:")
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.show()

"""
Train RFM (recency, frequecy, and monetary value) model 
"""

end_calibration = 84   # End of June 2018
train = transactions[transactions.week <= end_calibration]
holdout = transactions[transactions.week > end_calibration]

# Bin transaction by week
train2 = train.sort_values(['week'], ascending=True).groupby(['userId', 'week'],as_index=False)['Amount'].sum()
print("Show head of bin transaction by week:")
print(train2.head())

def compute_rfm(x, end_calibration):
    x['recency'] = x['week'].max() - x['week'].min()
    x['frequency'] = x['week'].count()-1
    x['T'] = (end_calibration - x['week'].min())
    x['monetary_value'] = x['Amount'].mean()
    return x


# use the function compute_rfm to compute recency, frequency, T and monetary value
# for each group (each customer).
train3 = train2.groupby(['userId']).apply(lambda x: compute_rfm(x, end_calibration))
train3.head()

#
rfm = train3[['userId', 'recency', 'frequency', 'T', 'monetary_value']].groupby(['userId']).first()
rfm.describe()


paretonbd_model="""
data{
int<lower=0> n_cust; //number of customers 
vector<lower=0>[n_cust] x; 
vector<lower=0>[n_cust] tx; 
vector<lower=0>[n_cust] T; 
}

parameters{
// vectors of lambda and mu for each customer. 
// Here I apply limits between 0 and 1 for each 
// parameter. A value of lambda or mu > 1.0 is unphysical 
// since you don't enough time resolution to go less than 
// 1 time unit. 
vector <lower=0,upper=1.0>[n_cust] lambda; 
vector <lower=0,upper=1.0>[n_cust] mu;

// parameters of the prior distributions : r, alpha, s, beta. 
// for both lambda and mu
real <lower=0>r;
real <lower=0>alpha;
real <lower=0>s;
real <lower=0>beta;
}

model{

// temporary variables : 
vector[n_cust] like1; // likelihood
vector[n_cust] like2; // likelihood 

// Establishing hyperpriors on parameters r, alpha, s, and beta. 
r ~ normal(0.5,0.1);
alpha ~ normal(10,1);
s ~ normal(0.5,0.1);
beta ~ normal(10,1);

// Establishing the Prior Distributions for lambda and mu : 
lambda ~ gamma(r,alpha); 
mu ~ gamma(s,beta);

// The likelihood of the Pareto/NBD model : 
like1 = x .* log(lambda) + log(mu) - log(mu+lambda) - tx .* (mu+lambda);
like2 = (x + 1) .* log(lambda) - log(mu+lambda) - T .* (lambda+mu);

// Here we increment the log probability density (target) accordingly 
target+= log(exp(like1)+exp(like2));
}
"""

# here's the data we will provide to STAN :
data={'n_cust':len(rfm),
    'x':rfm['frequency'].values,
    'tx':rfm['recency'].values,
    'T':rfm['T'].values
}


def stan_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm.sampling(**kwargs)

iterations = 3000
warmup = 500

pareto_nbd_fit = stan_cache(paretonbd_model, model_name='paretonbd_model', \
                                  data=data, chains=1, iter=iterations, warmup=warmup)
trace = pareto_nbd_fit.extract()
lambdas = trace['lambda']
mus = trace['mu']


dt_train = 52.0 # 12 months
training_predictions = (lambdas/mus-lambdas/mus*np.exp(-mus*dt_train)).mean(axis=0)
rfm['model_train_count'] = training_predictions

rmse_train_count = (rfm['model_train_count'] - rfm['frequency']).apply(lambda x : x*x)
rmse_train_count = np.sqrt(rmse_train_count.sum()/len(rfm))
print('RMSE =', rmse_train_count)


def plot_scatter(dataframe, colx, coly, xlabel='Observed Counts',
                 ylabel='Predicted Counts',
                 xlim=[0, 16], ylim=[0, 16], density=True):
    """This function will plot a scatter plot of colx on the x-axis vs coly on the y-axis.
    If you want to add a color to indicate the density of points, set density=True

    Args :
        - dataframe (dataframe) : pandas dataframe containing the data of interest
        - colx (str) : name of the column you want to put on the x axis
        - coly (str) : same but for the y axis
        - xlabel (str) : label to put on the x axis
        - ylabel (str) : same for y axis
        - xlim (list) : defines the range of x values displayed on the chart
        - ylim (list) same for the y axis.
        - density (bool) : set True to add color to indicate density of point.

    """

    if not density:
        plt.scatter(dataframe[colx].values, dataframe[coly].values)
    else:
        xvals = dataframe[colx].values
        yvals = dataframe[coly].values
        xy = np.vstack([xvals, yvals])
        z = gaussian_kde(xy)(xy)
        plt.scatter(xvals, yvals, c=z, s=10, edgecolor='')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(np.linspace(xlim[0], xlim[1], 100),\
             np.linspace(ylim[0], ylim[1], 100),\
             color='black')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot()
    plt.show()

plot_scatter(rfm, 'frequency', 'model_train_count')

print('model_train_count')


print("Comparisons Between Predictions and the Holdout (validation) Set Observations")
def prob_alive_at_T(lam, mu, t_x, T):
    """Computes the probability of being alive at T given lambda, mu, t_x, and T"""
    return 1 /( 1 + mu / (mu + lam) * (np.exp((lam + mu) * (T - t_x)) - 1) )

# Predictions made over the holdout period of 6 months :
dt_hold = 12 # 12 weeks from July 1, 2018 to September 27, 2018

# Here we extract recency and T values:
tmp = rfm['T'].values
T_values = np.tile(tmp, [iterations - warmup, 1])
tmp2 = rfm['recency'].values
recency_values = np.tile(tmp2, [iterations - warmup, 1])

# Holdout counts predictions :
holdout_predictions = ((lambdas/mus - lambdas/mus*np.exp(-mus*dt_hold)) * \
                                prob_alive_at_T(lambdas, mus, recency_values, T_values)).mean(axis=0)

#holdout_predictions = (lambdas/mus - lambdas/mus*np.exp(-mus*dt_hold)) * prob_alive_at_t(lambdas, mus, t_x, T)
rfm['model_holdout_count'] = np.asarray(holdout_predictions)


# lets look at the observed number of transactions during the same time period :
# counts per customer per date :
holdout_counts = holdout.groupby(['userId', 'week'], as_index=False).size().reset_index()

# counts per customer :
holdout_counts = holdout_counts.groupby(['userId']).size()

# Let's merge with the rfm object.
rfm_with_holdout = rfm.merge(pd.DataFrame(holdout_counts), how='left', left_index=True, right_index=True)
rfm_with_holdout.rename(columns={0:'obs_holdout_count'}, inplace=True)
rfm_with_holdout.fillna(0, inplace=True)

# Let's now plot the data :

rmse_holdout_count=(rfm_with_holdout['model_holdout_count'] - rfm_with_holdout['obs_holdout_count']).apply(lambda x :x*x)
rmse_holdout_count=np.sqrt(rmse_holdout_count.sum()/len(rfm_with_holdout))
print('RMSE =', rmse_holdout_count)
plot_scatter(rfm_with_holdout, 'obs_holdout_count', 'model_holdout_count',xlim=[0, 4], ylim=[0, 4])


print("Next section for Gamma-Gamma model")

# This gamma-gamm model follows the Fader et al. (2004) Gamma-Gamma model closely.
# Again, this model can take a while to train. Recommend a few 1000's iterations.

gamma_gamma_model="""
data {
    // this is the data we pass to STAN : 
    int<lower=1> n_cust;         // number of customers 
    vector[n_cust] x;            // frequency + 1 
    vector[n_cust] mx;           // average purchase amount for each customer 
}

parameters {
    // These are the model parameters : 
    real <lower=0>p;             // scale parameter of the gamma distribution. Note that 
                                 // this parameter is not a vector. All customers will have the 
                                 // same value of p. 
    vector<lower=0> [n_cust] v;   // shape parameter of the gamma distribution (nu)
    real <lower=0>q;             // shape parameter of the gamma prior distribtion on v 
    real <lower=0>y;             // scale parameter of the gamma prior distribution on v 
}

transformed parameters {
    vector<lower=0> [n_cust] px;
    vector<lower=0> [n_cust] nx; 
    px <- p * x;                 // getting px from p and x 
    for (i in 1:n_cust) 
        nx[i] <- v[i] * x[i]; 
}

model {
    p ~ exponential(0.1);    // prior distribution on p
    q ~ exponential(0.5);    // hyperprior distribution on q 
    y ~ exponential(0.1);    // hyperprior distribution on y 
//    v ~ gamma(q, q ./ y);    // prior distribution on nu  
//    mx ~ gamma(px, v);       // likelihood function 
    v ~ gamma(q,y); 
    mx ~ gamma(px,nx); 
}
"""

# here's the data we will provide to STAN :
data_gg={'n_cust':len(rfm), \
    'x':rfm['frequency'].values+1.0, \
    'mx':rfm['monetary_value'].values \
     }

# I recommend training for several 1000's iterations.
gamma_gamma_fit = stan_cache(gamma_gamma_model, model_name='gamma_gamma_model', \
                                  data=data_gg, chains=1, iter=5000, warmup=500)

trace_gg = gamma_gamma_fit.extract()
nu = trace_gg['v']
p = trace_gg['p']
gamma = trace_gg['y']
q = trace_gg['q']

pvalues = np.tile(np.array(p).T,(len(rfm),1))
E_M = (pvalues / nu.T).mean(axis=1)
rfm['E_M'] = E_M
rfm[['monetary_value', 'E_M']].head()


plot_scatter(rfm,'monetary_value','E_M',
             xlabel='Average Order Value in Training Period ($)',
             ylabel='E(M) ($)',
             xlim=[0,50], ylim=[0,50])

print("Gamma-Gamma model on test set for average order value")

holdout_value = holdout.groupby(['userId', 'week'], as_index=False)['Amount'].sum().reset_index()
holdout_value = holdout_value[['userId', 'Amount']].groupby(['userId'])['Amount'].mean()
holdout_value=pd.DataFrame(holdout_value)
holdout_value.rename(columns={'Amount':'obs_holdout_monetary_value'}, inplace=True)
holdout_value.head()

rfm_w_holdout_value  = rfm.merge(holdout_value, how='left', left_index=True, right_index=True)
rfm_w_holdout_value.fillna(0,inplace=True)


plot_scatter(rfm_w_holdout_value,'obs_holdout_monetary_value','E_M',\
             xlabel='Average Order Value in holdout Period ($)',\
             ylabel='E(M) ($)',\
             xlim=[0,200], ylim=[0,200])

rfm['E_M'].mean()
holdout_value.mean()


print("CLV comparison")


# compute both modeled and observed CLV in the holdout period :

rfm['model_holdout_clv'] = rfm_with_holdout['model_holdout_count'] * rfm['E_M']
rfm['obs_holdout_clv'] = rfm_with_holdout['obs_holdout_count'] * rfm_w_holdout_value['obs_holdout_monetary_value']
rmse_holdout_clv = (rfm['model_holdout_clv'] - rfm['obs_holdout_clv'])* \
                   (rfm['model_holdout_clv'] - rfm['obs_holdout_clv'])
rmse_holdout_clv = np.sqrt(rmse_holdout_clv.sum()/len(rfm))


# plot the final results :
print('RMSE =', rmse_holdout_clv)
plot_scatter(rfm, 'obs_holdout_clv', 'model_holdout_clv',
             xlabel='Observed Value in the Holdout Period',
             ylabel='Modeled Value in the Holdout Period',
             xlim=[0,100.0],ylim=[0,100.0])