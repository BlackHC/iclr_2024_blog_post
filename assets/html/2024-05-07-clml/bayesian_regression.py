#%%
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.base import clone
from tqdm.auto import tqdm
import scipy.stats as stats

#%%
# Generate synthetic data
np.random.seed(0)
n_samples, n_features = 500, 64
X = np.random.rand(n_samples*2, n_features)
# True coefficients
w = np.random.randn(n_features) + 2
# Targets with noise
y = X.dot(w)
# Split into train and test
X_train, X_test = X[:n_samples], X[n_samples:]
y_train, y_test = y[:n_samples], y[n_samples:]

# Different hyperparameters for the prior precision (lambda)
experiment_losses = {}


#%%

class BayesianLinearModel:
    def __init__(self, phi: float=1.0, num_features=2, sigma_noise=0.1):
        self.phi = phi
        self.num_features = num_features
        self.sigma_noise = sigma_noise
        self.mean_prior = np.zeros((num_features,))
        self.precision_prior = np.eye(num_features) / self.phi
    
    def fit(self, X, y, evidence_fraction=1.0):
        """Fit the model to the data (X, y).
        
        Parameters:
        X: np.ndarray (n_samples, n_features)
            The input features
        y: np.ndarray (n_samples,)
            The target values
        evidence_fraction: float
            The fraction of the data to use for fitting the model
        """
        precision_likelihood = evidence_fraction * X.T @ X / self.sigma_noise**2
        precision_posterior = self.precision_prior + precision_likelihood
        
        mean_posterior = np.linalg.solve(precision_posterior, (self.precision_prior @ self.mean_prior + evidence_fraction * X.T @ y / self.sigma_noise**2))
        
        self.mean_prior = mean_posterior
        self.precision_prior = precision_posterior
    
    def predictive_distribution(self, X_new):
        mean_pred = X_new @ self.mean_prior
        cov_pred = X_new @ np.linalg.solve(self.precision_prior, X_new.T) + self.sigma_noise**2
        return mean_pred, cov_pred
    
    def marginal_cross_entropy(self, X, y):
        """Compute the marginal cross-entropy for the data points (X, y).
        
        The marginal information is the negative log likelihood of the data points given the model.
        
        Parameters:
        X: np.ndarray (n_samples, n_features)
            The input features
        y: np.ndarray
            The target values

        Returns:
        np.ndarray
            The marginal information for each data point
        """
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) > 0, "X and y must not be empty"
        mean_pred, cov_pred = self.predictive_distribution(X)
        # Diagonalize the covariance matrix, so we have uncorrelated predictions
        cov_pred = stats.Covariance.from_diagonal(np.diag(cov_pred))
        # The loss is the negative log likelihood (cross-entropy) for the next data point
        marginal_information = -stats.multivariate_normal.logpdf(y, mean=mean_pred, cov=cov_pred, allow_singular=False)
        return marginal_information / len(X)
    
    def joint_marginal_information(self, X, y, mean=True):
        """Compute the joint information (negative log marginal likelihood) of the data given the model.
        
        The log marginal likelihood is the log probability of the data given the model, 
        marginalized over all possible model parameters.
        
        Parameters:
        X: np.ndarray (n_samples, n_features) 
            The input features
        y: np.ndarray (n_samples,)
            The target values
            
        Returns:
        float
            The log marginal likelihood
        """
        # Compute the precision matrix of the posterior distribution
        precision_posterior = self.precision_prior + X.T @ X / self.sigma_noise**2
        
        # Compute the mean of the posterior distribution
        mean_posterior = np.linalg.solve(precision_posterior, self.precision_prior @ self.mean_prior + X.T @ y / self.sigma_noise**2)
        
        # Compute the log marginal likelihood
        log_marginal_likelihood = (
            -0.5 * len(X) * np.log(2 * np.pi * self.sigma_noise**2)
            -0.5 * np.sum((y - X @ mean_posterior)**2) / self.sigma_noise**2
            -0.5 * (mean_posterior - self.mean_prior) @ self.precision_prior @ (mean_posterior - self.mean_prior)
            +0.5 * np.log(np.linalg.det(self.precision_prior))
            -0.5 * np.log(np.linalg.det(precision_posterior))
        )
        
        jmi = -log_marginal_likelihood
        
        if mean:
            return jmi / len(X)
        else:
            return jmi
    
    def __str__(self):
        return f"BayesianLinearModel(phi={self.phi}, sigma_noise={self.sigma_noise})"
    
    
#%% Test the BayesianLinearModel

def test_bayesian_linear_model():
    # Create a simple dataset
    X = np.random.rand(5, 2)
    w = np.random.randn(2)

    y = X @ w + np.random.randn(5) * 0

    # Create the BayesianLinearModel
    model = BayesianLinearModel(sigma_noise=0.0001)

    # Fit the model
    model.fit(X, y)

    # Predict the marginal information
    marginal_information = model.marginal_cross_entropy(X, y)

    print(marginal_information)

    # Create a grid of test points
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.c_[X1.ravel(), X2.ravel()]

    # Compute the predictive distribution
    mean_pred, cov_pred = model.predictive_distribution(X_test)
    mean_pred = mean_pred.reshape(X1.shape)
    cov_pred = np.diag(cov_pred).reshape(X1.shape)

    # Plot the predictive distribution
    plt.figure(figsize=(12, 6))
    plt.contourf(X1, X2, mean_pred, levels=20, cmap='coolwarm')
    plt.colorbar()
    plt.title('Predictive Distribution Mean')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.contourf(X1, X2, cov_pred, levels=20, cmap='coolwarm')
    plt.colorbar()
    plt.title('Predictive Distribution Covariance')
    plt.show()
    
    
test_bayesian_linear_model()

#%%
# Split the dataset into training sets of varying sizes
sizes = np.arange(1, n_samples).astype(int)

print(sizes)
#%%
from copy import deepcopy

# experiment_base = NoisyPredictor(Ridge(alpha=0.1), noise_std=0.1)
experiment_losses = {}
experiment_bases = [
    BayesianLinearModel(sigma_noise=0.8, phi=0.1, num_features=n_features),
    BayesianLinearModel(sigma_noise=1.0, phi=100, num_features=n_features),
    BayesianLinearModel(sigma_noise=1.2, phi=1, num_features=n_features)
]

num_trials = 5

#%%
for experiment_base in experiment_bases:
    # Check if the experiment base is already exists in the experiment_losses
    if str(experiment_base) in experiment_losses:
        continue
    
    metrics = {}
    
    metrics["iterative_train_loss"] = {}
    metrics["iterative_val_loss"] = {}
    for size in tqdm(sizes):
        # Copy the experiment base
        reg = deepcopy(experiment_base)
        
        reg.fit(X_train, y_train, size/n_samples)
        
        # Predict on the training set
        train_loss = reg.marginal_cross_entropy(X_train, y_train)
        val_loss = reg.marginal_cross_entropy(X_test, y_test)
        
        metrics["iterative_train_loss"][size] = train_loss
        metrics["iterative_val_loss"][size] = val_loss
        
    metrics["joint_marginal_information"] = {}
    metrics["conditional_joint_marginal_information_half"] = {}
    metrics["marginal_cross_entropy_train"] = {}
    metrics["marginal_cross_entropy_val"] = {}
        
    for trial in tqdm(range(num_trials)):
        seed = trial + 31
        np.random.seed(seed)
        # permute the training data
        perm = np.random.permutation(n_samples)
        # X_train_perm = np.concatenate((X_train[perm], X_test))
        # y_train_perm = np.concatenate((y_train[perm], y_test))
        X_train_perm = X_train[perm]
        y_train_perm = y_train[perm]
        
        metrics["joint_marginal_information"][trial] = {}
        metrics["conditional_joint_marginal_information_half"][trial] = {}
        metrics["marginal_cross_entropy_train"][trial] = {}
        metrics["marginal_cross_entropy_val"][trial] = {}
        
        for size in sizes:
            # Copy the experiment base
            reg = deepcopy(experiment_base)
        
            jmi = reg.joint_marginal_information(X_train_perm[:size], y_train_perm[:size], mean=False)
    
            # Train on the current subset of data
            reg.fit(X_train_perm[:size], y_train_perm[:size])
            
            if size < n_samples - 1:
                mce_train = reg.marginal_cross_entropy(X_train_perm[size:], y_train_perm[size:])
            mce_val = reg.marginal_cross_entropy(X_test, y_test)
            
            metrics["joint_marginal_information"][trial][size] = jmi
            metrics["marginal_cross_entropy_train"][trial][size] = mce_train
            metrics["marginal_cross_entropy_val"][trial][size] = mce_val
          
    experiment_losses[str(experiment_base)] = metrics
    
    
#%% Save the experiment losses to a file
import pickle

with open('bayesian_regression_results.pkl', 'wb') as f:
    pickle.dump(experiment_losses, f)
    
#%% Load the experiment losses from a file
import pickle

with open('bayesian_regression_results.pkl', 'rb') as f:
    experiment_losses = pickle.load(f)
    
#%%

for experiment_base in experiment_bases:
    experiment_losses[str(experiment_base)]['conditional_joint_marginal_information_half'] = {}
    for trial in tqdm(range(num_trials)):
        experiment_losses[str(experiment_base)]['conditional_joint_marginal_information_half'][trial] = {}
        for size in sizes:
            if size <= 1:
                continue
            
            experiment_losses[str(experiment_base)]['conditional_joint_marginal_information_half'][trial][size] = experiment_losses[str(experiment_base)]['joint_marginal_information'][trial][size] - experiment_losses[str(experiment_base)]['joint_marginal_information'][trial][size//2]
    
#%% Add joint_marginal_information_rate

for experiment_base in experiment_bases:
    experiment_losses[str(experiment_base)]['joint_marginal_information_rate'] = {}
    for trial in tqdm(range(num_trials)):
        experiment_losses[str(experiment_base)]['joint_marginal_information_rate'][trial] = {}
        for size in sizes:
            if size == 0:
                continue
            
            experiment_losses[str(experiment_base)]['joint_marginal_information_rate'][trial][size] = experiment_losses[str(experiment_base)]['joint_marginal_information'][trial][size] / size
    
    
#%% Turn the experiment losses into a DataFrame
import pandas as pd

df = pd.DataFrame(experiment_losses)
print(df)

"""
                                            BayesianLinearModel(phi=0.1, sigma_noise=0.8)  \
iterative_train_loss                    [116.3599064765003, 3.4268582439928608, 2.3751...   
iterative_val_loss                      [113.13281304925161, 3.194934292949091, 2.3217...   
joint_marginal_information              [[791.7223107868632, 67.8785017447558, 36.3604...   
conditional_joint_marginal_information  [[1.732014085114092, 1.1200774720898565, 1.038...   
marginal_cross_entropy_train            [[213.73700961662368, 5.569810576971788, 3.067...   
marginal_cross_entropy_val              [[210.73875030996902, 5.30781256492312, 2.9797...   

                                            BayesianLinearModel(phi=100, sigma_noise=1.0)  \
iterative_train_loss                    [2.9429216455692577, 1.6137002302712895, 1.379...   
iterative_val_loss                      [2.992731299595586, 1.6580392937367738, 1.4156...   
joint_marginal_information              [[5.7797700416063265, 4.132155782355027, 3.927...   
conditional_joint_marginal_information  [[1.2066774705261731, 1.1485958035421844, 1.09...   
marginal_cross_entropy_train            [[4.436575702635916, 3.896238906055772, 3.5235...   
marginal_cross_entropy_val              [[4.426701233040085, 3.894546917207375, 3.5212...   

                                              BayesianLinearModel(phi=1, sigma_noise=1.2)  
iterative_train_loss                    [4.3029531605782125, 1.7546186158615935, 1.550...  
iterative_val_loss                      [4.086622096954006, 1.772978360559056, 1.57418...  
joint_marginal_information              [[98.66537505168095, 8.933056270580206, 5.6024...  
conditional_joint_marginal_information  [[1.3120187036274622, 1.2479855753762326, 1.22...  
marginal_cross_entropy_train            [[19.046874252757462, 2.27570586288723, 1.9003...  
marginal_cross_entropy_val              [[18.674391898852846, 2.2574886323313277, 1.89...  
"""
#%%
from blackhc.project.utils import tree_namespace


def _join_keys(*keys: str) -> str:
    """Join keys with / and remove any leading or trailing /."""
    return "/".join(key for key in keys if key is not None).strip("/")


def all_items(obj: object, prefix: str = None) -> list[tuple[str, object]]:
    """Get all items of an object.

    For lists, tuples, and dicts, this returns the items of the elements.
    """
    if isinstance(obj, tree_namespace.TreeNamespace):
        yield from obj._items()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from all_items(value, _join_keys(prefix, str(key)))
    elif isinstance(obj, (list, tuple)):
        for key, value in enumerate(obj):
            yield from all_items(value, _join_keys(prefix, str(key)))
    else:
        yield (prefix, obj)


#%%
# Create a dataframe such that each row is a different row in the metrics and each column is the metric, experiment base and trial number (if applicable)
# Convert the experiment_losses dict into a flat list of tuples
flat_metrics = list(all_items(experiment_losses))

# Create a DataFrame from the flat list
df = pd.DataFrame(flat_metrics, columns=['metric', 'value'])

# Split the metric column into separate columns
df[['experiment_base', 'metric']] = df['metric'].str.split('/', n=1, expand=True)

# Split the metric column into separate columns
df[['metric', 'trial', 'size']] = df['metric'].str.rsplit('/', n=2, expand=True)

# If trial is NaN, then there was no trial, so move row to the correct column
mask = df['size'].isna()
df.loc[mask, 'size'] = df.loc[mask, 'trial'] 
df.loc[mask, 'trial'] = 0

# Fill remaining NaN values in the trial column with 0
df['trial'] = df['trial'].fillna(0)

# Convert the trial column to numeric
df['trial'] = pd.to_numeric(df['trial'], errors='coerce')

# Convert the size column to numeric
df['size'] = pd.to_numeric(df['size'], errors='coerce')

# Reorder the columns
df = df[['experiment_base', 'trial', 'size', 'metric', 'value']]

#%%
# Step 1: Filter the DataFrame for joint_marginal_information
jmi_df = df[df['metric'] == 'joint_marginal_information'].copy()

# Step 2: Pivot the DataFrame
pivot_df = jmi_df.pivot_table(index=['trial', 'size'], columns='experiment_base', values='value')

# Step 3: Convert to numpy array
jmi_trials = pivot_df.to_numpy()
jmi_trials.shape
#%%
jmi_trials = jmi_trials.reshape(num_trials, len(sizes), len(experiment_bases))
jmi_array = jmi_trials.mean(axis=0)
#%%
#%% 
# Step 4: Take differences with different periods
result_array = np.zeros((len(sizes), len(sizes), len(experiment_bases)))
for base in range(len(experiment_bases)):
    for T in range(1, len(sizes)):
        for t in range(len(sizes)):
            if t < T:
                result_array[t,T, base] = np.nan
            else: 
                result_array[t,T, base] = jmi_array[t,base] - jmi_array[t-T, base]

#%%
# Step 5: Pick the argmin for each base
# result_winner.fill(np.nan)
result_winner = np.argmin(result_array, axis=-1) + 0.0
result_winner[np.isnan(result_array[...,0])] = np.nan
result_winner
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

# Set white theme in seaborn
plt.style.use('seaborn-v0_8-whitegrid')
plt.style.use('tableau-colorblind10')

fig, ax = plt.subplots(figsize=(6/1.6, 6/1.6))
cmap = plt.cm.get_cmap('tab10', 10)
cmap.set_bad('none')


bounds = np.array([-0.5, 0.5, 1.5, 2.5])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=3)
im = ax.matshow(result_winner[:,:].T, cmap=cmap, norm=norm, origin='lower')

# Draw a 0.5 white line with slop 0.5
# This line shows where conditional_joint_marginal_information_half lies
ax.plot([0, len(sizes)], [0, len(sizes)//2], color='white', lw=1, alpha=0.5)

# Add a legend with the 3 experiments
# C0 is the first experiment, C1 is the second experiment, C2 is the third experiment
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap.colors[0], label=r'$\phi_1$'),
           plt.Rectangle((0, 0), 1, 1, color=cmap.colors[1], label=r'$\phi_2$'),
           plt.Rectangle((0, 0), 1, 1, color=cmap.colors[2], label=r'$\phi_3$')]
ax.legend(handles=handles, title=r'Model $\Phi$', loc='upper left')

# Move the ticks to the bottom
ax.xaxis.set_ticks_position('bottom')

ax.set_xlabel('Dataset Size')
ax.set_ylabel('Held-Back Size')

# Turn off the frame and add a color bar
ax.spines[:].set_visible(False)

# Save as svg
plt.savefig('binary_regression_conditional_joint_marginal_information_decision_boundary.svg', format='svg', dpi=300)
plt.savefig('binary_regression_conditional_joint_marginal_information_decision_boundary.png', format='png', dpi=300)

plt.show()

#%% Plot the metrics

plt.figure(figsize=(12, 12/1.6))

sns.set_context("paper")
sns.set_style("whitegrid")
# sns.set_color_codes("colorblind")
sns.set_palette("tab10")

sub_df = df
# sub_df = sub_df.sample(2000)
sub_df = sub_df[sub_df['metric'].isin(['joint_marginal_information', 'marginal_cross_entropy_train', 'marginal_cross_entropy_val', 'iterative_train_loss', 'conditional_joint_marginal_information_half', 'joint_marginal_information_rate'])]
sub_df = sub_df.copy()
# Map experiment base to the previous legend
experiment_base_map = {
    'BayesianLinearModel(phi=0.1, sigma_noise=0.8)': r'$\phi_1$',
    'BayesianLinearModel(phi=100, sigma_noise=1.0)': r'$\phi_2$',
    'BayesianLinearModel(phi=1, sigma_noise=1.2)': r'$\phi_3$'
}
legend_title = r'Model'
sub_df[legend_title] = sub_df['experiment_base'].map(experiment_base_map)

# Map the metric to readable names
metric_map = {
    'marginal_cross_entropy_val': 'Marginal Cross-Entropy (Validation)',
    'marginal_cross_entropy_train': 'Marginal Cross-Entropy (Train)',
    'iterative_train_loss': 'Training Speed (Approx)',
    'joint_marginal_information': 'Joint Marginal Information',
    'joint_marginal_information_rate': 'Joint Marginal Information Rate',
    'conditional_joint_marginal_information_half': 'Conditional Joint Marginal Information (Half)',
}
sub_df['metric'] = sub_df['metric'].map(metric_map)

sub_df['value'] = sub_df['value'] / np.log(2)

# Plot the metrics
g = sns.FacetGrid(sub_df, col="metric", col_wrap=3, margin_titles=True, sharey=False, col_order=metric_map.values())
g.map(sns.lineplot, "size", "value", legend_title, hue_order=experiment_base_map.values())
# set log on y axis
g.set(yscale="log")
unit_tex = r"$\mathrm{ / bits}$" 
down_arrow = r"$\downarrow$"
g.set_axis_labels("Dataset Size", f"Information {unit_tex} {down_arrow}")
g.add_legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
g.set_titles("{col_name}")
# Set constraint engine to avoid overlapping titles
g.fig.tight_layout()
# Set overall title for the plot
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Information Metrics for Different Bayesian Linear Models')

# Save as svg
plt.savefig('binary_regression_information_metrics.svg', format='svg', dpi=300)
plt.savefig('binary_regression_information_metrics.png', format='png', dpi=300)
plt.show()

# %%
"""
Figure 1: Decision boundary for the model with the lowest conditional joint marginal information, as a function of dataset size and held-back size. The three models $\phi_1$, $\phi_2$, and $\phi_3$ correspond to different prior precisions in the Bayesian linear regression. The white diagonal line shows where the conditional joint marginal information is computed using half the dataset size. In the region below this line, $\phi_1$ (blue) has the lowest conditional joint marginal information, while above the line $\phi_2$ (orange) and $\phi_3$ (green) are preferred for different dataset and held-back sizes.
Figure 2: Information metrics for the three Bayesian linear regression models as a function of dataset size. The joint marginal information and conditional joint marginal information (with half the dataset size) decrease with more data, showing the models are learning. The joint marginal information rate normalizes for dataset size. The marginal cross-entropy on the train and validation sets also decreases with more data, indicating improved generalization. Model $\phi_1$ (blue) has the highest information metrics, while models $\phi_2$ (orange) and $\phi_3$ (green) are fairly similar. All metrics are reported in bits (log base 2).
"""