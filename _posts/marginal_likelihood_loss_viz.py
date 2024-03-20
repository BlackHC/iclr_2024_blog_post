#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial

plt.xkcd()

# Define the number of subplots
n_rows, n_cols = 1, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15/1.75/1.6), squeeze=False)

prior_data_conflict_ax = axes[0,0]
model_misspecification_ax = axes[0,1]

# cmce_tex = r"\operatorname{H}_{\hat{\mathrm{p}} \Vert \mathrm{p}(\cdot \mid \Phi)}[X_N | X_{N-1}, ..., X_1]"
cmce_tex = r"\operatorname{H}({\hat{\mathrm{p}} \Vert \mathrm{p}(\circ \mid \phi_\circ)})[X_N | X_{N-1}, ..., X_1]"
unit_tex = r"\mathrm{ / bits}" 
cmce_title = "Multi-Class Classification: Num Samples vs Conditional Marginal CE (Cross-Validation NLL)"


# Simulate the ce for multi-class classification
nats_to_bits = 1 / np.log(2)
def get_multiclass_ce(num_classes, best_accuracy: float | None =None):
    assert num_classes > 1
    if best_accuracy is None:
        best_accuracy = 1/num_classes
    residual = 1 - best_accuracy
    p = np.array([residual/(num_classes-1)]*(num_classes-1) + [best_accuracy])
    return multinomial.entropy(1, p) * nats_to_bits

def simulate_loss(factor, N, initial_loss, best_loss):
    return np.exp(-factor * N) * (initial_loss - best_loss) + best_loss

# Setup dataset sizes to sample from and x to plot against.
x = np.linspace(0, 1, 100)
# We want N to run from 1 to infinity for the range of x values
N = 1 / (1 - x) - 1
idx = list(range(len(x)))

# Left plot: same loss in infinite sample limit
num_classes = 10
best_loss = get_multiclass_ce(num_classes, 0.96)
initial_loss = get_multiclass_ce(num_classes)

for i in range(1, 4):
    loss = simulate_loss(i, N, initial_loss, best_loss)
    prior_data_conflict_ax.plot(idx, loss, zorder=4-i, label=rf"$\phi_{i}$")

prior_data_conflict_ax.legend()
prior_data_conflict_ax.set_xlabel('Dataset size N')
prior_data_conflict_ax.set_ylabel(f'${cmce_tex} {unit_tex}$')

# Get current ticks
current_ticks = prior_data_conflict_ax.get_xticks()

def get_tick_label(tick_idx):
    match tick_idx:
        case tick_idx if tick_idx < 0:
            return " "
        case tick_idx if tick_idx >= len(idx):
            return "∞"
        case _:
            return f"{int(N[int(tick_idx)]*10000)}"

        
new_tick_labels = [get_tick_label(tick_idx) for tick_idx in current_ticks]
prior_data_conflict_ax.set_xticklabels(new_tick_labels)
prior_data_conflict_ax.set_title("Prior-Data Conflict: Same/Similar Loss for ∞ Sample Limit") 

# Right plot: misspecified models plot

for i, best_model_accuracy in zip(range(1, 4), [0.94, 0.86, 0.89]):
    best_model_loss = get_multiclass_ce(num_classes, best_model_accuracy)
    loss = simulate_loss(i, N, initial_loss, best_model_loss)
    model_misspecification_ax.plot(idx, loss, zorder=4-i, label=rf"$\phi_{i}$")
model_misspecification_ax.legend()

new_tick_labels = [get_tick_label(tick_idx) for tick_idx in current_ticks]
model_misspecification_ax.set_xticklabels(new_tick_labels)
model_misspecification_ax.set_title("Model Misspecification: Different Loss for ∞ Sample Limit") 


plt.suptitle(cmce_title)
# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#%%
## Create another figure with two plots that visualize:
# - how the area under the marginal cross-entropy is equal the joint cross-entropy using the chain rule
# - how the area under the conditional training loss (NLL) is equal the marginal likelihood loss using the chain rule
# (when training with batch size 1)
# In the second plot, we also show how batch size >> 1 leads to a staircase of upper-bounds

# Define the number of subplots
n_rows, n_cols = 1, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15/1.75/1.6), squeeze=False)

marginal_ce_ax = axes[0,0]
marginal_likelihoood_ax = axes[0,1]

jce_tex = r"\operatorname{H}(\hat{\mathrm{p}} \Vert \mathrm{p}(\cdot \mid \phi))[X_N, X_{N-1}, ..., X_1]"
clml_tex = r"\mathrm{p}(\circ \mid \phi_\circ))[x_N | x_{N-1}, ..., x_1]"
lml_tex = r"\mathrm{p}(\circ \mid \phi_\circ))[x_N, x_{N-1}, ..., x_1]"

# Left plot: joint cross-entropy as area under marginal cross-entropy
phi_accuracy = 0.92
phi_loss = get_multiclass_ce(num_classes, phi_accuracy)
loss = simulate_loss(2, N, initial_loss, phi_loss)
mce_line = marginal_ce_ax.plot(idx, loss, zorder=4-i, label=rf"$\phi_{i}$")
cmce_tex = r"\operatorname{H}({\hat{\mathrm{p}} \Vert \mathrm{p}(\circ \mid \phi)})[X_N | X_{N-1}, ..., X_1]"
marginal_ce_ax.text(20, loss[20], f"${cmce_tex}$", verticalalignment='bottom', horizontalalignment='left', c=mce_line[0].get_color())
marginal_ce_ax.fill_between(idx, np.zeros_like(loss), loss, alpha=0.2, label=f"Joint Cross-Entropy")
marginal_ce_ax.text(50, phi_loss/2, f"${jce_tex} {unit_tex}$", verticalalignment='center', horizontalalignment='center')

marginal_ce_ax.legend()
marginal_ce_ax.set_title("Joint Cross-Entropy as Area under Marginal Cross-Entropy")

plt.tight_layout()
plt.show()

# %%
