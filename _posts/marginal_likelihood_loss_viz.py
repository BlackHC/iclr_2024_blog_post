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
    return np.exp(-factor * N / 10000) * (initial_loss - best_loss) + best_loss

# Setup dataset sizes to sample from and x to plot against.
x = np.linspace(0, 1, 1000)
# We want N to run from 1 to infinity for the range of x values
N = (1 / (1 - x) - 1) * 10000
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
            return f"{int(N[int(tick_idx)])}"

        
new_tick_labels = [get_tick_label(tick_idx) for tick_idx in current_ticks]
prior_data_conflict_ax.set_xticklabels(new_tick_labels)
prior_data_conflict_ax.set_title("Prior-Data Conflict: Similar Performance in ∞ Sample Limit") 

# Right plot: misspecified models plot

for i, best_model_accuracy in zip(range(1, 4), [0.94, 0.86, 0.89]):
    best_model_loss = get_multiclass_ce(num_classes, best_model_accuracy)
    loss = simulate_loss(i, N, initial_loss, best_model_loss)
    model_misspecification_ax.plot(idx, loss, zorder=4-i, label=rf"$\phi_{i}$")
model_misspecification_ax.legend()

new_tick_labels = [get_tick_label(tick_idx) for tick_idx in current_ticks]
model_misspecification_ax.set_xticklabels(new_tick_labels)
model_misspecification_ax.set_title("Model Misspecification: Different Performance for ∞ Sample Limit") 


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
clml_tex = r"\mathrm{p}(x_N \mid x_{N-1}, ..., x_1, \phi)"
lml_tex = r"\mathrm{p}(x_N, x_{N-1}, ..., x_1, \phi)"
cmce_tex = r"\operatorname{H}({\hat{\mathrm{p}} \Vert \mathrm{p}(\circ \mid \phi)})[X_N \mid X_{N-1}, ..., X_1]"

# Left plot: joint cross-entropy as area under marginal cross-entropy
phi_accuracy = 0.92
phi_loss = get_multiclass_ce(num_classes, phi_accuracy)
loss = simulate_loss(2, N, initial_loss, phi_loss)
mce_line = marginal_ce_ax.plot(idx, loss, zorder=4-i, color="C0", label="Marginal Cross-Entropy")
# marginal_ce_ax.text(20, loss[20], f"$\phi$", verticalalignment='bottom', horizontalalignment='left', c=mce_line[0].get_color())
xy = (20, loss[20]+0.1)
xytext = (30, loss[20] + 0.1)
arrowprops = dict(facecolor=mce_line[0].get_color(), shrink=0.05)
marginal_ce_ax.annotate(f"${cmce_tex}$", zorder=10, xy=xy, xytext=xytext, arrowprops=arrowprops, verticalalignment='center', horizontalalignment='left', color=mce_line[0].get_color())

marginal_ce_ax.fill_between(idx, np.zeros_like(loss), loss, alpha=0.2, label=f"Joint Cross-Entropy", color="C1")
marginal_ce_ax.text(50, phi_loss/2, f"${jce_tex}$", verticalalignment='center', horizontalalignment='center', color="C1")

marginal_ce_ax.set_xlabel('Dataset size N')
marginal_ce_ax.set_ylabel(f'$1 {unit_tex}$')
marginal_ce_ax.legend()
marginal_ce_ax.set_title("Joint Cross-Entropy as Area under Marginal Cross-Entropy")

marginal_ce_ax.set_xticklabels(new_tick_labels)
marginal_ce_ax.set_title("Joint Cross-Entropy as Area under Marginal Cross-Entropy") 

# Right plot: Marginal likelihood as area under conditional training loss
# Here we need to add some noise to the loss because we look at individual samples

diff_N = np.diff(N, prepend=N[0])
print(diff_N)
scaled_noise = np.random.gumbel(0, 0.1 / diff_N**0.5, len(loss))
noised_loss = loss - scaled_noise

marginal_likelihoood_ax.plot(idx, noised_loss, zorder=4-i, color="C2", label="Conditional Marginal Likelihood")
xy = (60, noised_loss[60]+0.1)
xytext = (30, noised_loss[20] + 0.1)
arrowprops = dict(facecolor='C0', shrink=0.05)
marginal_likelihoood_ax.annotate(f"${clml_tex}$", zorder=10, xy=xy, xytext=xytext, arrowprops=arrowprops, verticalalignment='center', horizontalalignment='left', color="C2")

marginal_likelihoood_ax.fill_between(idx, np.zeros_like(noised_loss), noised_loss, alpha=0.2, label=f"Marginal Likelihood Loss", color="C3")
marginal_likelihoood_ax.text(50, phi_loss/2, f"${lml_tex}$", verticalalignment='center', horizontalalignment='center', color="C3")

marginal_likelihoood_ax.set_xlabel('Dataset size N')
marginal_likelihoood_ax.set_ylabel(f'$1 {unit_tex}$')
marginal_likelihoood_ax.legend()
marginal_likelihoood_ax.set_title("Marginal Likelihood as Area under Conditional Training Loss")

marginal_likelihoood_ax.set_xticklabels(new_tick_labels)
marginal_likelihoood_ax.set_title("Effect of Batch Size on Training Loss") 

# Simulate a larger batch size. 
batch_size = 512
batch_N = np.arange(0, 100000, batch_size)
# 

plt.tight_layout()
plt.show()

# %%
