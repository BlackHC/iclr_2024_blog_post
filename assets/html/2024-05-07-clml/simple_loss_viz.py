# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial

plt.xkcd()

# Define the number of subplots
n_rows, n_cols = 1, 2
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(15, 15 / 2.25 / 1.6 * 1.2), squeeze=False
)

prior_data_conflict_ax = axes[0, 1]
model_misspecification_ax = axes[0, 0]

y_label = "Performance (Validation Loss)"
x_label = "Num Samples/Training Steps (One-Epoch)"
model_template = r"$\mathcal{{M}}_{i}$"

# Simulate the ce for multi-class classification
nats_to_bits = 1 / np.log(2)


def get_multiclass_ce(num_classes, best_accuracy: float | None = None):
    assert num_classes > 1
    if best_accuracy is None:
        best_accuracy = 1 / num_classes
    residual = 1 - best_accuracy
    p = np.array([residual / (num_classes - 1)] * (num_classes - 1) + [best_accuracy])
    return multinomial.entropy(1, p) * nats_to_bits


def simulate_loss(factor, N, initial_loss, best_loss, N_factor: float):
    return (
        np.power(0.01, factor * N / N_factor) * (initial_loss - best_loss) + best_loss
    )


def get_N_x_mapping(x_power: float, N_factor: float):
    def N_to_x(N):
        return 1 - 1 / (1 + N / N_factor) ** x_power

    def x_to_N(x):
        return (1 / (1 - x) ** (1 / x_power) - 1) * N_factor

    def test_N_to_x_x_to_N():
        # Setup dataset sizes to sample from and x to plot against.
        Ns = np.linspace(0, 1, 1000)
        Xs = N_to_x(Ns)
        assert np.allclose(x_to_N(Xs), Ns)
        assert np.allclose(N_to_x(Ns), Xs)

    test_N_to_x_x_to_N()

    return N_to_x, x_to_N


N_factor = 50000
x_power = 2 / 3

N_to_x, x_to_N = get_N_x_mapping(x_power=x_power, N_factor=N_factor)

Ns = np.arange(0, 48000, 4)
Xs = N_to_x(Ns)
# Concat a linrange from max(x) to 1
Xs = np.concatenate([Xs, np.linspace(1, Xs[-1], 1000, endpoint=False)[::-1]])
Ns = x_to_N(Xs)


def find_idx_by_N(Ns, n):
    # Find n in N via binary search
    idx = np.searchsorted(Ns, n)
    return idx


def find_idx_by_x(Xs, x):
    # Find n in N via binary search
    idx = np.searchsorted(Xs, x)
    return idx


def get_tick_label(x):
    match x:
        case x if x < 0:
            return " "
        case x if x >= 1:
            return r"$\infty$"
        case _:
            return f"{int(x_to_N(x))}"


num_classes = 10
best_loss = get_multiclass_ce(num_classes, 0.96)
initial_loss = get_multiclass_ce(num_classes)

# Left plot: misspecified models plot
for i, (loss_factor, best_model_accuracy) in enumerate(
    zip([1, 1, 1], [0.80, 0.89, 0.94])
):
    best_model_loss = get_multiclass_ce(num_classes, best_model_accuracy)
    loss = simulate_loss(loss_factor, Ns, initial_loss, best_model_loss, N_factor)
    model_misspecification_ax.plot(Xs, loss, zorder=4 - i, label=model_template.format(i=i))
model_misspecification_ax.legend()

# Get current ticks
current_ticks = prior_data_conflict_ax.get_xticks()
new_tick_labels = [get_tick_label(x) for x in current_ticks]
model_misspecification_ax.set_xticks(current_ticks)
model_misspecification_ax.set_xticklabels(new_tick_labels)
model_misspecification_ax.get_xticklabels()[-1].set_fontsize(18)

model_misspecification_ax.set_title("Model Misspecification")
model_misspecification_ax.set_xlabel(x_label)
model_misspecification_ax.set_ylabel(y_label)

# Right plot: same loss in infinite sample limit
for i, loss_factor in enumerate([0.75, 1.25, 2.0]):
    loss = simulate_loss(loss_factor, Ns, initial_loss, best_loss, N_factor)
    prior_data_conflict_ax.plot(Xs, loss, zorder=4 - i, label=model_template.format(i=i))

prior_data_conflict_ax.legend()
prior_data_conflict_ax.set_xlabel(x_label)
prior_data_conflict_ax.set_xticks(current_ticks)
prior_data_conflict_ax.set_xticklabels(new_tick_labels)
prior_data_conflict_ax.get_xticklabels()[-1].set_fontsize(18)

prior_data_conflict_ax.set_title("Prior-Data Conflict")

# Add "blackhc@" text to the right corner
fig.text(1.0, 0.05, 'blackhc@', fontsize=10, color='gray', 
         ha='right', va='bottom', alpha=0.5, transform=fig.transFigure)

# Adjust layout and display the plot
plt.tight_layout()

# Save as SVG
plt.savefig(f"simple_prior_conflict_and_model_misspecification_{x_power:0.2f}.svg")
plt.savefig(f"simple_prior_conflict_and_model_misspecification_{x_power:0.2f}.png")

plt.show()

# %% Create another plot that has both a data-prior conflict and misspecification

# Define the number of subplots
n_rows, n_cols = 1, 1
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(15 / 2, 15 / 2.25 / 1.6 * 1.2), squeeze=False
)

combined_ax = axes[0, 0]

N_factor = 60000
x_power = 1.3

N_to_x, x_to_N = get_N_x_mapping(x_power=x_power, N_factor=N_factor)

Ns = np.arange(0, 60000, 4)
Xs = N_to_x(Ns)
# Concat a linrange from max(x) to 1
Xs = np.concatenate([Xs, np.linspace(1, Xs[-1], 1000, endpoint=False)[::-1]])
Ns = x_to_N(Xs)

for i, (loss_factor, best_model_accuracy) in enumerate(
    zip([0.5, 1.0, 3.0], [0.94, 0.89, 0.8])
):
    best_model_loss = get_multiclass_ce(num_classes, best_model_accuracy)
    loss = simulate_loss(loss_factor / 1.5, Ns, initial_loss, best_model_loss, N_factor)
    combined_ax.plot(Xs, loss, zorder=4 - i, label=model_template.format(i=i))
combined_ax.legend()

# Get current ticks
# combined_ax.set_xlim(0, 1.0)
current_ticks = combined_ax.get_xticks()
new_tick_labels = [get_tick_label(x) for x in current_ticks]
# combined_ax.set_xticks(current_ticks)
combined_ax.set_xticklabels(new_tick_labels)
combined_ax.get_xticklabels()[-1].set_fontsize(18)

combined_ax.set_title("Anti-Correlated Prior-Data Conflict & Model Misspecification")
combined_ax.set_xlabel(x_label)
combined_ax.set_ylabel(y_label)

# Add "blackhc@" text to the right corner
fig.text(1.0, 0.05, 'blackhc@', fontsize=10, color='gray', 
         ha='right', va='bottom', alpha=0.5, transform=fig.transFigure)

# Adjust layout and display the plot
plt.tight_layout()

# Save as SVG
plt.savefig(
    f"simple_anticorrelated_prior_conflict_and_model_misspecification_{x_power:0.2f}.svg"
)
plt.savefig(
    f"simple_anticorrelated_prior_conflict_and_model_misspecification_{x_power:0.2f}.png"
)

plt.show()

#%%
# Create a single figure with three subplots
n_rows, n_cols = 1, 3
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(20, 15 / 2.25 / 1.6 * 1.2), squeeze=False
)

# Function to copy content from one axes to another
def copy_axes_content(from_ax, to_ax):
    # Copy the plot content
    for line in from_ax.lines:
        to_ax.plot(line.get_xdata(), line.  get_ydata(), 
                   color=line.get_color(), 
                   linestyle=line.get_linestyle(), 
                   linewidth=line.get_linewidth(),
                   marker=line.get_marker(),
                   markersize=line.get_markersize(),
                   label=line.get_label())
    
    for collection in from_ax.collections:
        to_ax.add_collection(collection.copy())
        
    # Copy the axes settings
    to_ax.set_xlim(from_ax.get_xlim())
    to_ax.set_ylim(from_ax.get_ylim())
    to_ax.set_xlabel(from_ax.get_xlabel())
    to_ax.set_ylabel(from_ax.get_ylabel())
    to_ax.set_title(from_ax.get_title())
    
    # Copy x and y ticks and labels
    to_ax.set_xticks(from_ax.get_xticks())
    to_ax.set_xticklabels(from_ax.get_xticklabels())
    to_ax.set_yticks(from_ax.get_yticks())
    to_ax.set_yticklabels(from_ax.get_yticklabels())
    
    # Copy the axes settings
    to_ax.set_xlim(from_ax.get_xlim())
    to_ax.set_ylim(from_ax.get_ylim())
    to_ax.set_xlabel(from_ax.get_xlabel())
    to_ax.set_ylabel(from_ax.get_ylabel())
    to_ax.set_title(from_ax.get_title())
    
    # Copy the legend if it exists
    if from_ax.get_legend():
        to_ax.legend()

# Copy content from existing axes to new axes
copy_axes_content(model_misspecification_ax, axes[0, 0])
copy_axes_content(prior_data_conflict_ax, axes[0, 1])
copy_axes_content(combined_ax, axes[0, 2])
axes[0, 2].set_ylabel(None)

# Add "blackhc@" text to the right corner
fig.text(1.0, 0.05, 'blackhc@', fontsize=10, color='gray', 
         ha='right', va='bottom', alpha=0.5, transform=fig.transFigure)


plt.tight_layout()
plt.savefig(f"simple_combined_plots_{x_power:0.2f}.svg")
plt.savefig(f"simple_combined_plots_{x_power:0.2f}.png")
plt.show()


# %%
## Create another figure with two plots that visualize:
# - how the area under the marginal cross-entropy is equal the joint cross-entropy using the chain rule
# - how the area under the conditional training loss (NLL) is equal the marginal likelihood loss using the chain rule
# (when training with batch size 1)
# In the second plot, we also show how batch size >> 1 leads to a staircase of upper-bounds

# Define the number of subplots
n_rows, n_cols = 1, 2
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(15, 15 / 2.25 / 1.6 * 1.2), squeeze=False
)

marginal_ce_ax = axes[0, 0]
marginal_likelihoood_ax = axes[0, 1]

N_factor = 55000
x_power = 1.0

N_to_x, x_to_N = get_N_x_mapping(x_power=x_power, N_factor=N_factor)

Ns = np.arange(0, 48000, 16)
Xs = N_to_x(Ns)
# Concat a linrange from max(x) to 1
Xs = np.concatenate([Xs, np.linspace(1, Xs[-1], 1000, endpoint=False)[::-1]])
Ns = x_to_N(Xs)

auc_cv_loss_label = r"Summed CV-Loss $\propto$ Running CV-Loss"
train_loss_label = r"Training/Validation Loss"
auc_loss_label = r"Total Loss $\propto$ Running Loss"
cv_loss_label = r"Cross-Validation Loss"
performance_label = r"Loss"

# Left plot: joint cross-entropy as area under marginal cross-entropy
phi_accuracy = 0.92
phi_loss = get_multiclass_ce(num_classes, phi_accuracy)
loss = simulate_loss(1, Ns, initial_loss, phi_loss, N_factor)
mce_line = marginal_ce_ax.plot(
    Xs, loss, zorder=4 - i, color="C0", label=cv_loss_label
)
# marginal_ce_ax.text(20, loss[20], f"$\phi$", verticalalignment='bottom', horizontalalignment='left', c=mce_line[0].get_color())
xy = (0.5, loss[find_idx_by_x(Xs, 0.4)] + 0.1)
xytext = (0.5, loss[find_idx_by_x(Xs, 0.4)] + 1.0)
arrowprops = dict(facecolor=mce_line[0].get_color(), shrink=0.05)
marginal_ce_ax.annotate(
    cv_loss_label,
    zorder=10,
    xy=xy,
    xytext=xytext,
    arrowprops=arrowprops,
    verticalalignment="center",
    horizontalalignment="center",
    color=mce_line[0].get_color(),
)

marginal_ce_ax.fill_between(
    Xs,
    np.zeros_like(loss),
    loss,
    alpha=0.2,
    zorder=-1,
    label="Running Cross-Validation Loss",
    color="C1",
)
marginal_ce_ax.text(
    0.5,
    phi_loss / 2,
    auc_cv_loss_label,
    verticalalignment="center",
    horizontalalignment="center",
    color="k",
)

marginal_ce_ax.set_xlabel(x_label)
marginal_ce_ax.set_ylabel(performance_label)
marginal_ce_ax.legend()
current_ticks = marginal_ce_ax.get_xticks()
new_tick_labels = [get_tick_label(x) for x in current_ticks]
# marginal_ce_ax.set_xticks(current_ticks)
marginal_ce_ax.set_xticklabels(new_tick_labels)
marginal_ce_ax.get_xticklabels()[-1].set_fontsize(18)
marginal_ce_ax.set_title("Multiple Averaged Runs")

# Right plot: Marginal likelihood as area under conditional training loss
# Here we need to add some noise to the loss because we look at individual samples

diff_N = np.diff(Ns, prepend=Ns[0])
print(diff_N)
noise_scale = 0.1 / (Ns + 1) ** 0.4
scaled_noise = np.random.gumbel(0, noise_scale, len(loss))
noised_loss = loss - scaled_noise

marginal_likelihoood_ax.plot(
    Xs, noised_loss, zorder=4 - i, color="C2", label=train_loss_label
)
xy = (0.6, noised_loss[find_idx_by_x(Xs, 0.6)] + 0.1)
xytext = (0.5, noised_loss[find_idx_by_x(Xs, 0.6)] + 1.0)
arrowprops = dict(facecolor="C2", shrink=0.05)
marginal_likelihoood_ax.annotate(
    train_loss_label,
    zorder=10,
    xy=xy,
    xytext=xytext,
    arrowprops=arrowprops,
    verticalalignment="center",
    horizontalalignment="left",
    color="C2",
)

marginal_likelihoood_ax.fill_between(
    Xs,
    np.zeros_like(noised_loss),
    noised_loss,
    zorder=-1,
    alpha=0.2,
    label=f"Running Training/Validation Loss",
    color="C3",
)
marginal_likelihoood_ax.text(
    0.5,
    phi_loss / 2,
    auc_loss_label,
    verticalalignment="center",
    horizontalalignment="center",
    color="k",
)

marginal_likelihoood_ax.set_xlabel(x_label)
marginal_likelihoood_ax.set_ylabel(None)
marginal_likelihoood_ax.legend()

# marginal_likelihoood_ax.set_xticks(current_ticks)
marginal_likelihoood_ax.set_xticklabels(new_tick_labels)
marginal_likelihoood_ax.get_xticklabels()[-2].set_fontsize(18)
marginal_likelihoood_ax.set_title(r"One-Epoch Training $\Rightarrow$ Training Loss $\equiv$ Validation Loss")

plt.suptitle("Comparing (Cross-)Validation Loss and Training Loss Curves")

# # Simulate a larger batch size.
# batch_size = 512
# batch_idx = np.arange(0, 48000, batch_size)[:-1]
# batch_loss = loss[batch_idx]
# # Repeat each loss for the batch size times
# batched_loss = np.repeat(batch_loss[:, None], batch_size, axis=-1).flatten()
# noised_batched_loss = batched_loss + np.random.gumbel(0, noise_scale[:len(batched_loss)], len(batched_loss))

# marginal_likelihoood_ax.plot(Xs[:len(batched_loss)], noised_batched_loss, zorder=0, color="C4", label=f"Batch Size {batch_size}")

# Add "blackhc@" text to the right corner
fig.text(1.0, 0.05, 'blackhc@', fontsize=10, color='gray', 
         ha='right', va='bottom', alpha=0.5, transform=fig.transFigure)

plt.tight_layout()
# Save as SVG
plt.savefig(f"simple_area_under_curve_{x_power:0.2f}.svg")
plt.savefig(f"simple_area_under_curve_{x_power:0.2f}.png")
plt.show()

# %%
