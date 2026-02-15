
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

rounds = list(range(1, 11))
loss_values = [0.6923, 0.6827, 0.6720, 0.6597, 0.6445, 0.6271, 0.6084, 0.5895, 0.5694, 0.5507]
accuracy_values = [0.4918, 0.5246, 0.5902, 0.6885, 0.7377, 0.7705, 0.7705, 0.7705, 0.7869, 0.7705]

accuracy_pct = [acc * 100 for acc in accuracy_values]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

ax1.plot(rounds, loss_values, marker='o', linewidth=2.5, markersize=8, 
         color='#E74C3C', label='Global Model Loss')
ax1.fill_between(rounds, loss_values, alpha=0.3, color='#E74C3C')
ax1.set_xlabel('Training Round', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
ax1.set_title('Loss Reduction Over Federated Training Rounds', 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(rounds)

loss_improvement = ((loss_values[0] - loss_values[-1]) / loss_values[0]) * 100
ax1.text(0.98, 0.95, f'↓ {loss_improvement:.1f}% reduction', 
         transform=ax1.transAxes, fontsize=11, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.plot(rounds, accuracy_pct, marker='s', linewidth=2.5, markersize=8, 
         color='#27AE60', label='Global Model Accuracy')
ax2.fill_between(rounds, accuracy_pct, alpha=0.3, color='#27AE60')
ax2.set_xlabel('Training Round', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Improvement Over Federated Training Rounds', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(rounds)
ax2.set_ylim([45, 85])

acc_improvement = accuracy_pct[-1] - accuracy_pct[0]
ax2.text(0.98, 0.05, f'↑ +{acc_improvement:.1f}% improvement', 
         transform=ax2.transAxes, fontsize=11, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
x = np.arange(len(['Round 1', 'Round 5', 'Round 10']))
width = 0.35

initial_acc = accuracy_pct[0]
mid_acc = accuracy_pct[4]
final_acc = accuracy_pct[9]

bars = ax3.bar(x, [initial_acc, mid_acc, final_acc], width, 
               color=['#E74C3C', '#F39C12', '#27AE60'], alpha=0.8)

ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy Progression: Start → Middle → Final', 
              fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(['Round 1\n(Initial)', 'Round 5\n(Midpoint)', 'Round 10\n(Final)'])
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

ax4.axis('off')

summary = f"""
AEGIS FEDERATED LEARNING RESULTS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAINING CONFIGURATION
   • Clients (Hospitals):      3
   • Training Rounds:          10
   • Total Duration:           ~19 seconds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE METRICS

   Initial Accuracy:           {accuracy_pct[0]:.1f}%
   Final Accuracy:             {accuracy_pct[-1]:.1f}%
   Improvement:                +{acc_improvement:.1f}%
   
   Initial Loss:               {loss_values[0]:.4f}
   Final Loss:                 {loss_values[-1]:.4f}
   Reduction:                  -{loss_improvement:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, 
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('AEGIS: Privacy-Preserving Federated Learning for Heart Disease Prediction', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_file = 'aegis_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_file}")

try:
    plt.show()
except:
    print("(Plot display not available in headless mode - check the PNG file)")