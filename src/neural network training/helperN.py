import matplotlib.pyplot as plt
from IPython import display
plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.axhline(y=0, color='r', linestyle='--', label="y = 0")  # Add this line to draw y = 0 line

    plt.ylim(ymin=-10)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.0001)
    

    
def save_plot(scores, mean_scores, save_path):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_bar_scores(scores, values):
    plt.bar(scores, values, color='blue')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('number of score')
    plt.show(block=True)



