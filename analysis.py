import ast
from os import makedirs

from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

mpl_use('Agg')

fig_size = 6  # 4
rect_fig_size = (fig_size * 1.618, fig_size)
# rect_fig_size = (fig_size * 2, fig_size)
# rect_fig_size = (fig_size * 2 * 1.618, fig_size)
pic_type = '.png'

makedirs('./plots/', exist_ok=True)

def get_losses(file_path):
  losses = []
  with open(file_path, 'r') as data:
    line = data.readline() # throw away first line
    while True:
      line = data.readline()
      if not line:
        break
      dictionary = ast.literal_eval(line)
      losses.append(dictionary['loss'])
  return losses


def plot_losses():
  their_normal_losses = get_losses("train_stats.txt")[:400]
  our_normal_losses = get_losses("checkpoint/stats.txt.normal")
  our_orth_losses = get_losses("checkpoint/stats.txt.orth")

  fig = plt.figure(figsize=rect_fig_size)
  ax = fig.add_subplot(111)
  ax.spines['right'].set_color(None)
  ax.spines['top'].set_color(None)
  ax.set_xlabel('Step /100s')
  ax.set_ylabel('Loss')
  ax.plot(their_normal_losses, label='Original author\'s run')
  ax.plot(our_normal_losses, label='Ours - LARS')
  ax.plot(our_orth_losses,  label='Ours - Orthogonalised LARS')
  ax.legend()

  fig.savefig('plots/losses' + pic_type, bbox_inches='tight')
  plt.close(fig)


if __name__ == '__main__':
  plot_losses()
