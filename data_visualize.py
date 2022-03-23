import os
from pathlib import Path

def save_fig(filename: str, folder: str='./visualizations/'):
    """
    This function saves the current visualization in the 'visualizations' folder.

    Args:
    - filename (str): The name of the file, without the extension.
    - folder (str): Path to the folder to save the image.
    """
    folder = Path(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename += '.png'
    filename = folder / filename
    plt.savefig(filename, dpi=300, facecolor='w', bbox_inches='tight')

    
def show_values(axs, orient="v", space=.01):
    """
    This function adds data labels to one or several graphs.
    Args:
    - axs: One single matplotlib Ax object or an array with 2 or more of them.
    - orient: 'v' for vertical orientation, 'h' for horizontal orientation.
    - space: In case of horizontal orientation, space from the graph point to the label.
    """

    def _single(ax_):
        if orient == "v":
            for p in ax_.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                value = '{:.1f}'.format(p.get_height())
                ax_.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax_.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = '{:.1f}'.format(p.get_width())
                ax_.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
