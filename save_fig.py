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
