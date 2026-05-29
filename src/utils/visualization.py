import matplotlib.pyplot as plt
from config import FIGURES_DIR

def save_fig(title: str) -> None:
    plt.savefig(str(FIGURES_DIR / (title.replace(" ", "_").lower() + ".png")), bbox_inches='tight')