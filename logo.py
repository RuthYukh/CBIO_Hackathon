import pandas as pd
import numpy as np
import logomaker


class Logo_class:
    def __init__(self, emission_matrix) -> None:
        self.emission_matrix = emission_matrix

    def calculate_information_content(self):
        """
         Calculate entropy and then information content for each position
        """
        entropy_df = pd.DataFrame()
        for column in self.emission_matrix.columns:
            p = self.emission_matrix[column]
            # Entropy calculation -H(p)
            entropy = -np.sum(p * np.log2(p+1e-9))  # Adding a small number to avoid log(0)
            # Information content is max entropy - entropy
            max_entropy = np.log2(len(p))
            info_content = max_entropy - entropy
            entropy_df[column] = p * info_content  # Scale probabilities by information content
        return entropy_df

    # Normalize the scaled information content if desired
    # info_content_df = info_content_df / info_content_df.sum()

    def get_logo(self):
        info_content = self.calculate_information_content()

        # Create the logo
        logo = logomaker.Logo(info_content.transpose(), center_values=False)

        # Customize the logo
        logo.style_xticks(rotation=90)
        logo.ax.set_ylabel('Information Content')

        # Show or save the logo
        logo.fig.show()
        # To save:
        logo.fig.savefig('sequence_logo.png', dpi=300)