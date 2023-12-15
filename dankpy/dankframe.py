import pandas as pd


def from_list(list_of_dicts: list) -> pd.DataFrame:
    """
    Converts a list of dictionaries to pandas Dataframe object with keys as columns

    Args:
        list_of_dicts (list): list of dictionaries to convert

    """
    try: 
        return DankFrame(x.__dict__ for x in list_of_dicts)
    except:
        return pd.DataFrame(list_of_dicts)


class DankFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_list(self, list_of_dicts: list) -> pd.DataFrame:
        """
        Converts a list of dictionaries to pandas Dataframe object with keys as columns

        Args:
            list_of_dicts (list): list of dictionaries to convert

        """
        df = pd.DataFrame(x.__dict__ for x in list_of_dicts)
        df = df.replace({pd.np.nan: None})

        return df

    def list_filter(self, filter: list, column=None) -> pd.DataFrame:
        pattern = "|".join(filter)

        if column is None:
            self = self[
                self.apply(lambda r: r.str.contains(pattern, na=False).any(), axis=1)
            ]
        if isinstance(column, list):
            self = self[
                self[column].apply(
                    lambda r: r.str.contains(pattern, na=False).any(), axis=1
                )
            ]
        elif isinstance(column, str):
            self = self[self[column].str.contains(pattern, na=False)]

        return self

    def to_latex(self, filepath: str, caption: str, label: str = None, precision:int = 0) -> None:

        s = self.style.hide(axis="index").format(precision=precision).to_latex(
            position="h!",
            position_float="centering",
            caption=caption,
            label=f"tab:{label}",
            hrules=True,
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            s = s.replace(r"\centering",r"\footnotesize\centering")
            # s = s.replace(r"\begin{tabular}", r"\begin{tabularx}{\textwidth}")
            # s = s.replace(r"r", r"X")
            # s = s.replace(r"\begin{tabular}",r"\resizebox{\textwidth}{!}{\begin{tabular}")
            # s = s.replace(r"\end{tabular}",r"\end{tabularx}")

            f.write(s)

    def to_html(self) -> str:
        """
        Outputs pandas Dataframe to HTML table

        Returns:
            str: HTML table
        """
        return (
            self.to_html(
                index=False, classes=["table-bordered", "table-striped", "table-hover"]
            )
            .replace("\n", "")
            .replace("dataframe", "table")
        )
