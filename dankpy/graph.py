import io
import base64
import os
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

mapbox_access_token = "pk.eyJ1IjoiZGthZHlyb3YiLCJhIjoiY2t5Y2kzZW1uMDgxNzMxbjM0YWVoeGhpYSJ9.dBAYii97EtyfHVFVUYLPeg"

colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate color for any value in that range. Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]
    Others are just swatches that need to be constructed into a colorscale:
        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    Args:
        colorscale (str): A plotly continuous colorscale defined with RGB string colors._
        intermed (float): value in the range [0, 1]

    Raises:
        ValueError: if the colorscale is not a plotly continuous colorscale

    Returns:
        str: RGB string color
    """

    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width (int): width of figure in cm
        fraction (float, optional): fraction of width to use. Defaults to 1.
        subplots (tuple, optional): number of rows and columns of subplots. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "document":
        width_pt = 469.75502
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


class graph(go.Figure):
    """
    Custom Plotly figure object
    """
    zoom = None

    def __init__(self, *args, **kw):
        super(graph, self).__init__(*args, **kw)
        pio.full_figure_for_development(self, warn=False)
        # self = go.Figure()
        self.update_layout(
            font_family="Arial",
            template="simple_white",
            title={"x": 0.5, "y": 0.9, "xanchor": "center", "yanchor": "top"},
            yaxis=dict(
                mirror=True,
                ticks="",
                showline=True,
                # tickwidth=0,
                linecolor="#000000",
                fixedrange=True,
                # zeroline = True,
                # zerolinecolor="black",
                tickfont=dict(color="#000000"),
                showgrid=True,
            ),
            xaxis=dict(
                mirror=True,
                ticks="",
                showline=True,
                # tickwidth=0,
                linecolor="#000000",
                fixedrange=True,
                # zeroline = True,
                # zerolinecolor="black",
                tickfont=dict(color="#000000"),
                showgrid=True,
            ),
            legend=dict(
                yanchor="top",
                xanchor="right",
                borderwidth=0.5,
                x=0.995,
                y=0.99,
                bgcolor="rgba(255,255,255,0.5)",
            ),
        )

    def save_latex(self, name, keep_ticks=False):
        """
        Save figure for LaTeX PDF. 

        Args:
            name (str): name of the file
            keep_ticks (bool, optional): Whether to keep ticks or have plotly automatically scale them. Defaults to False.
        """
        self.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
            }
        )

        if not keep_ticks:
            self.update_layout(
                yaxis=dict(dtick=None, tickmode="auto"),
            )

        if len(self["data"]) == 1:
            if "line" in self["data"][0]:
                self["data"][0]["line"]["color"] = "black"
            if "marker" in self["data"][0]:
                self["data"][0]["marker"]["color"] = "black"

        self.update_layout(
            width=500,
            height=250,
            font_family="Serif",
            title=None,
            font_size=14,
            margin_l=5,
            margin_t=5,
            margin_b=5,
            margin_r=5,
        )
        pio.write_image(self, name, width=1.5 * 300, height=0.75 * 300)

    def save_image(
        self,
        name,
        dpi=300,
        height=None,
        width=None,
        scale=3,
    ):
        """
        Save figure as image.

        Args:
            name (str): name of the file
            dpi (int, optional): DPI of the image. Defaults to 300.
            height (int, optional): height of the image. Defaults to None.
            width (int, optional): width of the image. Defaults to None.
            scale (int, optional): scale of the image. Defaults to 3.
        """
        if height is None and width is None:
            width, height = 6 * dpi, 3 * dpi

        if len(self["data"]) == 1:
            if "line" in self["data"][0]:
                self["data"][0]["line"]["color"] = "black"
            if "marker" in self["data"][0]:
                self["data"][0]["marker"]["color"] = "black"

        self.update_layout(
            margin=dict(l=5, r=60, t=75, b=5),
            autosize=False,
            width=width,
            height=height,
            font_family="Arial",
        )

        self.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
            }
        )

        self.write_image(f"{name}", scale=scale)

    def save_html(self, name):
        """
        Save figure as HTML.

        Args:
            name (str): name of the file
        """
        self.write_html(f"{name}")

    def export_html(self):
        """
        Export figure as HTML.

        Returns:
            str: HTML code
        """
        return self.to_html(full_html=False)

    def to_bytes(self):
        """
        Export figure as bytes.

        Returns:
           bytes: bytes of the figure
        """
        return io.BytesIO(self.to_image(format="png"))

    def to_base64(self, dpi=120, keep_ticks=False):
        """
        Export figure as base64.

        Args:
            dpi (int, optional): DPI of the image. Defaults to 120.
            keep_ticks (bool, optional): Whether to keep ticks or have plotly automatically scale them. Defaults to False.

        Returns:
            str: base64 string of the figure
        """
        self.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),
            title=None,
        )

        if keep_ticks != True:
            self.update_layout(
                yaxis=dict(dtick=None, tickmode="auto"),
            )

        image = self.to_image(format="png", width=6 * dpi, height=3 * dpi, scale=3)

        return base64.b64encode(image).decode("ascii")