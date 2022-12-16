import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import ClientsideFunction, Input, Output, State
import plotly.express as px
from ipywidgets import widgets
from helper.plots import custom_dims_plot
import urllib.request, json

with open("mapping_dict_final_binary.json", "r") as f:
    dimension_mapper_binary = json.load(f)

with open("mapping_dict_final.json", "r") as f:
    dimension_mapper = json.load(f)

city_info_bin = pd.read_csv("data_bool_geo_final.csv")

filters = []
for k, v in dimension_mapper_binary.items():
    filters.append(dict(label=v, value=k))

temperature_df = pd.read_csv("city_temperature.csv")

# 重命名
data = pd.read_csv("df_final.csv")
data.rename(columns={"Cost of Living Index": "生活成本", "Purchasing Power Index": "购买力指数",
            "Safety Index": "安全指数", "Health Care Index": "医保指数", "Pollution Index": "污染指数",
            "Startup": "生命力", "Internet Speed": "网速", "Gender Equality": "男女比",
            "Immigration Tolerance": "移民亲和度", "LGBT Friendly": "LGBT宽容度", "Nightscene": "景色",
            "Freedom to make life choices": "生活自由度", "Generosity": "社会关爱感"
}, inplace=True)
city_info_num = data.copy()

colnames_to_lower = dict(
    zip(
        city_info_num.drop(columns=["City", "Country", "Lat", "Long"]).columns,
        map(
            str.lower,
            city_info_num.drop(columns=["City", "Country", "Lat", "Long"]).columns,
        ),
    )
)
city_info_num.rename(columns=colnames_to_lower, inplace=True)

city_info_num_agg = city_info_num.drop(columns=["City", "Country"]).apply(np.median)

# 主页div配置
filters_layout = html.Div(
    [
        html.Div(
            [
                html.H3("点我开始探索吧！", style={"display": "inline"}),
                html.Span(
                    [html.Span(className="Select-arrow", title="is_open")],
                    className="Select-arrow-zone",
                    id="select_filters_arrow",
                ),
            ],
        ),
        html.Div(
            [
                html.P("请选择感兴趣的问题", id="preferencesText"),
                dcc.Dropdown(
                    placeholder="（支持多选）",
                    id="filters_drop",
                    options=filters,
                    clearable=False,
                    className="dropdownMenu",
                    multi=True,
                ),
            ],
            id="dropdown_menu_applied_filters",
        ),
    ],
    id="filters_container",
    style={"display": "block"},
    className="stack-top col-3",
)

# index页介绍
initial_popup_layout = html.Div(
    [
        html.H1("全球知名城市可视化分析", className="title"),
        html.H3("功能列表"),
        html.P("1、提供按感兴趣方向探索城市功能"),
        html.P("2、气泡图可视化：展示GDP、生活自由程度、社会关爱感之间的关系"),
        html.P("3、并行坐标图可视化：支持按安全感、消费水平等指标进行选择"),
        html.P("4、分页图可视化：支持多个条件分页罗列比较"),
        html.P("5、折线图可视化：展示城市年均气温变化水平"),
        html.P("6、雷达图可视化：总览城市在多方面的综合情况"),
        html.H3("一键启动！"),
        html.Div(
            [
                html.Div(
                    [
                        html.H6("小组成员:"),
                        html.P("赵瑞坤  方鹏贺", className="author_name"),
                        html.P("杨安恺  张泽宇", className="author_name"),
                    ]
                ),
                html.Div(
                    [
                        html.H6("参考材料:"),
                        html.P(
                            "https://www.kaggle.com/stephenofarrell/cost-of-living",
                            className="source",
                        ),
                        html.P(
                            "https://www.economist.com/big-mac-index",
                            className="source",
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "align": "right", "bottom": "10%"},
        ),
    ],
    id="initial_popup",
)

# 导航栏
info_bar_layout = html.Div(
    [
        html.H1("全球知名城市可视化分析", className="title"),
        html.H3(
            "在此仪表板中，在选择感兴趣的指标后，点击城市以进一步探索！",
            className="subtitle",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H6("小组成员:"),
                        html.P(
                            "赵瑞坤 - 方鹏贺 - 杨安恺 - 张泽宇",
                            className="author_name",
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "align": "right"},
        ),
    ],
    className="stack-top info_bar row",
    style={"display": "block"},
    id="info_bar",
)

# 小窗可视化
selected_location_layout = html.Div(
    [
        html.Div(
            [
                html.H3("", id="title_selected_location"),
                html.Span("关闭", id="x_close_selection"),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("可视化参考指标"),
                        html.H6(
                            "人均GDP气泡图"
                        ),
                        dcc.Graph(id="bubble"),
                    ],
                    className="plot_container_child",
                ),
                html.Div(
                    [
                        html.H4("选定的标准"),
                        dcc.Graph(id="custom_dims_plot"),
                    ],
                    className="plot_container_child",
                ),
            ],
            className="plots_container",
        ),
        html.Div(
            [
                html.H4("气温折线图"),
                dcc.Graph(id="temperature_plot"),
            ],
            className="plots_container",
        ),
    ],
    id="selected_location",
    style={"display": "none"},
)

# 雷达图悬停
hovered_location_layout = html.Div(
    [html.Div([html.H3("city", id="hover_title"), dcc.Graph("radar")]),],
    id="hovered_location",
    style={"display": "none"},
)


app = dash.Dash(__name__, external_stylesheets="")

suppress_callback_exceptions = True

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        initial_popup_layout,
        html.Div(
            [
                html.Div(
                    id="width", style={"display": "none"}
                ),  # 获取当前窗口参数
                html.Div(
                    id="height", style={"display": "none"}
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="map",
                            clear_on_unhover=True,
                            config={"doubleClick": "reset"},
                        )
                    ],
                    style={"width": "100%", "height": "100%"},
                    className="background-map-container",
                ),
            ],
            id="map_container",
            style={"display": "flex"},
        ),
        filters_layout,
        info_bar_layout,
        selected_location_layout,
        hovered_location_layout,
    ],
    id="page-content",
    style={"position": "relative"},
)

selections = set()

# 绘图函数
@app.callback(Output("initial_popup", "style"), Input("initial_popup", "n_clicks"))
def close_initial_popup(n_clicks):
    show_block = {"display": "block"}
    hide = {"display": "none"}
    if n_clicks is not None:
        return hide
    else:
        return show_block


@app.callback(
    Output("dropdown_menu_applied_filters", "style"),
    Output("select_filters_arrow", "title"),
    Input("select_filters_arrow", "n_clicks"),
    State("select_filters_arrow", "title"),
)
def toggle_applied_filters(n_clicks, state):
    style = {"display": "none"}
    if n_clicks is not None:
        if state == "is_open":
            style = {"display": "none"}
            state = "is_closed"
        else:
            style = {"display": "block"}
            state = "is_open"

    return style, state


selected_location = ""
x_close_selection_clicks = -1


@app.callback(Output("bubble", "clickData"), [Input("map", "clickData")])
def update_bubble_selection(click_map):
    point = click_map
    return point


@app.callback(
    Output("selected_location", "style"),
    Output("title_selected_location", "children"),
    Output("custom_dims_plot", "figure"),
    Output("bubble", "figure"),
    Output("temperature_plot", "figure"),
    [Input("map", "clickData")],
    Input("x_close_selection", "n_clicks"),
    [Input("filters_drop", "value")],
    [Input("bubble", "selectedData"), Input("bubble", "clickData")],
    Input("width", "n_clicks"),
    Input("height", "n_clicks"),
    [State("bubble", "figure")],
)
def update_selected_location(
    clickData,
    n_clicks,
    dims_selected,
    bubbleSelect,
    bubbleClick,
    width,
    height,
    bubbleState,
):

    global selected_location
    global x_close_selection_clicks
    location = ""

    if clickData is not None or dims_selected is not None:
        if clickData is not None:
            location = clickData["points"][0]["text"]
        if len(location) != 0:
            selected_location = location
            style = {"display": "block"}
        else:
            selected_location = ""
            location = selected_location
            style = {"display": "none"}
    else:
        style = {"display": "none"}

    if n_clicks != x_close_selection_clicks:
        style = {"display": "none"}
        selected_location = ""
        x_close_selection_clicks = n_clicks

    if bubbleSelect is not None or bubbleClick is not None or bubbleState is not None:
        bubble_fig = update_color(bubbleSelect, bubbleClick, bubbleState, width, height)
    else:
        bubble_fig = build_bubble_figure(width, height)
    return (
        style,
        location+"数据可视化",
        update_custom_dims_plot(location, dims_selected, width, height),
        bubble_fig,
        update_temperature(location, width, height),
    )

# 气温折线图
def update_temperature(city, width, height):
    df = temperature_df
    row = df[df["City"] == city]

    avg = df.groupby("Month").mean().reset_index()

    trace0 = go.Scatter(
        x=row["Month"], y=row["AvgTemperature"], name=city, marker=dict(color="#f3d576")
    )

    trace1 = go.Scatter(
        x=avg["Month"],
        y=avg["AvgTemperature"],
        name="Average Temperature",
        marker=dict(color="#d1d1cf"),
    )

    layout = go.Layout(
        xaxis=dict(
            showline=True,
            linecolor="white",
            showgrid=False,
            tickmode="array",
            tickvals=[i for i in range(1, 13)],
            ticktext=[
                "一月",
                "二月",
                "三月",
                "四月",
                "五月",
                "六月",
                "七月",
                "八月",
                "九月",
                "十月",
                "十一月",
                "十二月",
            ],
        ),
        yaxis=dict(showline=True, linecolor="white", showgrid=False),
    )

    data = [trace0, trace1]

    fig = go.Figure(data=data, layout=layout)

    fig.update_yaxes(ticksuffix="°C")

    fig.update_layout(
        height=int(height * 0.2),
        width=int(width * 0.7),
        margin=dict(
            l=120,  # left margin
            r=120,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Open sans", size=12, color="White"),
    )

    return fig

# 分页对比图
def update_custom_dims_plot(location, dims_selected, width, height):
    if dims_selected is None or len(dims_selected) == 0:
        dims_selected = ["tourism"]
    if len(location) == 0:
        return go.Figure()
    fig = custom_dims_plot(
        location,
        dims_selected,
        city_info_num,
        city_info_num_agg,
        dimension_mapper,
        width,
        height,
    )
    return fig


hovered_location = ""


@app.callback(
    Output("hovered_location", "style"),
    Output("radar", "figure"),
    Output("hover_title", "children"),
    [Input("map", "hoverData")],
)# 悬停雷达图
def update_hovered_location(hoverData):
    global hovered_location
    location = ""
    if hoverData is not None:
        location = hoverData["points"][0]["text"]
        if location != hovered_location:
            hovered_location = location
            style = {"display": "block"}
        else:
            hovered_location = ""
            location = ""
            style = {"display": "none"}
    else:
        hovered_location = ""
        location = ""
        style = {"display": "none"}

    return style, update_radar(location), location


# 雷达图数据传入
def update_radar(city):
    df = data[
        [
            "City",
            "生活成本",
            "购买力指数",
            "安全指数",
            "医保指数",
            "污染指数",
        ]
    ]
    cat = df.columns[1:].tolist()

    select_df = df[df["City"] == city]

    Row_list = []
    r = []
    for index, rows in select_df.iterrows():
        for i in range(len(cat)):
            r.append(rows[cat[i]])

        Row_list.append(r)
        Row_list = list(np.concatenate(Row_list).flat)

    fig = go.Figure()

    fig.add_trace(
        go.Barpolar(
            r=Row_list,
            theta=cat,
            name=city,
            marker_color=["rgb(243,203,70)"] * 6,
            marker_line_color="white",
            hoverinfo=["theta"] * 9,
            opacity=0.7,
            base=0,
        )
    )

    fig.add_trace(
        go.Barpolar(
            r=df.mean(axis=0).tolist(),
            theta=cat,
            name="Average",
            marker_color=["#986EA8"] * 6,
            marker_line_color="white",
            hoverinfo=["theta"] * 9,
            opacity=0.7,
            base=0,
        )
    )

    fig.update_layout(
        title="",
        font_size=12,
        margin=dict(
            l=110,  # left margin
            r=120,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        ),
        height=150,
        width=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        legend=dict(orientation="h",),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(linewidth=3, showline=False, showticklabels=True),
            radialaxis=dict(
                showline=False,
                showticklabels=False,
                linewidth=2,
                gridcolor="rgba(0,0,0,0)",
                gridwidth=2,
            ),
        ),
    )

    return fig


@app.callback(
    Output("page-content", "style"),
    Input("width", "n_clicks"),
    Input("height", "n_clicks"),
)
def set_page_size(width, height):
    return {"width": width, "height": height}


@app.callback(
    Output("map", "figure"),
    [Input("filters_drop", "value")],
    Input("width", "n_clicks"),
    Input("height", "n_clicks"),
)
def update_map(filter_list, width, height):
    fig = go.Figure()

    if filter_list is not None and len(filter_list) != 0:

        filters = []
        for f in filter_list:
            filters.append(city_info_bin[f])
        highlighted = city_info_bin.loc[
            np.all(filters, 0), ["City", "Country", "Lat", "Long"]
        ]
        not_highlighted = city_info_bin.loc[
            ~np.all(filters, 0), ["City", "Country", "Lat", "Long"]
        ]

        fig.add_trace(
            go.Scattermapbox(
                lat=highlighted.Lat,
                lon=highlighted.Long,
                text=highlighted.City,
                name="比较匹配的城市",
                mode="markers",
                marker=go.scattermapbox.Marker(size=15, opacity=0.9, color="#F3D576",),
                hovertemplate="<extra></extra>",
            )
        )
    else:
        not_highlighted = city_info_bin

    fig.add_trace(
        go.Scattermapbox(
            lat=not_highlighted.Lat,
            lon=not_highlighted.Long,
            text=not_highlighted.City,
            name="不太符合的城市",
            mode="markers",
            marker=go.scattermapbox.Marker(size=10, opacity=0.9, color="#333333",),
            hovertemplate="<extra></extra>",
        )
    )

    mapbox_token = "pk.eyJ1IjoiZmFya2l0ZXMiLCJhIjoiY2ttaHYwZnQzMGI0cDJvazVubzEzc2lncyJ9.fczsOA4Hfgdf8_bAAZkdYQ"
    all_plots_layout = dict(
        mapbox=dict(
            style="mapbox://styles/farkites/ckn0lwfm319ae17o5jmk3ckvu",
            accesstoken=mapbox_token,
        ),
        legend=dict(
            bgcolor="rgba(51,51,51,0.6)",
            yanchor="top",
            y=0.35,
            xanchor="left",
            x=0,
            font=dict(family="Open Sans", size=15, color="white",),
        ),
        autosize=False,
        width=width,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
    )
    fig.layout = all_plots_layout

    return fig


md = data[
    [
        "City",
        "Employment",
        "生命力",
        "Tourism",
        "Housing",
        "Logged GDP per capita",
        "Transport",
        "Health",
        "Food",
        "网速",
        "社会关爱感",
        "生活自由度",
        "Access to Contraception",
        "男女比",
        "移民亲和度",
        "LGBT宽容度",
        "景色",
        "Beer",
        "Festival",
    ]
].copy()

for column in md.columns.tolist()[1:]:
    md["{column}.".format(column=column)] = pd.qcut(
        md[column].rank(method="first"), 4, labels=False
    )


# 并行坐标系图
quartiles = [
    "生命力.",
    "网速.",
    "男女比.",
    "移民亲和度.",
    "LGBT宽容度.",
    "景色.",
]


dimensions = []
for label in quartiles:
    tmp = go.parcats.Dimension(
        values=md[label], categoryorder="category descending", label=label
    )
    dimensions.append(tmp)

color = np.zeros(len(md), dtype="uint8")
colorscale = [[0, "gray"], [1, "rgb(243,203,70)"]]


gdp = 10 ** (md["Logged GDP per capita"] / 10)

sizeref = 2.0 * max(gdp) / (20 ** 2)

customdata = np.stack((pd.Series(md.index), md["City"], gdp.round(5) * 1000), axis=-1)

# 气泡图
def build_bubble_figure(width, height):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=md["生活自由度"],
                y=md["社会关爱感"],
                text=md["City"],
                customdata=customdata,
                hovertemplate="""<extra></extra>
        <em>%{customdata[1]}</em><br>
        人均GDP：%{customdata[2]}$""",
                marker={
                    "color": "#986EA8",
                    "sizeref": sizeref,
                    "sizemin": 0.005,
                    "sizemode": "area",
                    "size": gdp,
                },
                mode="markers",
                selected={"marker": {"color": "rgb(243,203,70)"}},
                unselected={"marker": {"opacity": 0.3}},
            ),
            go.Parcats(
                domain={"y": [0, 0.4]},
                dimensions=dimensions,
                line={
                    "colorscale": colorscale,
                    "cmin": 0,
                    "cmax": 1,
                    "color": color,
                    "shape": "hspline",
                },
            ),
        ]
    )

    fig.update_layout(
        font_color="white",
        font_size=9,
        xaxis={"title": "生活自由度"},
        yaxis={"title": "社会关爱感", "domain": [0.6, 1]},
        dragmode="lasso",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        bargap=0.35,
        height=int(height*0.26),
        width=int(width*0.46),
        margin=dict(
            l=15,  # left margin
            r=15,  # right margin
            b=10,  # bottom margin
            t=0,  # top margin
        ),
    )

    return fig


# 对是否点击进行颜色更新
def update_color(selectedData, clickData, fig, width, height):
    selection = None
    trigger = dash.callback_context.triggered[0]["prop_id"]
    if trigger == "bubble.clickData":
        selection = [point["pointNumber"] for point in clickData["points"]]
    if trigger == "bubble.selectedData":
        selection = [point["pointIndex"] for point in selectedData["points"]]
    # 更新散点图
    fig["data"][0]["selectedpoints"] = selection
    # 更新并行坐标系
    new_color = np.zeros(len(md), dtype="uint8")
    new_color[selection] = 1
    fig["data"][1]["line"]["color"] = new_color
    return fig


# 全局移动
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="get_window_width"),
    Output("width", "n_clicks"),
    [Input("url", "href")],
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="get_window_height"),
    Output("height", "n_clicks"),
    [Input("url", "href")],
)

app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="move_hover"),
    Output("hovered_location", "title"),
    [Input("map", "hoverData")],
)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
