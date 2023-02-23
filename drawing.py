import pandas as pd
import numpy as np
import pyecharts.options as opts  # 并加载pyecharts选项
from pyecharts.charts import *
import warnings
from pyecharts.globals import CurrentConfig, NotebookType  # 加载 Jupyter lab中设置 pyecharts 全局显示参数
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
warnings.filterwarnings("ignore")  # 忽略版本问题

# 数据导入
age_m = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='data_age', index_col=0)  # 各年龄段场均数据
win_lose_age = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='win_lose_age', index_col=0)  # 各年龄段胜败场数
home_away_age = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='home_away_age', index_col=0)  # 各年龄段主客场胜败场数
home_away = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='home_away', index_col=0)  # 主客场胜败总数
sp_age = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='dff_point', index_col=0)  # 各年龄段最大正负值
s_age = pd.read_excel('./data/Regular_Age.xlsx', sheet_name='age_sum', index_col=0)  # 各年龄段总数据

team_m = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='data_team', index_col=0)  # 对阵各队场均数据
win_lose_team = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='win_lose_team', index_col=0)  # 对阵各队胜败场数
home_away_team = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='home_away_team', index_col=0)  # 对阵各队主客场胜败场数
home_away = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='home_away', index_col=0)  # 主客场胜败总数
sp_team = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='dff_point', index_col=0)  # 对阵各队最大正负值
s_team = pd.read_excel('./data/Regular_Team.xlsx', sheet_name='team_sum', index_col=0)  # 对阵各队总数据

figsize = opts.InitOpts(width='600px', height='400px',
                        bg_color='#ffffff')  # 设置图形大小和背景色 rgb(225, 225, 225, 0.5)
# 投篮点图绘制
def scatter_paul_george() -> Scatter:
    df_george = pd.read_csv('./data/paul_george_shot_xy.csv')

    # 选择投中，投丢的数据
    df_george_shot_made_flag_1 = df_george[df_george.EVENT_TYPE == 'Made Shot']
    df_george_shot_made_flag_0 = df_george[df_george.EVENT_TYPE == 'Missed Shot']
    # 选择投篮半场的数据
    df_shot_in_midcourt_1 = df_george_shot_made_flag_1[df_george_shot_made_flag_1['LOC_Y'] <= 300]
    df_shot_in_midcourt_0 = df_george_shot_made_flag_0[df_george_shot_made_flag_0['LOC_Y'] <= 300]

    c = (
        Scatter(init_opts=opts.InitOpts(
            bg_color='#ffffff'))

        .add_xaxis(xaxis_data=df_shot_in_midcourt_0["LOC_X"].values.tolist())
        .add_yaxis(
            series_name="投偏",
            y_axis=df_shot_in_midcourt_0["LOC_Y"].values.tolist(),
            symbol_size=3,
            label_opts=opts.LabelOpts(is_show=False),
            color="#542481",
        )

        .add_xaxis(xaxis_data=df_shot_in_midcourt_1["LOC_X"].values.tolist())
        .add_yaxis(
            series_name="投进",
            y_axis=df_shot_in_midcourt_1["LOC_Y"].values.tolist(),
            symbol_size=2,
            label_opts=opts.LabelOpts(is_show=False),
            color="#FEDC56"
        )

        .set_series_opts()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=False),
                is_show=False
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                is_show=False
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            title_opts=opts.TitleOpts(
                title="保罗乔治职业生涯投篮点图）", pos_top="10%",
            ),
            legend_opts=opts.LegendOpts(pos_top="15%", pos_left="5%"),
        )
    )
    return c


v1 = [team_m.loc['ATL'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v2 = [team_m.loc['BOS'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v3 = [team_m.loc['BKN'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v4 = [team_m.loc['CHA'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v5 = [team_m.loc['CHI'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v6 = [team_m.loc['CLE'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v7 = [team_m.loc['DAL'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v8 = [team_m.loc['DEN'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v9 = [team_m.loc['DET'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v10 = [team_m.loc['GSW'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v11 = [team_m.loc['HOU'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v12 = [team_m.loc['IND'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v13 = [team_m.loc['LAC'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v14 = [team_m.loc['LAL'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v15 = [team_m.loc['MEM'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v16 = [team_m.loc['MIA'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v17 = [team_m.loc['MIL'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v18 = [team_m.loc['MIN'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v19 = [team_m.loc['NJN'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v20 = [team_m.loc['NOH'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v21 = [team_m.loc['NOP'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v22 = [team_m.loc['NYK'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v23 = [team_m.loc['OKC'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v24 = [team_m.loc['ORL'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v25 = [team_m.loc['PHI'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v26 = [team_m.loc['PHX'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v27 = [team_m.loc['POR'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v28 = [team_m.loc['SAC'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v29 = [team_m.loc['SAS'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v30 = [team_m.loc['TOR'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v31 = [team_m.loc['UTA'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
v32 = [team_m.loc['WAS'][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]


def radar_team1() -> Radar:
    c = (
        Radar(figsize)
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Games", max_=70, color='#000000'),
                opts.RadarIndicatorItem(name="Minutes", max_=40, color='#000000'),
                opts.RadarIndicatorItem(name="Total Rebounds", max_=6.5, color='#000000'),
                opts.RadarIndicatorItem(name="Assists", max_=6, color='#000000'),
                opts.RadarIndicatorItem(name="Steals", max_=2, color='#000000'),
                opts.RadarIndicatorItem(name="Blocks", max_=1, color='#000000'),
                opts.RadarIndicatorItem(name="Turnovers", max_=4, color='#000000'),
                opts.RadarIndicatorItem(name="Personal Fouls", max_=3.5, color='#000000'),
                opts.RadarIndicatorItem(name="Points", max_=34, color='#000000'),
            ],
            shape='circle',

        )

        .add('ATL', v1, linestyle_opts=opts.LineStyleOpts(color="#cc082f"), )
        .add('BOS', v2, linestyle_opts=opts.LineStyleOpts(color="#007a30"), )
        .add('BKN', v3, linestyle_opts=opts.LineStyleOpts(color="#2d2926"), )
        .add('CHA', v4, linestyle_opts=opts.LineStyleOpts(color="#f63f3a"), )
        .add('CHI', v5, linestyle_opts=opts.LineStyleOpts(color="#bc022b"), )
        .add('CLE', v6, linestyle_opts=opts.LineStyleOpts(color="#57263e"), )
        .add('DAL', v7, linestyle_opts=opts.LineStyleOpts(color="#0155ba"), )
        .add('DEN', v8, linestyle_opts=opts.LineStyleOpts(color="#498ecb"), )
        .add('DET', v9, linestyle_opts=opts.LineStyleOpts(color="#ee154b"), )
        .add('GSW', v10, linestyle_opts=opts.LineStyleOpts(color="#0168b3"), )
        .add('HOU', v11, linestyle_opts=opts.LineStyleOpts(color="#d00e42"), )
        .add('IND', v12, linestyle_opts=opts.LineStyleOpts(color="#00255d"), )
        .add('LAC', v13, linestyle_opts=opts.LineStyleOpts(color="#f1a8bb"), )
        .add('LAL', v14, linestyle_opts=opts.LineStyleOpts(color="#f1a8bb"), )
        .add('MEM', v15, linestyle_opts=opts.LineStyleOpts(color="#6189b9"), )
        .add('MIA', v16, linestyle_opts=opts.LineStyleOpts(color="#97002e"), )
        .add('MIL', v17, linestyle_opts=opts.LineStyleOpts(color="#e0d4b1"), )
        .add('MIN', v18, linestyle_opts=opts.LineStyleOpts(color="#00a950"), )
        .add('NJN', v19, linestyle_opts=opts.LineStyleOpts(color="#09285e"), )
        .add('NOH', v20, linestyle_opts=opts.LineStyleOpts(color="#0f5a82"), )
        .add('NOP', v21, linestyle_opts=opts.LineStyleOpts(color="#071f3f"), )
        .add('NYK', v22, linestyle_opts=opts.LineStyleOpts(color="#f68327"), )
        .add('OKC', v23, linestyle_opts=opts.LineStyleOpts(color="#f15032"), )
        .add('ORL', v24, linestyle_opts=opts.LineStyleOpts(color="#0071bb"), )
        .add('PHI', v25, linestyle_opts=opts.LineStyleOpts(color="#033ea6"), )
        .add('PHX', v26, linestyle_opts=opts.LineStyleOpts(color="#f9a11f"), )
        .add('POR', v27, linestyle_opts=opts.LineStyleOpts(color="#e93940"), )
        .add('SAC', v28, linestyle_opts=opts.LineStyleOpts(color="#6a47aa"), )
        .add('SAS', v29, linestyle_opts=opts.LineStyleOpts(color="#d6dcde"), )
        .add('TOR', v30, linestyle_opts=opts.LineStyleOpts(color="#bd0f33"), )
        .add('UTA', v31, linestyle_opts=opts.LineStyleOpts(color="#2c4e37"), )
        .add('WAS', v32, linestyle_opts=opts.LineStyleOpts(color="#002043"), )

        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(legend_opts=opts.LegendOpts(type_="scroll", selected_mode="single",
                                                     textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         title_opts=opts.TitleOpts(title="data-opp_team", pos_top='bottom',
                                                   title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         )

    )
    return c


a20 = [age_m.loc[20][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a21 = [age_m.loc[21][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a22 = [age_m.loc[22][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a23 = [age_m.loc[23][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a24 = [age_m.loc[24][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a25 = [age_m.loc[25][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a26 = [age_m.loc[26][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a27 = [age_m.loc[27][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a28 = [age_m.loc[28][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a29 = [age_m.loc[29][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a30 = [age_m.loc[30][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a31 = [age_m.loc[31][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]
a32 = [age_m.loc[32][['G', 'MP', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_list()]


def radar_team2() -> Radar:
    c = (
        Radar(figsize)
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Games", max_=82, color='#000000'),
                opts.RadarIndicatorItem(name="Minutes", max_=42, color='#000000'),
                opts.RadarIndicatorItem(name="Total Rebounds", max_=7, color='#000000'),
                opts.RadarIndicatorItem(name="Assists", max_=6.5, color='#000000'),
                opts.RadarIndicatorItem(name="Steals", max_=2.5, color='#000000'),
                opts.RadarIndicatorItem(name="Blocks", max_=1, color='#000000'),
                opts.RadarIndicatorItem(name="Turnovers", max_=6, color='#000000'),
                opts.RadarIndicatorItem(name="Personal Fouls", max_=3.5, color='#000000'),
                opts.RadarIndicatorItem(name="Points", max_=36, color='#000000'),
            ],
            shape='circle'
        )

        .add('20', a20, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('21', a21, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('22', a22, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('23', a23, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('24', a24, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('25', a25, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('26', a26, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('27', a27, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('28', a28, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('29', a29, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('30', a30, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )
        .add('31', a31, linestyle_opts=opts.LineStyleOpts(color="#663d7f"), )
        .add('32', a32, linestyle_opts=opts.LineStyleOpts(color="#fbc735"), )

        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(legend_opts=opts.LegendOpts(type_="scroll", selected_mode="single",
                                                     textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         title_opts=opts.TitleOpts(title="data-age", pos_top='bottom',
                                                   title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         )

    )
    return c


sunburst_data = [
    {
        "name": "Home",
        "itemStyle": {"color": "#fbc735"},
        'children': [
            {"name": "#WIN", "value": int(home_away['W/L.1'][0]), "itemStyle": {"color": "#feb846"}},
            {"name": "#LOSE", "value": int(home_away['W/L.1'][1]), "itemStyle": {"color": "#4c4632"}},
        ]
    },

    {
        "name": "Away",
        "itemStyle": {"color": "#663d7f"},
        'children': [
            {"name": "@WIN", "value": int(home_away['W/L.1'][2]), "itemStyle": {"color": "#feb846"}},
            {"name": "@LOSE", "value": int(home_away['W/L.1'][3]), "itemStyle": {"color": "#4c4632"}},
        ]
    }

]


def sunburst1() -> Sunburst:
    c = (
        Sunburst(figsize)
        .add(
            series_name="",
            data_pair=sunburst_data,

        )

        .set_global_opts(
            title_opts=opts.TitleOpts(title="Win-Lose", title_textstyle_opts=opts.TextStyleOpts(color='#000000')))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))

    )
    return c


team_list = win_lose_team.index.dropna().to_list()
win_list = win_lose_team[win_lose_team['W/L'] == 'W'].iloc[:, 1].to_list()
lose_list = win_lose_team[win_lose_team['W/L'] == 'L'].iloc[:, 1].to_list()
lose_list.insert(2, 0)
lose_list.insert(36, 0)
t_list = [win_list, lose_list]

heatmap1 = HeatMap(figsize)
value = [[i, j, int(t_list[j][i])] for i in range(32) for j in range(2)]
heatmap1.add_xaxis(team_list)
heatmap1.add_yaxis('', ['WIN', 'LOSE'], value)
heatmap1.set_global_opts(title_opts=opts.TitleOpts(title="Win-Lose-Team", pos_right='40%',
                                                   title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         xaxis_opts=opts.AxisOpts(
                             axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                         yaxis_opts=opts.AxisOpts(
                             axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                         visualmap_opts=opts.VisualMapOpts(max_=51, pos_right='right',
                                                           textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                         )

heatmap1.render_notebook()

age_list = win_lose_age.index.dropna().to_list()
list_win = win_lose_age[win_lose_age['W/L'] == 'W'].iloc[:, 1].to_list()
list_lose = win_lose_age[win_lose_age['W/L'] == 'L'].iloc[:, 1].to_list()
a_list = [list_win, list_lose]

heatmap2 = HeatMap(figsize)

value = [[i, j, int(a_list[j][i])] for i in range(13) for j in range(2)]
heatmap2.add_xaxis(age_list)
heatmap2.add_yaxis('', ['WIN', 'LOSE'], value)
heatmap2.set_global_opts(title_opts=opts.TitleOpts(title="Win-Lose-Age", pos_right='40%',
                                                   title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         xaxis_opts=opts.AxisOpts(
                             axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                         yaxis_opts=opts.AxisOpts(
                             axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                         visualmap_opts=opts.VisualMapOpts(max_=65, pos_right='right',
                                                           textstyle_opts=opts.TextStyleOpts(color='#000000')),
                         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                         )

heatmap2.render_notebook()

adata = s_age.reset_index()
aG = np.array(adata[['AgeY', 'G']]).tolist()

pie1 = Pie(figsize)

pie1.add('', aG, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie1.set_global_opts(title_opts=opts.TitleOpts(title='Games', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie1.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie1.render_notebook()

aPTS = np.array(adata[['AgeY', 'PTS']]).tolist()

pie2 = Pie(figsize)

pie2.add('', aPTS, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie2.set_global_opts(title_opts=opts.TitleOpts(title='Points', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie2.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie2.render_notebook()

aAST = np.array(adata[['AgeY', 'AST']]).tolist()

pie3 = Pie(figsize)

pie3.add('', aAST, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie3.set_global_opts(title_opts=opts.TitleOpts(title='Assists', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie3.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie3.render_notebook()
#
aTRB = np.array(adata[['AgeY', 'TRB']]).tolist()

pie4 = Pie(figsize)

pie4.add('', aTRB, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie4.set_global_opts(title_opts=opts.TitleOpts(title='Total Rebounds', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie4.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie4.render_notebook()
#
aSTL = np.array(adata[['AgeY', 'STL']]).tolist()

pie5 = Pie(figsize)

pie5.add('', aSTL, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie5.set_global_opts(title_opts=opts.TitleOpts(title='Steals', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie5.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie5.render_notebook()
#
aBLK = np.array(adata[['AgeY', 'BLK']]).tolist()

pie6 = Pie(figsize)

pie6.add('', aBLK, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie6.set_global_opts(title_opts=opts.TitleOpts(title='Blocks', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll",
                                                 textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie6.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie6.render_notebook()
#
aTOV = np.array(adata[['AgeY', 'TOV']]).tolist()

pie7 = Pie(figsize)

pie7.add('', aTOV, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie7.set_global_opts(title_opts=opts.TitleOpts(title='Turnovers', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll",
                                                 textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie7.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie7.render_notebook()

aPF = np.array(adata[['AgeY', 'PF']]).tolist()

pie8 = Pie(figsize)

pie8.add('', aPF, rosetype='area', radius=["24%", "75%"], label_opts=opts.LabelOpts(is_show=False))
pie8.set_global_opts(title_opts=opts.TitleOpts(title='Personal Fouls', pos_top='bottom', pos_left='1%',
                                               title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                     legend_opts=opts.LegendOpts(type_="scroll",
                                                 textstyle_opts=opts.TextStyleOpts(color='#000000'))
                     )
pie8.set_series_opts(label_opts=opts.LabelOpts(position="outside",
                                               formatter="{per|{b} year-old: {c}}",
                                               background_color="#eee",
                                               border_color="#aaa",
                                               border_width=1,
                                               border_radius=4,
                                               rich={
                                                   "per": {
                                                       "color": "#eee",
                                                       "backgroundColor": "#334455",
                                                       "padding": [2, 4],
                                                       "borderRadius": 2,
                                                   },
                                               },
                                               ))
pie8.render_notebook()

timeline = Timeline(figsize)
timeline.add(pie1, "Games")
timeline.add(pie2, "Points")
timeline.add(pie3, "Assists")
timeline.add(pie4, "Total Rebounds")
timeline.add(pie5, "Steals")
timeline.add(pie6, "Blocks")
timeline.add(pie7, "Turnovers")
timeline.add(pie8, "Personal Fouls")

timeline.add_schema(
    symbol='diamond',
    symbol_size=8,
    play_interval=1500,
    is_auto_play=True,
    label_opts=opts.LabelOpts(is_show=False),
    itemstyle_opts=opts.ItemStyleOpts(color='#000000')
)

timeline.render_notebook()

bar = Bar(figsize)
bar.add_xaxis(s_team.index.tolist())

bar.add_yaxis('Games', s_team.G.tolist(), yaxis_index=1, color='#0cb3a9')
bar.add_yaxis('Points', s_team.PTS.tolist(), yaxis_index=0, color='#0071bb')
bar.add_yaxis('Assists', s_team.AST.tolist(), yaxis_index=0, color='#f1a8bb')
bar.add_yaxis('Total Rebounds', s_team.TRB.tolist(), yaxis_index=0, color='#663d7f')
bar.add_yaxis('Steals', s_team.STL.tolist(), yaxis_index=0, color='#cc082f')
bar.add_yaxis('Blocks', s_team.BLK.tolist(), yaxis_index=0, color='#4c4632')
bar.add_yaxis('Turnovers', s_team.TOV.tolist(), yaxis_index=0, color='#feb846')
bar.add_yaxis('Personal Fouls', s_team.PF.tolist(), yaxis_index=0, color='#007a30')

bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

bar.extend_axis(
    yaxis=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))))
bar.set_global_opts(title_opts=opts.TitleOpts(title='Team-Sum', pos_bottom='bottom', pos_right='40%', padding=[1, 5],
                                              title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                    xaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                    yaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                    datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                    legend_opts=opts.LegendOpts(type_="scroll", textstyle_opts=opts.TextStyleOpts(color='#000000'),
                                                selected_mode="single"),
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis",
                                                  textstyle_opts=opts.TextStyleOpts(color='#000000')),

                    )

bar.render_notebook()

line1 = Line(figsize)
line1.add_xaxis(team_m.index.to_list())

line1.add_yaxis('FGP', team_m.FGP.to_list(), yaxis_index=0, color='#007a30')
line1.add_yaxis('3PP', team_m['3PP'].to_list(), yaxis_index=0, color='#663d7f')
line1.add_yaxis('FTP', team_m.FTP.to_list(), yaxis_index=0, color='#feb846')

line1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

line1.extend_axis(
    yaxis=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))))
line1.set_global_opts(title_opts=opts.TitleOpts(title='Shooting-Team', pos_left='right', padding=[1, 5],
                                                title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                      xaxis_opts=opts.AxisOpts(
                          axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                      yaxis_opts=opts.AxisOpts(
                          axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                      legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color='#000000')),
                      tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis",
                                                    textstyle_opts=opts.TextStyleOpts(color='#000000')),
                      datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],

                      )

line1.render_notebook()

line2 = Line(figsize)
line2.add_xaxis(age_m.index.astype(str).to_list())
line2.add_yaxis('FGP', age_m.FGP.to_list(), yaxis_index=0, color='#007a30')
line2.add_yaxis('3PP', age_m['3PP'].to_list(), yaxis_index=0, color='#663d7f')
line2.add_yaxis('FTP', age_m.FTP.to_list(), yaxis_index=0, color='#feb846')
line2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
line2.set_global_opts(title_opts=opts.TitleOpts(title='Shooting-Age', pos_left='top', padding=[1, 5],
                                                title_textstyle_opts=opts.TextStyleOpts(color='#000000')),
                      xaxis_opts=opts.AxisOpts(
                          axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                      yaxis_opts=opts.AxisOpts(
                          axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color='#000000'))),
                      legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color='#000000')),
                      tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis",
                                                    textstyle_opts=opts.TextStyleOpts(color='#000000')),

                      )

line2.render_notebook()

title = Pie().set_global_opts(title_opts=opts.TitleOpts(title="保罗乔治数据大屏",
                                                        title_textstyle_opts=opts.TextStyleOpts(font_size=40,
                                                                                                ),
                                                        pos_top=0))
title.render_notebook()

page = Page(layout=Page.DraggablePageLayout, page_title='保罗乔治数据大屏')
page.add(
    scatter_paul_george(),
    radar_team1(),
    radar_team2(),
    sunburst1(),
    heatmap1,
    heatmap2,
    timeline,
    bar,
    line1,
    line2,
    title
)
page.render('临时.html')  # 执行完毕后，打开临时html并排版，排版完成后点击Save Config，把json文件放到本目录下
print("生成完毕:临时.html")
