import pandas as pd


def get_data():
    def get_kv():
        kv = {
            'GAME_DATE': 'DATE',
            'Unnamed: 0': 'G',
            'MATCHUP': 'Tm',
            'WL': 'W/L',
            'MIN': 'MP',
            'FGM': 'FG',
            'FGA': 'FGA',
            'FG_PCT': 'FGP',
            'FG3M': '3P',
            'FG3A': '3PA',
            'FG3_PCT': '3PP',
            'FTM': 'FT',
            'FTA': 'FTA',
            'FT_PCT': 'FTP',
            'OREB': 'ORB',
            'DREB': 'DRB',
            'REB': 'TRB',
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TOV': 'TOV',
            'PF': 'PF',
            'PTS': 'PTS',
            'PLUS_MINUS': '+/-'
        }

        return kv

    def get_sort():
        ls = ['DATE', 'G', 'Tm', 'GL', 'Opp', 'W/L', 'MP', 'FG', 'FGA', 'FGP', '3P', '3PA', '3PP',
              'FT', 'FTA', 'FTP', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
              'PF', 'PTS', '+/-']
        return ls

    def format(path):
        data = pd.read_csv(path, encoding='utf-8')
        kv = get_kv()
        data = data[kv.keys()]
        age = data['GAME_DATE'].str.split(' ').str[-1]
        age = age.astype('int')
        age = age - 2011 + 21
        for i in kv.keys():
            data.rename(columns={i: kv[i]}, inplace=True)

        MATCHUP = data['Tm'].str.split(' ')
        data['Tm'], data['GL'], data['Opp'] = MATCHUP.str[0], MATCHUP.str[1], MATCHUP.str[2]
        data['G'] += 1
        new_data = data[get_sort()]
        new_data['AgeY'] = age
        new_data.insert(2, 'AgeY', new_data.pop('AgeY'))

        return new_data

    path = './data/career_data.csv'
    format(path)
    f_data = format(path)
    with pd.ExcelWriter(r'./data/george.xlsx') as writer:
        f_data.to_excel(writer, sheet_name='Regular')  # 保存数据

    # 数据读取
    Regular = pd.read_excel('./data/george.xlsx', 'Regular', index_col=0)

    # 数据分析
    # 1.场均贡献
    p_age = round(Regular.groupby(['AgeY'])['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'].mean(), 1)
    # 2.出场数及平均上场时间
    g_age = Regular.groupby(['AgeY'])['G'].count()
    m_age = round(Regular.groupby(['AgeY'])['MP'].mean(), 1)
    g_age = pd.concat([g_age, m_age], axis=1)
    # 3.场均命中率
    f_age = round(Regular.groupby(['AgeY'])['FG', 'FGA', '3P', '3PA', 'FT', 'FTA'].mean(), 1)

    f_age['FGP'] = round(Regular.groupby(['AgeY'])['FG'].sum() / Regular.groupby(['AgeY'])['FGA'].sum(), 3)
    f_age['3PP'] = round(Regular.groupby(['AgeY'])['3P'].sum() / Regular.groupby(['AgeY'])['3PA'].sum(), 3)
    f_age['FTP'] = round(Regular.groupby(['AgeY'])['FT'].sum() / Regular.groupby(['AgeY'])['FTA'].sum(), 3)

    f_age.insert(2, 'FGP', f_age.pop('FGP'))
    f_age.insert(5, '3PP', f_age.pop('3PP'))

    # 4.合并数据
    age_m = pd.concat([g_age, f_age, p_age], axis=1)

    # 输or赢 & 主or客
    # 1.各年龄段输赢数据
    win_lose_age = Regular.groupby(['AgeY'])['W/L'].value_counts()
    # 2.主客场输赢数据
    home_away_age = Regular.groupby(['AgeY', 'GL'])['W/L'].value_counts()
    home_away = Regular.groupby(['GL'])['W/L'].value_counts()

    # 对阵各队数据
    # 1.场均贡献
    p_team = round(Regular.groupby(['Opp'])['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'].mean(), 1)
    # 2.出场数及平均上场时间
    g_team = Regular.groupby(['Opp'])['G'].count()
    m_team = round(Regular.groupby(['Opp'])['MP'].mean(), 1)
    g_team = pd.concat([g_team, m_team], axis=1)
    # 3.场均命中率
    f_team = round(Regular.groupby(['Opp'])['FG', 'FGA', '3P', '3PA', 'FT', 'FTA'].mean(), 1)

    f_team['FGP'] = round(Regular.groupby(['Opp'])['FG'].sum() / Regular.groupby(['Opp'])['FGA'].sum(), 3)
    f_team['3PP'] = round(Regular.groupby(['Opp'])['3P'].sum() / Regular.groupby(['Opp'])['3PA'].sum(), 3)
    f_team['FTP'] = round(Regular.groupby(['Opp'])['FT'].sum() / Regular.groupby(['Opp'])['FTA'].sum(), 3)

    f_team.insert(2, 'FGP', f_team.pop('FGP'))
    f_team.insert(5, '3PP', f_team.pop('3PP'))

    # 4.合并数据
    team_m = pd.concat([g_team, f_team, p_team], axis=1)

    # 输or赢 & 主or客
    # 1.对阵各队输赢数据
    win_lose_team = Regular.groupby(['Opp'])['W/L'].value_counts()
    # 2.主客场输赢数据
    home_away_team = Regular.groupby(['Opp', 'GL'])['W/L'].value_counts()

    # 各年龄段数据总和
    s_age = Regular.groupby(['AgeY'])['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'].sum()
    s_age = pd.concat([Regular.groupby(['AgeY'])['G'].count(), s_age], axis=1)
    # 对阵各队数据总和
    s_team = Regular.groupby(['Opp'])['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'].sum()
    s_team = pd.concat([Regular.groupby(['Opp'])['G'].count(), s_team], axis=1)
    # 最大正负值分差
    Regular['+/-'] = Regular['+/-'].astype(int)
    pn_age = pd.DataFrame(Regular.groupby(['AgeY'])['+/-'].max()).rename(columns={'+/-': 'P_MAX'})
    sp_age = pd.concat([Regular.groupby(['AgeY'])['+/-'].min(), pn_age], axis=1)
    sp_age = sp_age.rename(columns={'+/-': 'N_MIN'})

    with pd.ExcelWriter(r'./data/Regular_Age.xlsx') as writer1:
        age_m.to_excel(writer1, sheet_name='data_age')
        win_lose_age.to_excel(writer1, sheet_name='win_lose_age')
        home_away_age.to_excel(writer1, sheet_name='home_away_age')
        home_away.to_excel(writer1, sheet_name='home_away')
        sp_age.to_excel(writer1, sheet_name='dff_point')
        s_age.to_excel(writer1, sheet_name='age_sum')

    # 球队
    pn_team = pd.DataFrame(Regular.groupby(['Opp'])['+/-'].max()).rename(columns={'+/-': 'P_MAX'})
    sp_team = pd.concat([Regular.groupby(['Opp'])['+/-'].min(), pn_team], axis=1)
    sp_team = sp_team.rename(columns={'+/-': 'N_MIN'})

    with pd.ExcelWriter(r'./data/Regular_Team.xlsx') as writer2:
        team_m.to_excel(writer2, sheet_name='data_team')
        win_lose_team.to_excel(writer2, sheet_name='win_lose_team')
        home_away_team.to_excel(writer2, sheet_name='home_away_team')
        home_away.to_excel(writer2, sheet_name='home_away')
        sp_team.to_excel(writer2, sheet_name='dff_point')
        s_team.to_excel(writer2, sheet_name='team_sum')
    # 数据总和
    regular_sum = Regular[['TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].sum()


if __name__ == '__main__':
    get_data()
    print('data get success')
