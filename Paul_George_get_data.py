import pandas as pd
from nba_api.stats.endpoints import playercareerstats, shotchartdetail, playergamelog
from nba_api.stats.static import players


def get_player_shotchartdetail(player_name):
    # player dictionary
    player_dict = ''
    nba_players = players.get_players()
    for player in nba_players:
        if player['full_name'] == player_name:
            player_dict = player

    # career dataframe
    career = playercareerstats.PlayerCareerStats(player_id=str(player_dict['id'])).get_dict()
    career_column = career['resultSets'][0]['headers']
    career_df = pd.DataFrame(career['resultSets'][0]['rowSet'], columns=career_column)

    # season_id
    num = 0
    id_season = career_df['SEASON_ID']
    for season_id in id_season:
        # team id during the season
        team_id = career_df[career_df['SEASON_ID'] == season_id]['TEAM_ID']

        # shotchartdetail endpoints
        shotchartlist = shotchartdetail.ShotChartDetail(team_id=int(team_id),
                                                        player_id=int(player_dict['id']),
                                                        season_type_all_star='Regular Season',
                                                        season_nullable=season_id,
                                                        context_measure_simple="FGA").get_data_frames()
        if num == 0:
            shot_detail_player = pd.DataFrame(shotchartlist[0])  # 输出csv文件
            shot_detail_player.to_csv('./data/paul_george_shot_xy.csv', mode='w', encoding='utf-8')
        else:
            shot_detail_player = pd.DataFrame(shotchartlist[0])
            shot_detail_player.to_csv('./data/paul_george_shot_xy.csv', mode='a', header=False, encoding='utf-8')
        num += 1
        print(num)

        numm = 0
        for season in career_df['SEASON_ID']:
            df = playergamelog.PlayerGameLog(player_id=str(player_dict['id']),
                                             season=season,
                                             season_type_all_star='Regular Season').get_data_frames()
            df1 = pd.DataFrame(df[0])
            if numm == 0:
                df1.to_csv('./data/career_data.csv', mode='w', encoding='utf-8')
            else:
                df1.to_csv('./data/career_data.csv', mode='a', header=False, encoding='utf-8')
            numm += 1


if __name__ == '__main__':
    get_player_shotchartdetail("Paul George")
    print('data get success')
