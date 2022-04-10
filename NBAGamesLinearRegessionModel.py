import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
###SKLEARN###
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


all_teams = ['Atlanta Hawks', 'Brooklyn Nets', 'Boston Celtics', 'Chicago Bulls', 'Cleveland Cavaliers', 'Charlotte Hornets', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Phoenix Suns', 'Philadelphia 76ers', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards']
team_abv = ['ATL', 'BRK', 'BOS', 'CHI', 'CLE', 'CHO', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHO', 'PHI', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
#playoff_teams = ['Atlanta Hawks', 'Brooklyn Nets', 'Boston Celtics', 'Chicago Bulls', 'Charlotte Hornets', 'Dallas Mavericks', 'Denver Nuggets', 'Golden State Warriors', 'Los Angeles Clippers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'Phoenix Suns', 'Philadelphia 76ers', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz']
#playoff_east = ['CHO', 'ATL', 'BRK', 'CLE', 'CHI', 'TOR', 'PHI', 'BOS', 'MIL', 'MIA']
#playoff_west = ['SAS', 'NOP', 'LAC', 'MIN', 'DEN', 'UTA', 'DAL', 'GSW', 'MEM', 'PHO']


team_avgs = 'https://www.basketball-reference.com/leagues/NBA_2022.html' #df[0]
team_per_game = 'https://www.basketball-reference.com/teams/{team}/2022/gamelog/' #df[4]

##############SCRAP/CLEAN DATA###############
url = team_avgs.format()
print(url)
df = pd.read_html(url, header = 0)
df = df[4]
    
for i in range(0, len(all_teams)):
    df = df.replace(to_replace = all_teams[i],
                 value = team_abv[i])
    df = df.replace(to_replace = all_teams[i] + '*',
             value = team_abv[i])
df = df.drop(columns = df.columns[0])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[1])
df = df.drop(columns = df.columns[2])
df = df.drop(columns = df.columns[2])
df = df.drop(columns = df.columns[2])
df = df.drop(columns = df.columns[2])
df = df.drop(columns = df.columns[2])
df = df.drop(columns = df.columns[5])
df = df.drop(columns = df.columns[5])
df = df.drop(columns = df.columns[6])
df = df.drop(columns = df.columns[7])
df = df.drop(columns = df.columns[7])
df = df.drop(columns = df.columns[4])
    
df = df.drop(30)
TeamAverages = df
print(df)
#############################################
prev_df = df
count = 0
for team in team_abv:
    url = team_per_game.format(team = team)
    
    df = pd.read_html(url, header = 0)
    df = df[0]
    
    df = df.drop(columns = df.columns[0])
    df = df.drop(columns = df.columns[0])
    df = df.drop(columns = df.columns[0])
    df = df.drop(columns = df.columns[0])
    df = df.drop(columns = df.columns[0])
    df = df.drop(columns = df.columns[3])
    df = df.drop(columns = df.columns[3])
    df = df.drop(columns = df.columns[3])
    df = df.drop(columns = df.columns[3])
    df = df.drop(columns = df.columns[3])
    df = df.drop(columns = df.columns[4])
    df = df.drop(columns = df.columns[4])
    df = df.drop(columns = df.columns[6])
    df = df.drop(columns = df.columns[6])
    df = df.drop(columns = df.columns[7])
    df = df.drop(columns = df.columns[8])
    df = df.drop(columns = df.columns[9])
    df = df.drop(columns = df.columns[9])
    df = df.drop(columns = df.columns[9])
    df = df.drop(columns = df.columns[9])
    df = df.drop(columns = df.columns[9])
    df = df.drop(columns = df.columns[10])
    df = df.drop(columns = df.columns[10])
    df = df.drop(columns = df.columns[12])
    df = df.drop(columns = df.columns[12])
    df = df.drop(columns = df.columns[13])
    df = df.drop(columns = df.columns[14])
    df = df.drop(columns = df.columns[8])
    df = df.drop(columns = df.columns[0]) #bring back W/L
    
    df = df.drop(0)
    
    df.columns = ["team_pts", "opp_pts", "team_3P%", "team_FT%", 'team_ORB', 
                  'team_STL', 'team_TOV', 'opp_3P%', 'opp_FT%', 'opp_ORB', 'opp_STL', 'opp_TOV']
    
    if count > 0:
        df = pd.concat([prev_df, df], ignore_index = True)
    prev_df = df
    count += 1
    if count == 30:
        PerGame = df
PerGame = PerGame[PerGame.opp_TOV != 'Opponent']
PerGame = PerGame[PerGame.opp_TOV != 'TOV']
##############CREATE/TRAIN/TEST LINEAR MODEL###############

X = PerGame
X = X.drop(columns = df.columns[0:2])
Y = PerGame
Y = Y.drop(columns = df.columns[2:14])
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

model = linear_model.LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercpet:', model.intercept_)
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coeffient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))

##############PUT IN AWAY & HOME INTO LISTS###############
Y_test = np.asarray(Y_test, dtype = 'float64')

Y_test_home = np.delete(Y_test, 0, axis = 1)
Y_pred_home = np.delete(Y_pred, 0, axis = 1)
Y_t_h = []
Y_p_h = []
for i in range(0, 432):
    Y_t_h.append(Y_test_home[i][0])
    
for i in range(0, 432):
    Y_p_h.append(Y_pred_home[i][0])



Y_test = np.delete(Y_test, 1, axis = 1)
Y_pred = np.delete(Y_pred, 1, axis = 1)
Y_t = []
Y_p = []
print(Y_test[0][0])
for i in range(0, 432):
    Y_t.append(Y_test[i][0])
    
for i in range(0, 432):
    Y_p.append(Y_pred[i][0])

##############SCATTER PLOTS###############
Away = plt.figure(1)
plt.scatter(Y_t, Y_p, alpha = 0.5, color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Actual vs Predicted Away Team Score")
plt.ylim(75, 150)
plt.xlim(75, 150)
plt.tight_layout()
Away.show()

Home = plt.figure(2)
plt.scatter(Y_t_h, Y_p_h, alpha = 0.5, color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Actual vs Predicted Home Team Score")
plt.ylim(75, 150)
plt.xlim(75, 150)
plt.tight_layout()
Home.show()
    
    


    