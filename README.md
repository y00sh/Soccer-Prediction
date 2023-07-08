# Soccer Match Prediction Using Machine Learning
Predicting JLeague matches usings particle swarm optimization and XGBoost

Inspired by the feature engineering method, _rating feature learning_ from the paper "Incorporating domain knowledge in machine Learning for soccer outcome prediction" by Daniel Berrar, Philippe Lopes and Werner Dubitzky 2017.  (Berrar 2017)

Our goal is the find the probabilities of a home team win, tie or an away team win. The three probabilities will add up to 100%. 

First we gather the information: we need match data containing team name, score, and xG for both the home and away team. I get my data from Sporteria but the earliest they started to collect xG data is 2019. I have data for all the teams in J1 and J2 League :jp: since the 2019 season, There are currently 63 teams in my model and +4000 matches as of the 2023 season. When teams are relegated from J2 to J3, their data is not deleted from the model but since they don't have matches in J1 or J2 their ratings/features don't get updated. When teams from J3 get promoted their match data is added as the season progresses and may be inaccurate for their first few games.

Table 1. Information required

|Date|Home|$home_{goals}$|$home_{xG}$|Away|$away_{goals}$|$away_{xG}$|
|----------|----------------|----|---------|-------------|----|---------|
| 2/22/19  | Cerezo Osaka  | 1  |  1.208  | Vissel Kobe | 0  |  1.299  |
| 2/23/19  | Sagan Tosu    | 0  |  0.845  |  Grampus    | 4  |  1.777  |
| 2/23/19  | Vegalta Sendai | 0 |  0.479  | Urawa Reds  | 0  |  0.922  |

First, we define four quantitative features that capture a team’s _performance rating_ in terms of its _ability_ to score goals and _inability_ to prevent goals at both the home and away venues, respectively:

- _Home attacking strength_ reflects a team’s ability to score goals at its _home_ venue—the higher the value, the higher the strength.  
- _Home defensive weakness_ reflects a team’s _inability_ to prevent goals by the opponent at its _home_ venue—the higher the value, the higher the weakness.  
- _Away attacking strength_ reflects a team’s ability to score goals at the opponent’s venue— the higher the value, the higher the strength.  
- _Away defensive weakness_ reflects a team’s _inability_ to prevent goals by the opponent at the opponent’s venue—the higher the value, the higher the weakness.  

$$\text{Eq. 3: }T_{hatt}^{t+1} = T_{hatt}^{t} +ω_{hatt}(g_{h} − \hat{g}_h)$$

$$\text{Eq. 4: }T_{hdef}^{t+1} = T_{hdef}^{t} +ω_{hdef}(g_{a} − \hat{g}_a)$$

Where <br>
$T_{hatt}^{t+1}$ is the new home attacking strength of _T_ after match. $T_{hatt}^{t+1} \in ℝ$ <br>
$T_{hatt}^{t}$ is the previous home attacking strength of _T_ before match. $T_{hatt}^{t} \in ℝ$  <br>
$T_{hdef}^{t+1}$ is the new home defensive weakness of _T_ after match. $T_{hdef}^{t+1} \in ℝ$  <br>
$T_{hdef}^{t}$ the previous home defensive weakness of _T_ before match. $T_{hdef}^{t} \in ℝ$  <br>
$\omega_{hatt}$ is the update weight for home attacking strength. $\omega_{hatt} \in ℝ^+$  <br>
$\omega_{hdef}$ is the update weight for home defensive weakness. $\omega_{hdef} \in ℝ^+$  <br>
$g_h$, $g_a$ are the observed goals scored by home/away team. $g_h, g_a \in ℕ_0$  <br>
$\hat{g}_h$, $\hat{g}_a$ are the predicted goals scored by home/away team. $\hat{g}_h, \hat{g}_a \in ℝ_0^+$  <br>


$$\text{Eq. 5: }T_{aatt}^{t+1} = T_{aatt}^{t} +ω_{aatt}(g_{a} − \hat{g}_a)$$

$$\text{Eq. 6: }T_{adef}^{t+1} = T_{adef}^{t} +ω_{adef}(g_{h} − \hat{g}_h)$$

Where <br>
$T_{aatt}^{t+1}$ is the new away attacking strength of _T_ after match. $T_{aatt}^{t+1} \in ℝ$ <br>
$T_{aatt}^{t}$ is the previous away attacking strength of _T_ before match. $T_{aatt}^{t} \in ℝ$ <br>
$T_{adef}^{t+1}$ is the new away defensive weakness of _T_ after match. $T_{adef}^{t+1} \in ℝ$ <br>
$T_{adef}^{t}$ the previous away defensive weakness of _T_ before match. $T_{adef}^{t} \in ℝ$ <br>
$\omega_{aatt}$ is the update weight for away attacking strength. $\omega_{aatt} \in ℝ^+$ <br>
$\omega_{adef}$ is the update weight for away defensive weakness. $\omega_{adef} \in ℝ^+$ <br>

Based on these four performance rating features (per team), Eqs. 1 and 2 define a _goal- prediction model_ that predicts the _goals_ scored by the home and away team, respectively.  <br>  


$$\text{Eq. 1: } \hat{g}_h(H_{hatt},A_{adef})= \frac{\alpha_h}{1+\exp(-\beta_h(H_{hatt}+A_{adef})-\gamma_h)}$$ <br>

$$\text{Eq. 2: }\hat{g}_a(A_{aatt},H_{hdef})= \frac{\alpha_a}{1+\exp(-\beta_a(A_{aatt}+H_{hdef})-\gamma_a)}$$

where <br>
$H_{hatt}$ are the home team’s attacking strength in home games. $H_{hatt} \in ℝ$ <br>
$H_{hdef}$ are the home team’s defensive weakness in home games. $H_{hdef} \in ℝ$ <br>
$A_{aatt}$ are the away team’s attacking strength in away games. $A_{aatt} \in ℝ$ <br>
$A_{adef}$ are the away team’s defensive weakness in away games. $A_{adef} \in ℝ$ <br>
$\alpha_h,\alpha_a$ are constants defining maximum for $\hat{g}_h, \hat{g}_a$. $\alpha_h,\alpha_a \in ℝ^+$
 
$\beta_h,\beta_a$ are constants defining steepness of sigmoidal curves. $\beta_h,\beta_a \in ℝ_{0}^+$ <br>
$\gamma_h,\gamma_a$ are constants defining the curves’ threshold point. $\gamma_h,\gamma_a \in ℝ$ <br>  

After every match the team's _performance rating_ are updated depending on whether they were the home or away team using Eq 3 - 6. Those ratings are then used to create the predicted goals for the next match on the schedule with Eq 1 and 2.  

Table 2: Performance Rating 

|   Team        |   H_hatt    |   H_hdef    |   A_aatt    |   A_adef    |
|-------------- |------------|-------------|-------------|-------------|
| Cerezo Osaka  |  0.8606843 |  -7.532469  | -1.898231   | -10.148069  |
| Sagan Tosu    | -3.6531719 |  -1.244026  | -6.971303   |  -4.820117  |
| Vegalta Sendai| -1.3261847 |  -1.327007  |  4.644055   |  -5.945538  |

Equation 1 - 6 requires many parameters beyond the given information in Table 1. At the beginning of my model all of the teams home/away defensive and attacking strength started at zero. To calculate the missing parameters we use individual goal-prediction error, $\epsilon_g$ in equation 7-9.  Note this differs from Berrar 2017 in that I use the expected goals, xG instead of only the observed goals. 

$$\text{Eq. 7: }\epsilon_g=\frac{1}{2}[(g_h-\hat{g}_h)^2+(g_a-\hat{g}_a)^2]$$

$$\text{Eq. 8: }g_h = (home_{xG} \times \rho) + (home_{goals} \times (1-\rho))$$

$$\text{Eq. 9: }g_a = (away_{xG} \times \rho) + (away_{goals} \times (1-\rho))$$

Using particle swarm optimization (PSO) we find the 11 missing parameters by adjusting the parameters until the individual goal-prediction error, $\epsilon_g$ is minimized. I used 80 particles and to help with computation we defined a lower and upper bound to search for each parameter. Here were the parameters values that I found:

$\beta_h = 0.02539$ <br>
$\beta_a = 0.03$ <br>
$\gamma_h = -0.6711$ <br>
$\gamma_a = -0.7728$ <br>
$\omega_{hatt} = 2.1694$ <br>
$\omega_{hdef} = 1.7701$ <br>
$\omega_{aatt} = 1.3964$ <br>
$\omega_{adef} = 2.4794$ <br>
$\alpha_h = 4.2$ <br>
$\alpha_a = 4.0758$ <br>
$\rho = 0.876$ <br>

After running our PSO code with all of the historical matches we should have a table with each teams ratings and which the outcome of the match

| home_H_hatt_mix | home_H_hdef_mix | home_A_aatt_mix | home_A_adef_mix | away_H_hatt_mix | away_H_hdef_mix | away_A_aatt_mix | away_A_adef_mix |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|   -0.51819662   |   -0.2692562    |   -0.2124204    |   -0.59226160   |   -0.51819662   |   -0.2692562    |   -0.2124204    |   -0.59226160   |
|   -1.47981522   |    1.3623530    |    1.0747815    |   -1.69132273   |   -1.47981522   |    1.3623530    |    1.0747815    |   -1.69132273   |
|   -2.17377591   |   -0.8524936    |   -0.6725455    |   -2.48447006   |   -2.17377591   |   -0.8524936    |   -0.6725455    |   -2.48447006   | 
|   0.00891013    |   -0.4438178    |   -0.3501348    |    0.01018364   |   0.00891013    |   -0.4438178    |   -0.3501348    |    0.01018364   |  
|   -0.49386014   |   -1.2284265    |   -0.9691248    |   -0.56444675   |   -0.49386014   |   -1.2284265    |   -0.9691248    |   -0.56444675   |
|   -0.43666572   |   -0.3803888    |   -0.3000946    |   -0.49907762   |   -0.43666572   |   -0.3803888    |   -0.3000946    |   -0.49907762   |

These 8 home/away attack and defense rating for each team will be our features into XGBoost.  

Our machine learning will gradient boost and attempt to minimize the error when classifying the results into home win, away win or draw. XGboost only reads and writes numerical vectors. The 8 features are turned into a matrix and split randomly 80% - 20% for the training and test set.  The accompanying score line used to update the home and away teams attack/defense ratings are turned into a single columns with the result 0, 1, 2 representing the numerical value for a   This column is the target values that XGBoost will try and predict in the training and test set. 

I fine tuned my XGBoost parameters and found:

Feature subsampling = 1.0
Learning rate = 0.025
Maximum tree depth = 5
Training set subsample = 0.67

I saw that with a max tree depth of 5, I'd see the optimal learning rate at around 100 rounds so I set my total rounds to 150.

As you plot the error versus a range of the parameter you are trying to find tune, you should see a parabolic or convex shape to the curve, where the error is minimized. This is how you fine tune your parameters
<p align="center">
  <img width="400" height="300" src="https://github.com/y00sh/Soccer-Prediction/assets/90585099/3857c6d0-8a83-47f7-92fd-3fb84945574f">
</p>

The error in our case is found by minimizing the Ranked Probability Score. 

$$RPS = \frac{1}{N-1} \sum_{i=1}^{N-1} (F_{i} - O_{i})^{2}$$

$$F_{i} = \sum_{j=1}^{i} p_{j}$$

$$O_{i} = \sum_{j=1}^{i} o_{j}$$

Where:
- $N$ is the total number of outcomes.
- $p_{j}$ is the predicted probability for outcome $j$.
- $o_{j}$ is the observed outcome (1 if $j$ is the true outcome, 0 otherwise).

The code for this is in the excel files. 

There isn't a lot published soccer predictions beyond gambling odds. [FiveThirtyEight](https://projects.fivethirtyeight.com/soccer-predictions/j1-league/) used to (stopped in June 2023) so I have compared my predictions with theirs. They use a Poisson regression which has been explained quite well by [Opisthokonta](https://opisthokonta.net/?p=296) and [StatsandSnakeOil](https://www.statsandsnakeoil.com/2019/01/06/predicting-the-premier-league-with-dixon-coles-and-xg/) 

One downside of Poisson is the time period in which you evaluate the teams performance. WIthout PSO you have to set the weight yourself or use the entire period of your data.  [Time-Weight example](https://opisthokonta.net/?p=1013). The PSO tries to optimize the weighting of historical performance but I have seen that it is slower then Poisson even as I tried to maximize $\alpha_h$ and $\alpha_a$.  Over the long term which is more accurate? Is a slower update of performance more accurate since humans are biased toward recent matches. Or will the machine learning model update too slowly as players get injured/transferred and managers change?  We will have to see. If I had more time an more access to data I think incorporating player data such as the starting lineup as a weight of the teams performance ratings could increase the accuracy. This data would most likely be from Wyscout or Opta which I do not pay for at the moment.  

But the main downside of Poisson is that draws are undervalued. Dixon-Coles attempts to fix this but if you look at FiveThirtyEight Jleague prediction the probability of a tie will always be lower than my model. 

I found the approach to predict soccer outcomes from Berrar 2017 unique and interesting. I'll continue to update the predictions to compare the performance against a Poisson model. I also hope you watch more Jleague soccer as it's one of the more entertaining leagues to watch. More parity than Big 5 leagues, and more tactical skills than MLS. 
