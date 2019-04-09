# nba-final4-projections

**Goal:** Predict the probability of an NBA team finishing the regular season in the top 8 (of 30) teams based on their players.

- Why top 8? Every team that has won a championship in the last 20 years has finished in the top 8 during the regular season. 

**Why:** Every year, NBA teams make decisions in free agency on what players to add to their team. While there's many thing to weight when making such a decision, (including whether or not the player wants to join the team), it would be very helpful to understand which players would give the team a higher probability of making it to the top 8. 

Two different players may seem like they could bring the same level of improvement to the team based on certain stats or what salaries they demand, but obviously they likely have some differing skill sets. 

How each players' skill sets fits with the team is not easy to measure, but alas, I've made an attempt to measure it.

**Approach:**

1. Get advanced NBA player stats for the last 12+ years.
    *What are "advanced stats"?* Think aggregated and normalized stats. Many a smart people have tried to add better measurables to the NBA game.
        - One brief example. **VORP** (No, not an alien overlord.) Victories Over Replacement Player. 
        Without getting into the weeds, it basically attempts to measure how many wins a given player contributed to his team over an average player


2. Examine the stats, see which stats seem to correlate best to player performance and team wins. (AKA **EDA**)

3. Make a simple model to predict a teams performance. 
    - First, I tried just using VORP.
    - I got a team's roster for a given year, took those players' VORP scores from the year before, and tried to predict the team's performance (finishing top 8) based on that.

4. Test and learn. Try different combinations of stats, different models, and combining and weighing stats differently (AKA **feature engineering**).
    - Improved the predictions (even if only slightly):
        - Age 
        - Splitting stats by position (and normalizing by position)
        - Separating offensive and defensive stats
        - Taking prior 2 years into account instead of just 1
        - Adding All-NBA Defensive Team votes as a metric

    - **DID NOT** improve the predictions:
        - Creating my own version of Offensive/Defensive Impact scores
        - Trying to add player tracking data such as touches per game (not enough data available)







| Starters | |
| ----------- | ----------- |
| Starters | Title |
| C | *[Fill with free agent or Backup- Dwight Powell]* |
| PF | Kristaps Porzingis |
| SF | Luka Doncic|
| SG | Tim Hardaway Jr. *[or fill with free agent]* |
| PG | *[Fill with free agent or Backup - Jalen Bruson]* |
| Starters | Title |
| C/PF | Dwight Powell |
| C/PF | Maxi Kleber |
| SF/SG | Dorian Finney-Smith|
| SG/PG | JJ Barea |
| PG | Jalen Brunson |



