# Predicting NBA team standings from player stats

**Goal:** Predict the probability of an NBA team finishing the regular season in the top 8 (of 30) teams based on their players. **Why top 8?** Every team that has won a championship in the last 20 years has finished in the top 8 during the regular season.


<img align="right" src="https://github.com/dmacUT/nba-final4-projections/blob/master/images/top8v2.png">  

**Application:** Every year, NBA teams make decisions in free agency on what players to add to their team. While there's many thing to weight when making such a decision, (including whether or not the player wants to join the team), it would be very helpful to understand which players would give the team a higher probability of making it to the top 8. 

**Thinking beyond basketball:** The data analysis and aggregation, machine learning, and findings are relatable to several questions outside of sport and basketball in particular. A few high ways are resource allocation (marketing mix), feature prioritization (product development), and customer segmentation (Lifetime Customer Value or bulding personas). 

**Approach:**

1. Get advanced NBA player stats for the last 12+ years.
    *What are "advanced stats"?* Think aggregated and normalized stats. Many a smart people have tried to add better measurables to the NBA game.
        - One brief example. **VORP** (No, not an alien overlord.) Victories Over Replacement Player. 
        Without getting into the weeds, it basically attempts to measure how many wins a given player contributed to his team over an average player

2. Examine the stats, see which stats seem to correlate best to player performance and team wins. (AKA **EDA**)

3. Make a simple model to predict a teams performance. 
    - First, I tried using just points per game (PPG) from the prior season.
    - I got a team's roster for a given year, took those players' PPG from the year before, and tried to predict the team's performance (finishing top 8) based on that.

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

**The Final Model:**
    - My final model ended up being a random forest model, which just beat out a gradient boosting model. 
    - It took 30 total features, or 6 stats per the traditional 5 NBA positions (PG, SG, SF, PF, C) with a weighted average from the prior two seasons.

**Results:**
Initially, I trained the model on '05-'16 seasons, and tested in on '17-'19 seasons. This resulted in a .677 F1 score. Essentially, if the model predicted the 2/3 of top 8 teams over 3 seasons.

I retrained the model on '05-'18 data to only predict the '19 season, and the results are below in comparison with Nate Silver's FiveThirtyEight's pre-season predicitons.

<img align="center" src="https://github.com/dmacUT/nba-final4-projections/blob/master/images/Results19.png">  

While the results are exciting, they come with caveats. When the player's on a team don't change from year to year, the team tends to perform to the same level as the did th prior year. 

**Takeaways:**
**My 3 main basketball takeaways are:**
1. "Defense wins championships!" as the old adage goes
    - (See graphic below)
2. The top and the bottom are more predictable than the middle
    - While the model predicted the top 8, and even the top 16, decently, it struggled the most between places 13 and 25. 
3. It's hard to predict a big leap in player improvement or worsening from year to year
    - Some players can play at a consistent level into their mid-thirties while other's really start falling off in their early thirties. Some players go from "developing" to "arrived" in year 2, year 4, or (most likely) never.

    

**Next steps:**
- Diving deeper into predicing when players will make a big leap or big drop off, and taking into account pre-NBA data will be in the v2 to better account for takeaways 2 and especially 3.

**Thinking beyond basketball:**




