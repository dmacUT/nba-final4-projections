import pandas as pd
import numpy as np

def get_defvotes():
        adt = pd.read_csv("data/Pstats - adtw19.csv")

        adv18 = {}
        adv17 = {}
        adv16= {}
        adv15= {}
        adv14= {}
        adv13= {}
        adv12= {}
        adv11= {}
        adv10= {}
        adv9= {}
        adv8= {}
        adv7= {}
        adv6= {}
        adv5= {}

        adl = [adv5, adv6, adv7, adv8, adv9, adv10, adv11, adv12, adv13, adv14, adv15, adv16, adv17, adv18]

        #add value and create dict 5-18
        for i in range(0, 14):
                adl[i] = adt[adt.YR == (i+5)].drop('YR', axis = 1)
                adl[i] = {i+5:dict(zip(adl[i].Player, adl[i].Votes))}

        #reverse it
        adlrev = []
        for i in range (13, -1, -1):
                adlrev.append(adl[i])

        #divide latest 5 yrs by 4
        for i in range(0,5):
                for k, v in adlrev[i].items():
                        for key, val in adlrev[i][k].items():
                                adlrev[i][k][key] = adlrev[i][k][key]/4


        ad18 = "Chris Paul, Houston, 74 (20); Paul George, Oklahoma City, 69 (22); Giannis Antetokounmpo, Milwaukee, 43 (15); Kevin Durant, Golden State, 31 (7); Klay Thompson, Golden State, 24 (8); Josh Richardson, Miami, 22 (3); Marcus Smart, Boston, 18 (5); Andre Roberson, Oklahoma City, 17 (3); Jaylen Brown, Boston, 16 (5); Ben Simmons, Philadelphia, 16 (5); P.J. Tucker, Houston, 13 (2); Kyle Lowry, Toronto, 7 (1); Russell Westbrook, Oklahoma City, 7 (1); Danny Green, San Antonio, 6 (2); Luc Mbah a Moute, Houston, 5 (1); Ricky Rubio, Utah, 4 (2); Andre Drummond, Detroit, 3; Gary Harris, Denver, 3; LaMarcus Aldridge, San Antonio, 2; Al-Farouq Aminu, Portland, 2; Avery Bradley, LA Clippers, 2 (1); Steven Adams, Oklahoma City, 1; Will Barton, Denver, 1; Eric Bledsoe, Milwaukee, 1; Ed Davis, Portland, 1; Derrick Favors, Utah, 1; LeBron James, Cleveland, 1; DeAndre Jordan, LA Clippers, 1; Damian Lillard, Portland, 1; Donovan Mitchell, Utah, 1; Fred VanVleet, Toronto, 1"

        ad17 = "Avery Bradley, Boston, 46 (12); Klay Thompson, Golden State, 45 (16); John Wall, Washington, 38 (14); DeAndre Jordan, LA Clippers, 35 (1); Paul Millsap, Atlanta, 35; Hassan Whiteside, Miami, 25 (1); Marcus Smart, Boston, 21 (5); Jimmy Butler, Chicago, 18; LeBron James, Cleveland, 12 (1); Robert Covington, Philadelphia, 11 (2); Russell Westbrook, Oklahoma City, 10 (5); Paul George, Indiana, 7; Kevin Durant, Golden State, 6; Dwight Howard, Atlanta, 6 (1); Mike Conley, Memphis, 5 (1); Jae Crowder, Boston, 5; Jrue Holiday, New Orleans, 5; Wesley Matthews, Dallas, 4 (2); Stephen Curry, Golden State, 3; Andre Iguodala, Golden State, 3 (1); Michael Kidd-Gilchrist, Charlotte, 3; Ricky Rubio, Minnesota, 3; P.J. Tucker, Toronto, 3; Trevor Ariza, Houston, 2; Nicolas Batum, Charlotte, 2; Marc Gasol, Memphis, 2; Eric Gordon, Houston, 2 (1); Karl-Anthony Towns, Minnesota, 2 (1); Steven Adams, Oklahoma City, 1; LaMarcus Aldridge, San Antonio, 1; Al-Farouq Aminu, Portland, 1; Kentavious Caldwell-Pope, Detroit, 1; George Hill, Utah, 1; Serge Ibaka, Toronto, 1; Damian Lillard, Portland, 1; Luc Mbah a Moute, LA Clippers, 1; Austin Rivers, LA Clippers, 1; Isaiah Thomas, Boston, 1; Cody Zeller, Charlotte, 1"

        ad16 = "Rudy Gobert, Utah, 64 (17); Klay Thompson, Golden State, 49 (16); Jae Crowder, Boston, 47 (3); LeBron James, Cleveland, 43 (5); Kyle Lowry, Toronto, 43 (9); Danny Green, San Antonio, 39 (9); Russell Westbrook, Oklahoma City, 35 (12); Tim Duncan, San Antonio, 33 (5); Ricky Rubio, Minnesota, 30 (6); Kentavious Caldwell-Pope, Detroit, 27 (3); Anthony Davis, New Orleans, 24 (3); Andre Drummond, Detroit, 14 (5); Serge Ibaka, Oklahoma City, 14 (1); Stephen Curry, Golden State, 13 (3); Andre Iguodala, Golden State, 13 (3); Patrick Beverley, Houston, 11 (1); Al Horford, Atlanta, 7 (1); Marcus Smart, Boston, 7 (2); John Wall, Washington, 6; Giannis Antetokounmpo, Milwaukee, 3; Trevor Ariza, Houston, 3; Kent Bazemore, Atlanta, 3; Andrew Bogut, Golden State, 3 (1); DeMarcus Cousins, Sacramento, 3 (1); Nicolas Batum, Charlotte, 2; Victor Oladipo, Orlando, 2 (1); LaMarcus Aldridge, San Antonio, 1; Harrison Barnes, Golden State, 1; Bismack Biyombo, Toronto, 1; Mike Conley, Memphis, 1; Kevin Durant, Oklahoma City, 1; Derrick Favors, Utah, 1; George Hill, Indiana, 1; Wesley Matthews, Dallas, 1; Luc Mbah a Moute, Los Angeles Clippers, 1; Kristaps Porzingis, New York, 1; Andre Roberson, Oklahoma City, 1; Mike Scott, Atlanta, 1; Dwyane Wade, Miami, 1"

        ad15 = "Rudy Gobert, Utah, 54 (5); LeBron James, Cleveland, 47 (6); Russell Westbrook, Oklahoma City, 35 (13); Avery Bradley, Boston, 26 (5); Michael Kidd-Gilchrist, Charlotte, 21 (2), Klay Thompson, Golden State, 19 (3); Marc Gasol, Memphis, 18 (2); Danny Green, San Antonio, 18; Trevor Ariza, Houston, 17 (1); Stephen Curry, Golden State, 14 (2); DeMarre Carroll, Atlanta, 11 (1); Patrick Beverley, Houston, 10 (1); Khris Middleton, Milwaukee, 9 (1); Serge Ibaka, Oklahoma City, 8; Andre Iguodala, Golden State, 8; Paul Millsap, Atlanta, 8; Jeff Teague, Atlanta, 7 (1); Mike Conley, Memphis, 6 (2); Joakim Noah, Chicago, 6; Nerlens Noel, Philadelphia, 5 (1); P.J. Tucker, Phoenix, 4; Giannis Antetokounmpo, Milwaukee, 3; Pau Gasol, Chicago, 3; Wesley Matthews, Portland, 3; James Harden, Houston, 2; Marcus Smart, Boston, 2; Hassan Whiteside, Miami, 2 (1); Eric Bledsoe, Phoenix, 1; DeMarcus Cousins, Sacramento, 1; Andre Drummond, Detroit, 1; Manu Ginobili, San Antonio, 1; Blake Griffin, L.A. Clippers, 1; George Hill, Indiana, 1; Al Horford, Atlanta, 1; Victor Oladipo, Orlando, 1; Zaza Pachulia, Milwaukee, 1; Elfrid Payton, Orlando, 1; Zach Randolph, Memphis, 1; Rajon Rondo, Dallas, 1; Iman Shumpert, Cleveland, 1; Dwyane Wade, Miami, 1"

        ad14 = "DeAndre Jordan, L.A. Clippers, 63 (14); Anthony Davis, New Orleans, 62 (18); Tony Allen, Memphis, 60 (17); Tim Duncan, San Antonio, 45 (12); Dwight Howard, Houston, 26 (6); Taj Gibson, Chicago, 21 (2); Mike Conley, Memphis, 21 (5); Ricky Rubio, Minnesota, 19 (5); Lance Stephenson, Indiana, 14 (3); P.J. Tucker, Phoenix, 13 (2); Kevin Durant, Oklahoma City, 10 (2); Kyle Lowry, Toronto, 10 (3); Eric Bledsoe, Phoenix, 9 (1); Marc Gasol, Memphis, 8; John Wall, Washington, 8 (1); Thabo Sefolosha, Oklahoma City, 8 (1); Kirk Hinrich, Chicago, 7 (2); Trevor Ariza, Washington, 5 (2); Avery Bradley, Boston, 5 (1); Russell Westbrook, Oklahoma City, 5 (1); Klay Thompson, Golden State, 5; Andrew Bogut, Golden State, 4; Chris Bosh, Miami, 4 (1); Luol Deng, Cleveland, 4 (1); Wesley Matthews, Portland, 4 (1); Tony Parker, San Antonio, 4 (1); Nicolas Batum, Portland, 3 (1); Stephen Curry, Golden State, 3 (1); Danny Green, San Antonio, 3 (1); Michael Kidd-Gilchrist, Charlotte, 3; Shaun Livingston, Brooklyn, 3 (1); Victor Oladipo, Orlando, 3 (1); DeMarre Carroll, Atlanta, 2; Matt Barnes, L.A. Clippers, 2 (1); James Harden, Houston, 2; George Hill, Indiana, 2; Jeff Teague, Atlanta, 2; Dwyane Wade, Miami, 2 (1); Kemba Walker, Charlotte, 2; David West, Indiana, 2; Arron Afflalo, Orlando, 1; Corey Brewer, Minnesota, 1; Michael Carter-Williams, Philadelphia, 1; Darren Collison, L.A. Clippers, 1; DeMar DeRozan, Toronto, 1; Andre Drummond, Detroit, 1; Monta Ellis, Dallas, 1; Danny Granger, L.A. Clippers, 1; Draymond Green, Golden State, 1; Reggie Jackson, Oklahoma City, 1; David Lee, Golden State, 1; Paul Millsap, Atlanta, 1; Rajon Rondo, Boston, 1"

        ad13 = "Andre Iguodala, Denver, 16 (2); Larry Sanders, Milwaukee, 16 (4); Thabo Sefolosha, Oklahoma City, 15 (2); Luol Deng, Chicago, 11 (1); Dwight Howard, L.A. Lakers, 9 (3); Kobe Bryant, L.A. Lakers, 6 (1); Roy Hibbert, Indiana, 6 (2); Kenneth Faried, Denver, 4 (1); Russell Westbrook, Oklahoma City, 4 (1); Shane Battier, Miami, 2; Nicolas Batum, Portland, 2 (1); Corey Brewer, Denver, 2; George Hill, Indiana, 2; Mike James, Dallas, 2 (1); Kawhi Leonard, San Antonio, 2, (1); Tony Parker, San Antonio, 2 (1); Dwyane Wade, Miami, 2; Metta World Peace, L.A. Lakers, 2 (1); Eric Bledsoe, L.A. Clippers, 1; Kevin Durant, Oklahoma City, 1; Jrue Holiday, Philadelphia, 1; Andrei Kirilenko, Minnesota, 1; Iman Shumpert, New York, 1; David West, Indiana, 1"

        ad12 = "Andre Iguodala, Philadelphia, 19 (4); Joakim Noah, Chicago, 14; Iman Shumpert, New York, 13 (4); Paul George, Indiana, 10 (2); Russell Westbrook, Oklahoma City, 9 (2); Josh Smith, Atlanta, 8 (2); Dwyane Wade, Miami, 5 (1); Thabo Sefolosha, Oklahoma City, 5 (1); Grant Hill, Phoenix, 5 (1); Tim Duncan, San Antonio, 5 (1); Avery Bradley, Boston, 3 (1); Marc Gasol, Memphis, 3 (1); Metta World Peace, L.A. Lakers, 3; Shawn Marion, Dallas, 3; Joe Johnson, Atlanta, 2, (1); Mike Conley, Memphis, 2; Derrick Rose, Chicago, 1; Jrue Holiday, Philadelphia, 1; Carlos Boozer, Chicago, 1; Luc Mbah a Moute, Milwaukee, 1"

        ad11 = "Derrick Rose, Chicago, 14 (4); Dwyane Wade, Miami, 13 (3); Russell Westbrook, Oklahoma City, 13 (4); Gerald Wallace, Portland, 11 (1); Grant Hill, Phoenix, 11 (4); Luol Deng, Chicago, 11 (4); Tim Duncan, San Antonio, 11 (3); Chuck Hayes, Houston, 10 (2); Al Horford, Atlanta, 7 (3); Josh Smith, Atlanta, 7 (1); Ron Artest, Los Angeles Lakers, 7 (1); Serge Ibaka, Oklahoma City, 6 (1); Shane Battier, Memphis, 5 (2); Thabo Sefolosha, Oklahoma City, 5; Wesley Matthews, Portland, 4 (2); Kendrick Perkins, Oklahoma City, 3; Nicolas Batum, Portland, 3, (1); Joe Johnson, Atlanta, 2 (1); Keith Bogans, Chicago, 2 (1); Kyle Lowry, Houston, 2; Lamar Odom, Los Angeles Lakers, 2; Luc Mbah a Moute, Milwaukee, 2; Manu Ginobili, San Antonio, 2 (1); Andrew Bogut, Milwaukee, 1; Andrew Bynum, Los Angeles Lakers, 1; Arron Afflalo, Denver, 1; Jrue Holiday, Philadelphia, 1; Kirk Hinrich, Atlanta, 1; Nene Hilario, Denver, 1; Ronnie Brewer, Chicago, 1; Shawn Marion, Dallas, 1; Tayshaun Prince, Detroit, 1"

        ad10 = "Jason Kidd, Dallas, 12 (4); Marcus Camby, Portland, 12 (1); Ron Artest, Los Angeles, 11, (3); Deron Williams, Utah, 10, (2); Kirk Hinrich, Chicago, 9 (1); Andrew Bogut, Milwaukee, 8; Luc Mbah a Moute, Milwaukee, 8 (1); Arron Afflalo, Denver, 6 (1); Kenyon Martin, Denver, 5 (1); Kevin Garnett, Boston, 5 (1); Grant Hill, Phoenix, 4 (2); Joakim Noah, Chicago, 4; Kendrick Perkins, Boston, 4 (1); Shane Battier, Houston, 4 (1); Andrei Kirilenko, Utah, 3 (1); Russell Westbrook, Oklahoma City, 3; Trevor Ariza, Houston, 3 (1); Andre Iguodala, Philadelphia, 2; George Hill, San Antonio, 2 (1); Jermaine O’Neal, Miami, 2 (1); Joe Johnson, Atlanta, 2 (1); Lamar Odom, L.A. Lakers, 2 (1); Luis Scola, Houston, 2; Manu Ginobili, San Antonio, 2 (1); Nicolas Batum, Portland, 2; Caron Butler, Dallas, 1; Chauncey Billups, Denver, 1; Jared Dudley, Phoenix, 1; Kevin Durant, Oklahoma City, 1; Raymond Felton, Charlotte, 1; Marc Gasol, Memphis, 1; Pau Gasol, L.A. Lakers, 1; Chuck Hayes, Houston, 1; Brendan Haywood, Dallas, 1; Al Horford, Atlanta, 1; Serge Ibaka, Oklahoma City, 1; Ersan Ilyasova, Milwaukee, 1; Stephen Jackson, Charlotte, 1; Nene Hilario, Denver, 1; Chris Paul, New Orleans, 1; Tayshaun Prince, Detroit, 1; Earl Watson, Indiana, 1"

        ad09 = "Tayshaun Prince, Detroit, 15 (3); Raja Bell, Charlotte, 8 (2); Joel Przybilla, Portland, 7 (1); Chauncey Billups, Denver, 5; Ronnie Brewer, Utah, 5 (1); Andre Iguodala, Philadelphia, 5; Yao Ming, Houston, 5; Emeka Okafor, Charlotte, 5 (1); Kendrick Perkins, Boston, 4 (1); Samuel Dalembert, Philadelphia, 3; Derek Fisher, L.A. Lakers, 3 (1); Udonis Haslem, Miami, 3; Jason Kidd, Dallas, 3 (1); Anderson Varejao, Cleveland, 3; Deron Williams, Utah, 3; Trevor Ariza, L.A. Lakers, 2; Kirk Hinrich, Chicago, 2; Joe Johnson, Atlanta, 2 (1); Andrei Kirilenko, Utah, 2 (1); David Lee, New York, 2 (1); James Posey, New Orleans, 2; J.R. Smith, Denver, 2 (1); Gerald Wallace, Charlotte, 2; Nene Hilario, Denver, 1; Chris Andersen, Denver, 1; Pau Gasol, L.A. Lakers, 1; Antonio McDyess, Detroit, 1; Andre Miller, Philadelphia, 1; Travis Outlaw, Portland, 1; Brandon Roy, Portland, 1; Rasheed Wallace, Detroit, 1"

        ad08 = "Chauncey Billups, Detroit, 14 (5); Jason Kidd, Dallas, 13 (4); Rasheed Wallace, Detroit, 13 (3); Rajon Rondo, Boston, 11 (3); Deron Williams, Utah, 8, (3); Josh Smith, Atlanta, 8, (3); Ron Artest, Sacramento, 8 (2); Tyson Chandler, New Orleans, 8 (1); Andrei Kirilenko, Utah, 6; Derek Fisher, Los Angeles Lakers, 4 (1); LeBron James, Cleveland, 4 (1); Manu Ginobili, San Antonio, 4 (1); Kirk Hinrich, Chicago, 3 (1); Samuel Dalembert, Philadelphia, 3 (1); Andre Iguodala, Philadelphia, 2 (1); Brandon Roy, Portland, 2; Paul Pierce, Boston, 2; Andre Miller, Philadelphia, 1; Andres Nocioni, Chicago, 1; Baron Davis, Golden State, 1; Caron Butler, Washington, 1; Chris Bosh, Toronto, 1; Dikembe Mutombo, Houston, 1; Josh Howard, Dallas, 1; Richard Hamilton, Detroit, 1; Ronnie Brewer, Utah, 1"

        ad07 = "Shane Battier, Houston, 17 (6); Shawn Marion, Phoenix, 15 (5); Chauncey Billups, Detroit, 13 (5); Ron Artest, Sacramento, 12 (4); Gerald Wallace, Charlotte, 11 (2); Alonzo Mourning, Miami, 9 (4); Devin Harris, Dallas, 8 (2); Tyson Chandler, New Orleans/Oklahoma City, 8 (2); Josh Howard, Dallas, 7; Emeka Okafor, Charlotte, 6 (1); Luol Deng, Chicago, 5 (2); Rasheed Wallace, Detroit, 5 (1); Dwight Howard, Orlando, 5 (1); Andrei Kirilenko, Utah, 5; Josh Smith, Atlanta, 4 (1); Manu Ginobili, San Antonio, 4 (1); Richard Hamilton, Detroit, 3 (1); Jermaine O’Neal, Indiana, 3 (1); Andre Iguodala, Philadelphia, 3; Deron Williams, Utah, 3; Elton Brand, L.A. Clippers, 2 (1); Dwyane Wade, Miami, 2; Trenton Hassell, Minnesota, 2 (1); Tony Parker, San Antonio, 2; Caron Butler, Washington, 2; Chris Duhon, Chicago, 1; Allen Iverson, Denver, 1; Udonis Haslem, Miami, 1; Richard Jefferson, New Jersey, 1; Chris Paul, New Orleans/Oklahoma City, 1; Andre Miller, Philadelphia, 1; Joel Przybilla, Portland, 1; Francisco Elson, San Antonio, 1; Chris Bosh, Toronto, 1; Gilbert Arenas, Washington, 1"

        ad06 = "Rasheed Wallace, Detroit, 12 (3); Gerald Wallace, Charlotte, 11 (3); Shawn Marion, Phoenix, 11 (1); Raja Bell, Phoenix, 9 (2); Dwyane Wade, Miami, 8 (3); Kirk Hinrich, Chicago, 7 (3); Alonzo Mourning, Miami, 7 (1); Shane Battier, Memphis, 6 (1); Gilbert Arenas, Washington, 4 (1); Andre Iguodala, Philadelphia, 4; P.J. Brown, New Orleans/Okla. City, 3; Manu Ginobili, San Antonio, 3; Dwight Howard, Orlando, 3; Josh Howard, Dallas, 3 (1); Brevin Knight, Charlotte, 3; Shaquille O’Neal, Miami, 3 (1); Elton Brand, L.A. Clippers, 2 (1); LeBron James, Cleveland, 2 (1); Tony Parker, San Antonio, 2 (1); Quinton Ross, L.A. Clippers, 2 (1); Allen Iverson, Philadelphia, 2; Eddie Jones, Memphis, 2; Jason Collins, New Jersey, 1; Boris Diaw, Phoenix, 1; Pau Gasol, Memphis, 1; Trenton Hassel, Minnesota, 1; Mike James, Toronto, 1; Richard Jefferson, New Jersey, 1; Ruben Patterson, Denver, 1; Chris Paul, New Orleans/Okla. City, 1; Jason Terry, Dallas, 1; Yao Ming, Houston, 1"

        ad05 = "Shaquille O’Neal, Miami, 16 (3); Allen Iverson, Philadelphia, 14 (6); Kobe Bryant, Los Angeles Lakers, 12 (3); Shawn Marion, Phoenix, 9 (4); Manu Ginobili, San Antonio, 9; Rasheed Wallace, Detroit, 8 (1); Kenyon Martin, Denver, 6; Tony Parker, San Antonio, 5 (2); Andre Iguodala, Philadelphia, 5 (2); Shane Battier, Memphis, 4; Andre Miller, Denver, 4 (1); Tyson Chandler, Chicago, 3; Trenton Hassell, Minnesota, 3 (1); Kirk Hinrich, Chicago, 3; Joe Johnson, Phoenix, 3; Earl Watson, Memphis, 3 (1); Udonis Haslem, Miami, 2; Tracy McGrady, Houston, 2; Josh Howard, Dallas, 2 (1); Joel Przybilla, Portland, 2; Desmond Mason, Milwaukee, 2; Brevin Knight, Charlotte, 2; Vince Carter, New Jersey, 1; P.J. Brown, New Orleans, 1; Mickael Pietrus, Golden State, 1; Kurt Thomas, New York, 1; Jeff Foster, Indiana, 1; Greg Buckner, Denver, 1; Gilbert Arenas, Washington, 1; Gary Payton, Boston, 1; Emeka Okafor, Charlotte, 1; Eddie Jones, Miami, 1; Clifford Robinson, New Jersey, 1; Brendan Haywood, Washington, 1; Anthony Johnson, Indiana, 1; Andres Nocioni, Chicago, 1; Amaré Stoudemire, Phoenix, 1"

        advotes = [ad18, ad17, ad16, ad15, ad14, ad13, ad12, ad11, ad10, ad09, ad08, ad07, ad06, ad05]

        def text_to_votes(test):
                tl = test.split(";")
                tl = [i.split(",") for i in tl]
                tldict = {i[0].lstrip(): int(i[2].split()[0]) for i in tl}
                return tldict

        adv_dict = []
        for i in range(len(advotes)):
                adv_dict.append(text_to_votes(advotes[i]))

        for i in adv_dict:
                for k, v in i.items():
                        k = k.lstrip()

        #adding year
        yr = 18
        for i in range(0, 14):
                adv_dict[i] = {yr:adv_dict[i]}
                yr -= 1

        for i in range(0,5):
                for k, v in adv_dict[i].items():
                        for key, val in adv_dict[i][k].items():
                                adv_dict[i][k][key] = adv_dict[i][k][key]/4


        #combining all-team defensive votes with players who got votes but didn't make all-def team
        fv_dict2 = {}
        year = 18
        for i in range(0,14):
                adv_dict[i][year].update(adlrev[i][year])
                fv_dict2[year] = adv_dict[i][year]
                year -= 1

        def sum_past_yrs(alladv):
                new = alladv
                for k, v in new.items():
                        count = 1
                        prior = k-count
                        # 15 - prior
                        while prior > 4:
                                # p, v in 15
                                for player, votes in new[k].items():
                                        # p, in 14
                                        if player in new[prior].keys():
                                                # p15 + p14
                                                new[k][player] += new[prior][player]
                                prior -= 1
                return new

        def replace_nans(sum_dict):
                sd = pd.DataFrame.from_dict(sum_dict)
                for i in range(6,19):
                        sd[i] = sd[i].fillna(value=(.75 * sd[i-1]))
                sd_dict = sd.to_dict()
                return sd_dict


        def make_df(fv_dict2):
                for k, v in fv_dict2.items():
                        for key, value in fv_dict2[k].items():
                                fv_dict2[k][key] = k, value
                #d_votes = pd.DataFrame.from_dict(fv_dict2[17], orient="index", columns=["YR","advotes"])
                d_votes = pd.DataFrame(columns=['Player','YR','advotes'])
                for i in range(18, 4, -1):
                        a = pd.DataFrame.from_dict(fv_dict2[i], orient="index", columns=["YR","advotes"])
                        a = a.reset_index()
                        a = a.rename(columns={"index":"Player"})
                        d_votes = d_votes.append(a, ignore_index=True)

                d_votes['YR'] = d_votes['YR'].astype("int32")
                return d_votes

        sdd = replace_nans(fv_dict2)
        d_votes = make_df(sdd)

        return d_votes


