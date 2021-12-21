#ESCI895 Lab 10 Python Script
#Script By: Tim Hoheneder, University of New Hampshire
#Date Created: 1 December 2021

#%% Description of Dataset and Purpose of Code: 

#Description: 
# Project File Taking Numerous Stream Gauges for a Given Watershed Area and Evaluating How That Pulse of Water
#Moves Through The Given Watershed for a Single Storm Event

#Major Products Inlcude: Time Series Plots, Z-Score Values of Discharge, Basic Stats of Discharge Comparison

#Major Conclusion: The Storm Event Factors More on the Path of the Storm than the Site Hydrologic Properties 

#%% Run "Example Code for My Purposes:

#Install Libraries and !Pip: 
import pandas as pd
import datetime
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt

#%% Importing Raw Datafiles: 

#Importing Data Files: 
albright_file= 'Albright.txt'
blackwater_file= 'Blackwater.txt'
bowden_file= 'Bowden.txt'
hendricks_file = 'Hendricks.txt'
parsons_file= 'Parsons.txt'
rockville_file= 'Rockville.txt'

#%% Importing DataFrames:  

#Albright:
df_albright= pd.read_table(albright_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_albright= df_albright.drop(columns={"5s", "15s", "6s", "10s", "10s.1"})
#Rename Columns: 
df_albright= df_albright.rename(columns={"14n": "Discharge (cfs)"})
df_albright= df_albright.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_albright.interpolate(method = 'linear', inplace = True)

#Blackwater: 
df_blackwater= pd.read_table(blackwater_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_blackwater= df_blackwater.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_blackwater= df_blackwater.rename(columns={"14n": "Discharge (cfs)"})
df_blackwater= df_blackwater.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_blackwater.interpolate(method = 'linear', inplace = True)
    
#Bowden: 
df_bowden= pd.read_table(bowden_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_bowden= df_bowden.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_bowden= df_bowden.rename(columns={"14n": "Discharge (cfs)"})
df_bowden= df_bowden.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_bowden.interpolate(method = 'linear', inplace = True)
    
#Hendricks: 
df_hendricks= pd.read_table(hendricks_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_hendricks= df_hendricks.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_hendricks= df_hendricks.rename(columns={"14n": "Discharge (cfs)"})
df_hendricks= df_hendricks.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_hendricks.interpolate(method = 'linear', inplace = True)  
    
#Parsons: 
df_parsons= pd.read_table(parsons_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_parsons= df_parsons.drop(columns={"5s", "15s", "6s", "10s", "10s.1"})
#Rename Columns: 
df_parsons= df_parsons.rename(columns={"14n": "Discharge (cfs)"})
df_parsons= df_parsons.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_parsons.interpolate(method = 'linear', inplace = True)
    
#Rockville: 
df_rockville= pd.read_table(rockville_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -9999, 8888, -8888])
#Drop Columns: 
df_rockville= df_rockville.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_rockville= df_rockville.rename(columns={"14n": "Discharge (cfs)"})
df_rockville= df_rockville.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_rockville.interpolate(method = 'linear', inplace = True)

#%% Create List of DataFrames and Constants for Functions and Looping: 

#Create List of Gauge Station DataFrames: 
df_list= [df_albright, df_blackwater, df_bowden, df_hendricks, df_parsons, df_rockville]

#Define Watershed Area: 
watershed_area= 1422 #sq-mi

#Define Start and End Dates--Initial Load Includes a Second Storm I Want to Avoid: 
starting_date= pd.to_datetime('2019-06-28 00:00:00')
ending_date= pd.to_datetime('2019-07-05 00:00:00')

#%% Trim DataFrames to Desried Lengths--This Refuses to Work in a Function (See Below): 

#Define Function for Trimming DataFrames: 
def time_trim(df):
    #Trim DataFrames:
    df= df[starting_date:ending_date]
    return df    

#Iterative For Loop to Trim DataFrames: 
for item in df_list: 
    time_trim(item) #Doesn't Trim DataFrames???

#Text Output Statement to Let Me Know For Loop Ran: 
print('')
print('DataFrames Trimmed')
print('')

#Trim DataFrame--For Real This Time: 
df_albright_trim= df_albright[starting_date:ending_date]
df_blackwater_trim= df_blackwater[starting_date:ending_date]
df_bowden_trim= df_bowden[starting_date:ending_date]
df_hendricks_trim= df_hendricks[starting_date:ending_date]
df_parsons_trim= df_parsons[starting_date:ending_date]
df_rockville_trim= df_rockville[starting_date:ending_date]

#Create df_trim List: 
df_trim_list = [df_albright_trim, df_blackwater_trim, df_bowden_trim, df_hendricks_trim, 
                df_parsons_trim, df_rockville_trim]

#%% DataFrame Operations: 

#Define Function for Z-Score Calculations: 
def zscore_Q(df):
    #Calculate Z-Score for DataFrame:
    df['Z-Score Q']= (df['Discharge (cfs)'] - df['Discharge (cfs)'].mean()) / df['Discharge (cfs)'].std()

#Define Function for Discharge Equivalence in cm/hr: 
def discharge_cmhr(df): 
    #Calculate Discharge in cm/hr: 
    df['Discharge (cm/hr)']= (df['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)

#For Loop to Calculate Z-Score for Each DataFrame: 
for item in df_list:
    #Iterate For Loop to Calculate Z-Scores:
    zscore_Q(item)
    #Iterate For Loop to Calculate Discharge Equivalence: 
    discharge_cmhr(item)

for item in df_trim_list:
    #Iterate For Loop to Calculate Z-Scores:
    zscore_Q(item)
    #Iterate For Loop to Calculate Discharge Equivalence: 
    discharge_cmhr(item)

#Output Text Statement: 
print('Calculated Z-Scores for DataFrames')
print('')
print('Discharge Equivalence in cm/hr Calculated for Each DataFrame')
print('')

#%% Plotting Initial Time Series Curves Over Full Duration: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright_trim['Discharge (cm/hr)'], ',', linestyle='-', color='navy', label='Albright')
#Blackwater: 
ax1.plot(df_blackwater_trim['Discharge (cm/hr)'], ',', linestyle='-', color='grey', label='Blackwater')
#Bowden: 
ax1.plot(df_bowden_trim['Discharge (cm/hr)'], ',', linestyle='-', color='dodgerblue', label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks_trim['Discharge (cm/hr)'], ',', linestyle='-', color='maroon', label='Hendricks')
#Parsons: 
ax1.plot(df_parsons_trim['Discharge (cm/hr)'], ',', linestyle='-', color='orange', label='Parsons')
#Rockville: 
ax1.plot(df_rockville_trim['Discharge (cm/hr)'], ',', linestyle='-', color='darkgreen', label='Rockville')

#Axis Formatting: 
ax1.set_ylim(bottom = 0)
ax1.set_xlim(df_albright_trim.index[0], df_albright_trim.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Discharge Curves for Cheat River Watershed', fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75)) 
    
#%% Plotting Initial Time Series Curve Z-Scores Over Full Duration: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright_trim['Z-Score Q'], ',', linestyle='-', color='navy', label='Albright')
#Blackwater: 
ax1.plot(df_blackwater_trim['Z-Score Q'], ',', linestyle='-', color='grey', label='Blackwater')
#Bowden: 
ax1.plot(df_bowden_trim['Z-Score Q'], ',', linestyle='-', color='dodgerblue', label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks_trim['Z-Score Q'], ',', linestyle='-', color='maroon', label='Hendricks')
#Parsons: 
ax1.plot(df_parsons_trim['Z-Score Q'], ',', linestyle='-', color='orange', label='Parsons')
#Rockville: 
ax1.plot(df_rockville_trim['Z-Score Q'], ',', linestyle='-', color='darkgreen', label='Rockville')

#Axis Formatting: 
ax1.set_xlim(df_albright_trim.index[0], df_albright_trim.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Z-Scored Discharge Curves for Cheat River Watershed', 
             fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75)) 

#%% Hydrograph Seperation Function: 

#Define Function:
def hydrograph_sep(totalq, watershed):

    #Find totalq: 
    totalq['Diff'] = totalq['Discharge (cm/hr)'].diff()
    
    #Find Antecedent Discharge and Date using 0.000104 Threshold: 
    global antQ_date
    antQ = (totalq.loc[totalq['Diff'] > 0.000104, 'Discharge (cm/hr)'])
    antQ_date = antQ.index[0]
    antQ_val = round(antQ[0], 3)
    
    #Find Peak Discharge Date: 
    peakQ_date = totalq['Discharge (cm/hr)'].idxmax()
    peakQ = totalq['Discharge (cm/hr)'].max()   
    
    #Calculate Event Duration:
    N = 0.82*(watershed*1e-6)**0.2
    #Calculate End of Event: 
    global end_of_event
    end_of_event = peakQ_date + datetime.timedelta(days = N)
    
    #Calculate Ending Discharge Value: 
    end_Q = totalq.iloc[totalq.index.get_loc(end_of_event,method='nearest'), 
                        totalq.columns.get_loc('Discharge (cm/hr)')]
    
    #Create baseQ Dataframe:
    global baseQ
    baseQ = totalq[['Discharge (cm/hr)']].copy()
    
    #Calculate Base Discharge Curve Before Peak: 
    slope1, intercept1= np.polyfit(totalq.loc[totalq.index < antQ_date].index.astype('int64')
                                /1E9, totalq.loc[totalq.index < antQ_date, 'Discharge (cm/hr)'], 1) 

    #Append Data Before Peak: 
    baseQ.loc[antQ_date:peakQ_date,"Discharge (cm/hr)"] = slope1 * (totalq.loc[antQ_date:peakQ_date].index.view('int64')/1e9) + intercept1
    
    #Calculate Base Discharge Curve After Peak: 
    slope2, intercept2= np.polyfit([peakQ_date.timestamp(), end_of_event.timestamp()], 
                               [baseQ.loc[peakQ_date, 'Discharge (cm/hr)'], end_Q], 1)
    
    #Append Data After Peak: 
    baseQ.loc[peakQ_date:end_of_event,"Discharge (cm/hr)"] = slope2 * (totalq.loc[peakQ_date:end_of_event].index.view('int64')/1e9) + intercept2
    
    #Append BaseQ Values to DataFrame: 
    totalq['BaseQ (cm/hr)'] = baseQ['Discharge (cm/hr)']
    
    #Return Variables: 
    return (baseQ, antQ_date, antQ_val, peakQ_date, peakQ, end_of_event, end_Q)

#%% Modified Time Series Plotting Containing Baseflow: 
    
#Define Function with Keyword Arguement for Baseflow: 
def timeseriesplot(df1, startdate, enddate, baseflow= None):    

    #Create Plot Area: 
    fig, ax1 = plt.subplots()

    #Plot Discharge Data: 
    ax1.plot(df1['Discharge (cm/hr)'], ',', linestyle='-', color='navy', label='Discharge (cm/hr)')

    #Axis Formatting: 
    ax1.set_ylim(bottom = 0)
    ax1.set_xlim(startdate, enddate)
    fig.autofmt_xdate()

    #Axis Labels: 
    ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
    ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
    
    #Optional Arguement Boolean Indicator: 
    if baseflow is not None: 
        ax1.plot(baseflow['Discharge (cm/hr)'], ',', linestyle='-', color='darkseagreen', 
                 label=' Baseflow (cm/hr)')
    
    #Legend: 
    fig.legend(bbox_to_anchor= (0.65, 0.0))   

#%% Running Functions per Watershed: 

#Create Empty Array for Storm Totals: 
storm_totals = []    

#Define Function for Running Functions per Watershed: 
def watershed_function(df):
    #Run Hydrograph Seperation Function: 
        (baseQ, antQ_date, antQ_val, peakQ_date, peakQ, end_of_event, end_Q) = hydrograph_sep(df, watershed_area)
        #Integrating Storm Total: 
        storm_frame= df[antQ_date : end_of_event]
        discharge_total= storm_frame['Discharge (cm/hr)'].sum()
        storm_totals.append(discharge_total)
        #Run Time Series Plotting Function: 
        timeseriesplot(df, df.index[0], df.index[-1], baseQ)

#For Loop to Iterate Through Locations: 
for item in df_list: 
    watershed_function(item)

#Output Text Statement to Let Me Know This Ran: 
print('')
print('Congrats, You Probably Have Some Graphs Now...')
print('')

#%% Determine Effective Flow: 

#Define Function: 
def effect_flow(df): 
    #Calculate Effective Flow: 
    #Ensure All Values of Event Flow are Positive: 
    df['BaseQ (cm/hr)']= np.where(df['BaseQ (cm/hr)'] > 0, df['BaseQ (cm/hr)'], 0)
    #Redefine Values of Event Flow Equal to Discharge as 0: 
    df['Eff Flow (cm/hr)']= np.where(df['Discharge (cm/hr)'] - df['BaseQ (cm/hr)'] > 0,  
                                     df['Discharge (cm/hr)'] - df['BaseQ (cm/hr)'], 0)
        
#Create For Loop to Run Function for Each DataFrame: 
for item in df_list:
    #Run Event Flow-Effect Flow Function: 
    effect_flow(item)

#Output Text Statement to Confirm For Loop:
print('')
print('Event Flow Calculated for Each DataFrame')
print('')

#%% Plotting Effective Flow Curves Over Full Duration: 

#Define Function for Variable Plotting Windows: 
def eventflow_plotting(start_window, end_window):     

    #Create Plotting Area:     
    fig, ax1 = plt.subplots()

    #Plot Discharge Data: 
    #Albright:
    ax1.plot(df_albright['Eff Flow (cm/hr)'], ',', linestyle='-', color='navy', label='Albright')
    #Blackwater: 
    ax1.plot(df_blackwater['Eff Flow (cm/hr)'], ',', linestyle='-', color='grey', label='Blackwater')
    #Bowden: 
    ax1.plot(df_bowden['Eff Flow (cm/hr)'], ',', linestyle='-', color='dodgerblue',  label='Bowden')
    #Hendricks: 
    ax1.plot(df_hendricks['Eff Flow (cm/hr)'], ',', linestyle='-', color='maroon', label='Hendricks')
    #Parsons: 
    ax1.plot(df_parsons['Eff Flow (cm/hr)'], ',', linestyle='-', color='orange', label='Parsons')
    #Rockville: 
    ax1.plot(df_rockville['Eff Flow (cm/hr)'], ',', linestyle='-', color='darkgreen', label='Rockville')

    #Axis Formatting: 
    ax1.set_ylim(bottom = 0)
    ax1.set_xlim(df_albright.index[start_window], df_albright.index[end_window])
    fig.autofmt_xdate()

    #Axis Labels: 
    ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
    ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
    fig.suptitle('Event Flow Discharge Curves for Cheat River Watershed', fontweight= "bold", fontsize=18)

    #Legend: 
    fig.legend(bbox_to_anchor= (1.15, 0.75))  

#Function for Full Duration: 
eventflow_plotting(0, -1)
    
#Function for Zoomed-In Duration: 
eventflow_plotting(275, -675) 

#%% Calculate Z-Score for Effective Flow:  

#Define Fnction: 
def zscore_eventflow(df):
    #Create Z-Score for Event Flow: 
    df['Z-Score EffQ']= (df['Eff Flow (cm/hr)'] - df['Eff Flow (cm/hr)'].mean()) / df['Eff Flow (cm/hr)'].std() 

#For Loop to Iterate Through: 
for item in df_list: 
    #Iterate Thorugh Function: 
    zscore_eventflow(item)
    
#Let Me Know This Ran: 
print('Solved Z-Score for Event Flows')
print('')

#%% Plotting Z-Scored Effective Flow Curves Over Full Duration: 

#Create Function for Z-Score Plotting:     
def zscore_event_plotting(start_window, end_window): 
    
    #Create Plotting Area:     
    fig, ax1 = plt.subplots()

    #Plot Discharge Data: 
    #Albright:
    ax1.plot(df_albright['Z-Score EffQ'], ',', linestyle='-', color='navy', label='Albright')
    #Blackwater: 
    ax1.plot(df_blackwater['Z-Score EffQ'], ',', linestyle='-', color='grey', 
         label='Blackwater')
    #Bowden: 
    ax1.plot(df_bowden['Z-Score EffQ'], ',', linestyle='-', color='dodgerblue', label='Bowden')
    #Hendricks: 
    ax1.plot(df_hendricks['Z-Score EffQ'], ',', linestyle='-', color='maroon', 
         label='Hendricks')
    #Parsons: 
    ax1.plot(df_parsons['Z-Score EffQ'], ',', linestyle='-', color='orange', label='Parsons')
    #Rockville: 
    ax1.plot(df_rockville['Z-Score EffQ'], ',', linestyle='-', color='darkgreen', 
         label='Rockville')

    #Axis Formatting: 
    ax1.set_xlim(df_albright.index[start_window], df_albright.index[end_window])
    fig.autofmt_xdate()

    #Axis Labels: 
    ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
    ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
    fig.suptitle('Z-Scored Event Discharge Curves', 
             fontweight= "bold", fontsize=18)

    #Legend: 
    fig.legend(bbox_to_anchor= (1.15, 0.75))     

#Function for Full Duration: 
zscore_event_plotting(0, -1)
    
#Function for Selected Window: 
zscore_event_plotting(275, -675)

#%% Determine Maximum Values of Discharge: 

#Create Empty Arrays for Quick Reference of Values: 
#Maximum Discharge Values: 
max_q = []
max_q_zscore= []

#Average Discharge Values: 
avg_q= []

#Maximum Event Flow Discharge Values: 
highest_events= []
highest_events_zscore= []

#Minimum Event Flow Discharge Values: 
min_q= [] 
min_event_zscore= [] 

#Function for Max Discharge: 
def max_discharge(df):
    #Calculate Maximum:     
    big_q= df['Discharge (cm/hr)'].max()
    #Append to List: 
    max_q.append(big_q)
    
#Function for Max Discahrge Z-Score: 
def max_discharge_zscore(df):
    big_q_zscore= df['Z-Score Q'].max()
    max_q_zscore.append(big_q_zscore)

#Function for Max Discharge: 
def max_event(df):
    max_event_val= df['Eff Flow (cm/hr)'].max()
    highest_events.append(max_event_val)
    
#Function for Max Discahrge Z-Score: 
def max_event_zscore(df):
    max_event_valz= df['Z-Score EffQ'].max()
    highest_events_zscore.append(max_event_valz)

#Function for Average Discharge: 
def avg_event(df):
    avg_event_val= df['Eff Flow (cm/hr)'].mean()
    avg_q.append(avg_event_val)

#Function for Max Discharge: 
def min_event(df):
    min_event_val= df['Eff Flow (cm/hr)'].min()
    min_q.append(min_event_val)
    
#Function for Max Discahrge Z-Score: 
def min_event_zscore(df):
    min_event_valz= df['Z-Score EffQ'].min()
    min_event_zscore.append(min_event_valz)

#For Loop to Iterate Through Gauges: 
for item in df_list: 
    #Iterate Through Functions: 
    max_discharge(item)
    max_discharge_zscore(item)
    max_event(item)
    max_event_zscore(item)
    avg_event(item)
    min_event(item)
    min_event_zscore(item)
    
#Output Text Statement: 
print('')
print('For Loop Complete: Values Appended to Lists')
print('')

#%% Plotting Total Discharge Storm Event Volumes: 

#Define Bar Graph Plotting Function: 
def bar_plotting(series, barplot_title, barplot_ylab):    

    #Create Plotting Area: 
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    #Add Data Bars: 
    locations = ['Albright', 'Davis', 'Bowden', 'Hendricks', 'Parsons', 'Rockville']
    discharge_series = series
    plotting_series = pd.Series(discharge_series, locations)

    #Colors by Watershed: 
    barplot_colors=  ['dodgerblue', 'darkorange', 'indigo', 'seagreen', 'dodgerblue', 'maroon']

    #Axis Labels: 
    ax.set_ylabel(barplot_ylab, color='k', fontweight="bold", fontsize= 12)
    ax.set_xlabel('Location of Measurement', color='k', fontweight="bold", fontsize= 12)
    ax.set_title(barplot_title, fontweight= "bold", fontsize=18)
    
    #Display Bar Plot: 
    plotting_series.plot(kind='bar', color=barplot_colors)
    plt.show()

#Run Function for Various Series: 
bar_plotting(storm_totals, 'Total Storm Event Discharge', 'Total Discharge (cm)') #Storm Totals
bar_plotting(avg_q, 'Average Storm Event Discharge', 'Average Discharge (cm)') #Average Discharge
bar_plotting(max_q, 'Maximum Storm Event Discharge', 'Maximum Discharge (cm)') #Maximum Discharge
bar_plotting(max_q_zscore, 'Max Storm Event Discharge: Z-Score', 'Max Discharge (cm)') #Max Q as Z-Score

#%% Pearson Coefficient Calculation for Time Series: 

#Create Empty Array to Store PEarson Values for Comparison: 
pearson_array= []    

#Albright-Davis Correlation: 
AlbrightDavisQ=df_albright['Discharge (cm/hr)'].corr(df_blackwater['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(AlbrightDavisQ)

#Albright-Bowden Correlation: 
AlbrightBowdenQ=df_albright['Discharge (cm/hr)'].corr(df_bowden['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(AlbrightBowdenQ)
    
#Albright-Hendricks Correlation: 
AlbrightHendricksQ=df_albright['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(AlbrightHendricksQ)

#Albright-Parsons Correlation:
AlbrightParsonsQ=df_albright['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(AlbrightParsonsQ)    

#Albright-Rockville Correlation: 
AlbrightRockvilleQ=df_albright['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(AlbrightRockvilleQ)

#Davis-Bowden:
DavisBowdenQ=df_blackwater['Discharge (cm/hr)'].corr(df_bowden['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(DavisBowdenQ)    

#Davis-Hendricks:
DavisHendricksQ=df_blackwater['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(DavisHendricksQ)    

#Davis-Parsons:
DavisParsonsQ=df_blackwater['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(DavisParsonsQ)

#Davis-Rockville: 
DavisRockvilleQ=df_blackwater['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(DavisRockvilleQ)

#Bowden-Hendricks: 
BowdenHendricksQ=df_bowden['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(BowdenHendricksQ)    

#Bowden-Parsons: 
BowdenParsonsQ=df_bowden['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(BowdenParsonsQ)    

#Bowden-Rockville: 
BowdenRockvilleQ=df_bowden['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(BowdenRockvilleQ)

#Hendricks-Parsons: 
HendricksParsonsQ=df_hendricks['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(HendricksParsonsQ)    

#Hendricks-Rockville: 
HendricksRockvilleQ=df_hendricks['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(HendricksRockvilleQ)

#Parsons-Rockville: 
ParsonsRockvilleQ=df_parsons['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])
#Append to List: 
pearson_array.append(ParsonsRockvilleQ)

#%% Basic Insight to Pearson Coefficients: 

#Most Similar Gauge Profiles: 
max_pearson = max(pearson_array)
#Output Text Statement: 
print('')
print('The Highest Pearson Coefficient Was %6.4f' %max_pearson)
print('')

#Least Similar Gauge Profiles: 
min_pearson = min(pearson_array)
#Output Text Statement: 
print('')
print('The Lowest Pearson Coefficient Was %6.4f' %min_pearson)
print('')

#Average Across Watershed: 
#Sum of Values: 
sum_pearson= sum(pearson_array)
#LEngth of Array as Proxy of Number of Records: 
len_pearson= len(pearson_array)
#Mean of Array Values: 
avg_pearson = sum_pearson / len_pearson
#Output Text Statement: 
print('')
print('The Average Pearson Coefficient Was %6.4f' %avg_pearson)
print('')

#%% Creating Small DataFrame for Pearson Values: 
      
#List of Gauge Combinations for Reference: 
ref_list = ['Albright-Davis', 'Albrihgt-Bowden', 'Albright-Hendricks', 'Albright-Parsons', 'Albright-Rockville', 
            'Davis-Bowden', 'Davis-Hendricks', 'Davis-Parsons', 'Davis-Rockville', 
            'Bowden-Hendricks', 'Bowden-Parsons', 'Bowden-Rockville', 
            'Hendricks-Parsons', 'Hendricks-Rockville', 
            'Parsons-Rockville']
  
# Calling DataFrame constructor after Zipping:
pearsondf = pd.DataFrame(list(zip(ref_list, pearson_array)),
               columns =['Gauge Combination', 'Pearson Coefficient Value'])

#%% Bar Graph to Display Pearson Values Not in Table: 

#Create Plotting Area: 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

#Add Data Bars: 
locations = ref_list
bars = pearson_array
ax.bar(locations, bars, color = 'navy')

#Axis Labels: 
ax.set_ylabel('Pearson Coefficient Value', color='k', fontweight="bold", fontsize= 12)
ax.set_xlabel('USGS Gauge Location Combination', color='k', fontweight="bold", fontsize= 12)
ax.set_title('Pearson Coefficient Values by Gauge Station Combination', fontweight= "bold", fontsize=18)
    
#Display Bar Plot: 
plt.show() 

#%% End of Code