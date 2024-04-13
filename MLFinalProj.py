import pandas as pd 
import numpy as np
import random

from sklearn.model_selection import train_test_split

data = pd.read_csv('Mental_Health_Survey_Feb_20_22.csv')
data = data.drop(['StartDate', 'EndDate', 'Status', 'Progress', 'Duration (in seconds)', 'Finished', 
                  'RecordedDate', 'ResponseId','DistributionChannel', 'UserLanguage', 'ProlificID', 
                  'BrowserInfo_Browser', 'BrowserInfo_Version', 'BrowserInfo_Operating System', 
                  'BrowserInfo_Resolution', 'timing_intro_First Click', 'timing_intro_Last Click', 
                  'timing_intro_Page Submit', 'timing_intro_Click Count', 'timing_consent_First Click', 
                  'timing_consent_Last Click', 'timing_consent_Page Submit','timing_consent_Click Count', 
                  'timing_mh_intro_First Click','timing_mh_intro_Page Submit','timing_mh_intro_Click Count', 
                  'timing_mh_intro_Last Click','timing_phq9_Last Click','timing_phq9_First Click',
                  'timing_phq9_Last Click','timing_phq9_Page Submit','timing_phq9_Click Count',
                  'timing_gad7_First Click','timing_gad7_Last Click','timing_gad7_Page Submit', 
                  'timing_gad7_Click Count','acha_services_1','acha_services_2','acha_services_3',
                  'attention','acha_timing_First Click','acha_timing_Last Click','acha_timing_Page Submit',
                  'acha_timing_Click Count','race_1','race_2','race_3','race_4','race_5','race_6',
                  'timing_controls_First Click','timing_controls_Last Click','timing_controls_Page Submit',
                  'timing_controls_Page Submit','timing_controls_Click Count','purpose','feedback',
                  'acha_part2_timing_First Click','acha_part2_timing_Last Click','acha_part2_timing_Page Submit',
                  'acha_part2_timing_Click Count','timing_feedback_First Click','timing_feedback_Last Click',
                  'timing_feedback_Click Count','timing_feedback_Page Submit','PROLIFIC_PID','FL_11_DO_PHQ-9',
                  'FL_11_DO_GAD-7','FL_11_DO_ACHA', 'Create New Field or Choose From Dropdown...', 'surveys', 'survey_intro'], axis='columns')
data = data.drop(1,axis=0)
yes =  data[data['acha_depression'] == 'Yes']
no =  data[data['acha_depression'] == 'No']

#CONVERT NUMERICAL DATA TO CATEGORICAL DATA FOR YEAR COLUMN
columns_to_replace = ['year_1']

def replace_years(value):
    if pd.notna(value):
        try:
            num_value = int(value)
            if num_value < 1965:
                return 'Boomer'
            elif num_value < 1981:
                return 'Gen X'
            elif num_value < 1997:
                return 'Millenial'
            else:
                return 'Gen Z'
        except ValueError:
            return value
    return value

data[columns_to_replace] = data[columns_to_replace].applymap(replace_years)

#CONVERT NUMERICAL DATA TO CATEGORICAL DATA FOR THE BELOW COLUMNS
columns_to_replace = ['acha_12months_times_1', 'acha_12months_times_2', 'acha_12months_times_3',
                      'acha_12months_times_4', 'acha_12months_times_5', 'acha_12months_times_6',
                      'acha_12months_times_7']

def replace_values(value):
    if pd.notna(value):
        try:
            if '11 or more times' in value:
                return 'Almost all of the time'
            num_value = int(value.split('-')[0])
            if num_value > 5:
                return 'Many times'
            elif num_value <= 5:
                return 'A couple times'
        except ValueError:
            return value
    return value

data[columns_to_replace] = data[columns_to_replace].applymap(replace_values)

#MOVE 'DEPRESSION' TO THE LAST COLUMN SINCE IT IS THE OUTPUT
cols = list(data.columns)
cols.remove('acha_depression')
data = data[cols + ['acha_depression']]

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) 
x_train['acha_depression'] = y_train

depressed_df = x_train[x_train['acha_depression'] == 'Yes']
not_depressed_df = x_train[x_train['acha_depression'] == 'No']

Apriori = x_train['acha_depression'].value_counts() / len(x_train)

# CALCULATE PROBABILTIIES 

all_probabilities_depressed = []
all_probabilities_not_depressed = []

order = ["year_1", "state_1", "general_health", "phq9_1", "phq9_2", "phq9_3","phq9_4","phq9_5","phq9_6","phq9_7","phq9_8","phq9_9", "gad7_1", "gad7_2","gad7_3","gad7_4","gad7_5","gad7_6","gad7_7","acha_12months_times_1", "acha_12months_times_2", "acha_12months_times_3", "acha_12months_times_4", "acha_12months_times_5", "acha_12months_times_6", "acha_12months_times_7", "acha_12months_any_1", "acha_12months_any_2","acha_12months_any_3", "acha_12months_any_4", "acha_12months_any_5", "acha_12months_any_6", "acha_12months_any_7", "acha_12months_any_8", "acha_12months_any_9","acha_12months_any_10", "acha_12months_any_11", "acha_12months_any_12",	"acha_12months_any_13", "acha_12months_any_14", "acha_12months_any_15", "acha_12months_any_16", "acha_12months_any_17", "acha_12months_any_18", "acha_12months_any_19", "acha_12months_any_20", "acha_12months_any_21", "acha_12months_any_22", "acha_12months_any_23", "acha_12months_any_24", "acha_12months_any_25", "acha_12months_any_26" , "acha_12months_any_27", "acha_12months_any_28", "acha_12months_any_29", "sex" , "fulltime", "international"]


for i in range(len(order)):
    series_depressed = depressed_df[order[i]].value_counts() / len(depressed_df[order[i]])
    series_not_depressed = not_depressed_df[order[i]].value_counts() / len(not_depressed_df[order[i]])
    all_probabilities_depressed.append(series_depressed)
    all_probabilities_not_depressed.append(series_not_depressed)




#PRINT DATAFRAME AND CREATE UPDATED CSV TO BETTER VISUALIZE IT
#data.to_csv('Updated_Mental_Health_Survey.csv', index=False)
print(data)

#USE VALUE COUNTS TO GET APRIORI FOR EACH OUTCOME
Fdata = data[data['acha_depression'] != 'Have you ever been diagnosed with depression?']
outcome_counts1 = Fdata['acha_depression'].value_counts()
apriori_prob1 = outcome_counts1 / len(Fdata)
print('\n\nApriori Probability:')
print(Apriori)

#CREATE A PROBABILITY TABLE
data1 = pd.read_csv('Updated_Mental_Health_Survey.csv')
outcome_column = data1.columns[-1] 
probabilistic_table = pd.crosstab([data1[data1.columns[i]] for i in range(len(data1.columns)-1)], data1[outcome_column], normalize='index')
probabilistic_table = probabilistic_table.applymap(lambda x: 'Yes' if x > 0 else 'No')
print('\n\nProbability Table:')
print(probabilistic_table)

#PRINT DATAFRAME AND CREATE UPDATED CSV TO BETTER VISUALIZE IT
probabilistic_df = pd.DataFrame(probabilistic_table)
probabilistic_table.to_csv('PROBABILISTIC_TABLE.csv')

#USE THE PROBABILISTIC DATAFRAME TO PREDICT FUTURE SCENARIOS
scenario_1 = ('Gen X','Alaska','Fair','Not at all','Not at all',
              'Several days','Several days','Not at all',
              'Not at all','Several days','Not at all',
              'Not at all','Several days','Not at all',
              'Not at all','Not at all','Not at all','Not at all',
              'Not at all','Never','A couple times','A couple times',
              'A couple times','Never','Never','Never','No','No','No','Yes',
              'No','No','No','No','No','No','No','No','No','No','No','Yes','No','No','Yes',
              'Yes','No','No','No','No','No','No','Yes','No','No','Male','No','No')



scenario_2 = ('Gen Z','Connecticut','Fair','More than half of the days',
              'More than half of the days','Several days','More than half of the days',
              'Nearly every day','More than half of the days','Several days',
              'Several days','Several days','More than half of the days',
              'More than half of the days','More than half of the days',
              'Several days','Several days','Nearly every day','Several days',
              'Almost all of the time','Almost all of the time','Almost all of the time',
              'Almost all of the time','Almost all of the time','A couple times',
              'Never','No','Yes','Yes','Yes','Yes','No','Yes','No','No','No','No',
              'No','Yes','Yes','No','No','Yes','No','Yes','No','No','No','No','No',
              'No','No','No','No','No','Female','Yes','No')

def calc_probability_depressed(answers):
    prob_depressed = Apriori[1]
    prob_not_depressed = Apriori[0]
    for i in range(len(order)):
        try:
            prob_depressed = prob_depressed * all_probabilities_depressed[i][answers[i]]
            prob_not_depressed = prob_not_depressed * all_probabilities_not_depressed[i][answers[i]]
        except:
            prob_depressed = prob_depressed
            prob_not_depressed = prob_not_depressed
    if(prob_depressed > prob_not_depressed):
        return True
    else:
        return False

depressed_scenario1 = calc_probability_depressed(scenario_1)
depressed_scenario2 = calc_probability_depressed(scenario_2)

# PRINT THE PREDICTIONS FOR THE FUTURE SCENARIOS
output_text1 = f"Person Depressed: {depressed_scenario1} "
output_text2 = f"Person Depressed: {depressed_scenario2}"

print("\nChances for Scenario 1:")
print(output_text1)
print("\nChances for Scenario 2:")
print(output_text2)

x_test_list = x_test.values.tolist()
y_test_list = y_test.values.tolist()

errors = 0

for i in range(len(x_test)):
    prediction = calc_probability_depressed(tuple(x_test_list[i]))
   
    if(prediction == True and y_test_list[i] == "No"):
        errors = errors + 1
    elif(prediction == False and y_test_list[i] == "Yes"):
        errors = errors + 1
    
error_rate = errors / len(x_test);
print("Error rate: " + str(error_rate))

