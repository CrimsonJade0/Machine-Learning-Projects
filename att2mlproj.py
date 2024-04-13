"""
Created on Wed Nov 15 13:41:31 2023
@author: Blair Bram
"""
import pandas as pd 
data = pd.read_csv('Mental_Health_Survey_Feb_20_22.csv')
#drop unneeded columns 
data = data.drop(['StartDate', 'EndDate', 'Status', 'Progress', 'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId','DistributionChannel', 'UserLanguage', 'ProlificID', 'BrowserInfo_Browser', 'BrowserInfo_Version', 'BrowserInfo_Operating System', 'BrowserInfo_Resolution', 'timing_intro_First Click', 'timing_intro_Last Click', 'timing_intro_Page Submit', 'timing_intro_Click Count', 'timing_consent_First Click', 'timing_consent_Last Click', 'timing_consent_Page Submit','timing_consent_Click Count', 'timing_mh_intro_First Click','timing_mh_intro_Page Submit','timing_mh_intro_Click Count', 'timing_mh_intro_Last Click','timing_phq9_Last Click','timing_phq9_First Click','timing_phq9_Last Click','timing_phq9_Page Submit','timing_phq9_Click Count','timing_gad7_First Click','timing_gad7_Last Click','timing_gad7_Page Submit', 'timing_gad7_Click Count','acha_services_1','acha_services_2','acha_services_3','attention','acha_timing_First Click','acha_timing_Last Click','acha_timing_Page Submit','acha_timing_Click Count','race_1','race_2','race_3','race_4','race_5','race_6','timing_controls_First Click','timing_controls_Last Click','timing_controls_Page Submit','timing_controls_Page Submit','timing_controls_Click Count','purpose','feedback','acha_part2_timing_First Click','acha_part2_timing_Last Click','acha_part2_timing_Page Submit','acha_part2_timing_Click Count','timing_feedback_First Click','timing_feedback_Last Click','timing_feedback_Click Count','timing_feedback_Page Submit','PROLIFIC_PID','FL_11_DO_PHQ-9','FL_11_DO_GAD-7','FL_11_DO_ACHA'], axis='columns')
data = data.drop(1,axis=0)
#class drepression y/n
yes =  data[data['acha_depression'] == 'Yes']
no =  data[data['acha_depression'] == 'No']
