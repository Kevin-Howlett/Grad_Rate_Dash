import streamlit as st
import os
import sys
import re
from io import StringIO
import numpy as np
import pandas as pd
import pickle
import sklearn
from scipy.stats import mode
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.utils.validation import check_is_fitted, check_array
from missingpy import MissForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier as RC
from sklearn.utils.extmath import softmax



from streamlit import legacy_caching
legacy_caching.clear_cache()



# ======================== #
st.set_page_config(
    page_title="Predicting NCF Grad Rate",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(PAGE_STYLE, unsafe_allow_html=True)
# ======================== #
 

def main():

    if 'button_pressed' not in st.session_state:
        st.session_state['button_pressed'] = False

    # Main panel
    st.title("Predicting NCF Grad Rate")
    

    # ======================== #

    # Side panel

    st.sidebar.title("Select Timeframe")

    option = st.sidebar.selectbox(
     'After what term are you running predictions?',
     ('First term', 'Second term (first year)'))

    st.session_state['option'] = option

    st.sidebar.write('You selected:', option)

    st.sidebar.title("Data Upload")
    
    files_read_in = dict()

    # File uploaders

    # TESTING MULTIPLE FILE UPLOAD

    MyVars = vars()

    uploaded_files = st.sidebar.file_uploader("Upload datasets:", accept_multiple_files = True)
    required_datasets = ['retention', 'course desig', 'sat', 'act', 'hs gpa', 'college gpa', 'scholarships', 'ap ib aice', 'rank', 'distance', 'zip code', 'residency', 'income', 'parent_education' ,'sap'] # partial strings to match
    for uploaded_file in uploaded_files:
        
        file_name = str(uploaded_file.name) # parse file_name such that file_name like variable names - # SAT from fall 2016.csv
        file_name = file_name.rsplit(".", 1)[0] # remove file extension - # SAT from fall 2016
        file_name = file_name.lower() # convert to lowercase - # sat from fall 2016

        # Check for ambiguous file names
        matches = []
        for keyword in required_datasets:
            if keyword in file_name:
                matches.append(keyword + ".csv")
        if len(matches) > 1:
                st.error("### Please rename {} to something like one of the following: {}".format(uploaded_file.name, matches))
                break
        

        file_name = re.sub(" ", "_", file_name) # replace spaces with underscores
        print(file_name)
        file_str_name = str(file_name) + "" # copy file_name as string
        if 'retention' in file_name:
            retention = load_data(uploaded_file)
            files_read_in['retention'] = retention.columns
        if 'course_desig' in file_name:
            course_desig = load_data(uploaded_file)
            files_read_in['course_desig'] = course_desig.columns
        if 'sat' in file_name:
            sat = load_data(uploaded_file)
            files_read_in['sat'] = sat.columns
        if 'act' in file_name:
            act = load_data(uploaded_file)
            files_read_in['act'] = act.columns
        if 'hs_gpa' in file_name:
            hs_gpa = load_data(uploaded_file)
            files_read_in['hs_gpa'] = hs_gpa.columns
        if 'college_gpa' in file_name:
            college_gpa = load_data(uploaded_file)
            files_read_in['college_gpa'] = college_gpa.columns
        if 'scholarships' in file_name:
            scholarships = load_data(uploaded_file)
            files_read_in['scholarships'] = scholarships.columns
        if 'aice' in file_name:
            ap_ib_aice = load_data(uploaded_file)
            files_read_in['ap_ib_aice'] = ap_ib_aice.columns
        if 'rank' in file_name:
            rank = load_data(uploaded_file)
            files_read_in['rank'] = rank.columns
        if 'distance' in file_name:
            distance = load_data(uploaded_file)
            files_read_in['distance'] = distance.columns
        if 'zip' in file_name:
            zip_code = load_data(uploaded_file)
            files_read_in['zip_code'] = zip_code.columns
        if 'residency' in file_name:
            residency = load_data(uploaded_file)
            files_read_in['residency'] = residency.columns
        if 'income' in file_name:
            income = load_data(uploaded_file)
            files_read_in['income'] = income.columns
        if 'education' in file_name:
            parent_education = load_data(uploaded_file)
            files_read_in['parent_education'] = parent_education.columns
        if 'sap' in file_name:
            sap = load_data(uploaded_file)
            files_read_in['sap'] = sap.columns
            



    # ========================== #




    # Dict of needed columns to check if user inputted all necessary columns

    cols_needed = dict()

    cols_needed['retention'] = ['UNIV_ID', 'ADMIT_TERM', 'ADMIT_TYPE', 'BIRTH_DATE', 'GENDER_MASTER', 'RACE_MASTER']

    cols_needed['course_desig'] = ['SQ_COUNT_STUDENT_ID', 'TERM', 'CLASS_TITLE', 'GRADABLE_INDICATOR', 'PART_TERM', 'CRS_SUBJ', 'CRS_NUMB', 'CRS_DIVS_DESC', 'ACAD_HIST_GRDE_DESC']

    cols_needed['sat'] = ['N_NUMBER', 'TEST_REQ_CD', 'TEST_SCORE_TYP', 'TEST_SCORE_N']

    cols_needed['act'] = ['UNIV_ID', 'ACT_ENGLISH', 'ACT_MATH', 'ACT_READING', 'ACT_SCIENCE']

    cols_needed['hs_gpa'] = ['UNIV_ID', 'GPA_HIGH_SCHOOL']

    cols_needed['college_gpa'] = ['N_NUMBER', 'GPA_CODE', 'GPA']

    cols_needed['scholarships'] = ['TermCode', 'SPRIDEN_ID', 'FundTitle', 'FORMATTED_PAID_AMT']

    cols_needed['ap_ib_aice'] = ['N_NUMBER', 'TEST_DESC']

    cols_needed['rank'] = ['N_NUMBER', 'HS_CLASS_RANK', 'HS_CLASS_SIZE']

    cols_needed['distance'] = ['N_NUMBER', 'dist_from_ncf']

    cols_needed['zip_code'] = ['N_NUMBER', 'ZIP']

    cols_needed['residency'] = ['N_NUMBER', 'TERM_ATTENDED', 'RESIDENCY']

    cols_needed['income'] = ['SPRIDEN_ID', 'DEMO_TIME_FRAME', 'PARENTS_INCOME', 'STUDENT_INCOME', 'FAMILY_CONTRIB']

    cols_needed['parent_education'] = ['SPRIDEN_ID', 'FatherHIGrade', 'MotherHIGrade']

    if st.session_state['option']=='Second term (first year)':
        cols_needed['retention'].append('RETURNED_FOR_SPRING')
        cols_needed['sap'] = ['N_NUMBER', 'TERM', 'SAPCODE']





    # ========================== #

    # Analysis button
    run_analysis = st.sidebar.button('Run analysis')

    if run_analysis:
        st.session_state.button_pressed = True

    # ========================= #

    # Check for missing columns from data upload

    missing_cols = False
    
    if st.session_state['button_pressed']:    # Check for missing datasets
        for k in cols_needed:
            missing_col_list = []
            if k in files_read_in:
                uploaded_cols = files_read_in[k]
            else:
                uploaded_cols = []
            for col in cols_needed[k]:
                if col not in uploaded_cols:
                    missing_cols = True
                    missing_col_list.append(col)
            if len(missing_col_list) > 0:
                st.markdown('#### Columns missing from '+str(k)+':')
                st.markdown(missing_col_list)
                st.markdown('Please add these columns to the respective dataset.')

    # ========================= #

    # Write the dataset upload schema if any file is not uploaded
    # or the "run analysis" button is not pressed
    if not st.session_state.button_pressed or (st.session_state['option']=='Second term (first year)' and len(files_read_in)<15) or (st.session_state['option']=='First term' and len(files_read_in)<14):
        st.markdown("### Dataset Upload Schemas")
        st.markdown('''Please upload the following datasets, with at least the 
            specified columns (Note: Spelling, spacing, and capitalization is important).''')
        
        if st.session_state['option'] != 'Second term (first year)':
            table_schemas = open("Table_Schemas.txt", "r")
        # Change table schema appearance if option is full year
        elif st.session_state['option']=='Second term (first year)':
            table_schemas = open("Table_Schemas_fullyear.txt", "r")


        st.markdown(table_schemas.read())








    # =============================================== #

    # Code to run after all files uploaded and user hit "Run Analysis" button


    if st.session_state['button_pressed'] and len(files_read_in) >=14 and missing_cols==False and st.session_state['option']=='First term':
        # Generate and store munged features
        # on which to run model
        retention = prepare_retention(retention, sat, act, college_gpa, hs_gpa, ap_ib_aice, 
                        distance, residency, rank, zip_code, scholarships, course_desig,
                        income, parent_education, sap=None)

        munged_df = prepare_first_term(retention)

        # Check for Missing Values
        na_df = pd.DataFrame({'Feature': munged_df.columns.tolist(), 'Prop_NA':(munged_df.isna().sum()/munged_df.shape[0])}).reset_index(drop=True)

        # Check for fully NA columns
        if (na_df['Prop_NA'] == 1).any():
            fully_na_features = na_df.loc[na_df['Prop_NA'] == 1]
            st.write('### WARNING: All values are missing for the following columns:')
            st.write(fully_na_features.rename(columns = {'Prop_NA': 'Percent_Missing'}))

        # Warn user about Features missing more than 50%
        elif (na_df['Prop_NA'] >= 0.5).any():
            half_na_features = na_df.loc[na_df['Prop_NA'] >= 0.5].sort_values(by = 'Prop_NA', ascending=False)
            st.write('### WARNING: At least 50% of values are missing for the following columns:')
            st.write(half_na_features.rename(columns = {'Prop_NA': 'Percent_Missing'}))
        


        # Generate and store predictions
        prediction_df = output_preds(munged_df,
            cat_vars_path='static/grad_rate_pickles/GradRate_first_term_cat_vars.pkl', 
            num_vars_path='static/grad_rate_pickles/GradRate_first_term_num_vars.pkl', 
            stats_path='static/grad_rate_pickles/GradRate_first_term_statistics.pkl', 
            scaler_path='static/grad_rate_pickles/GradRate_first_term_scaler.pkl',
            model_path='static/grad_rate_pickles/GradRate_first_term_model.pkl',
            model_type='ridge',
            cats=['GENDER_M', 'IS_WHITE', 'IS_TRANSFER', 'CONTRACT_1_GRADE', 'IN_STATE', 'AP_IB_AICE_FLAG'])
 
        # Display predictions
        st.write('## Predictions')
        st.write(prediction_df)

        # Convert preds to csv and download
        pred_csv = prediction_df.to_csv(index=False).encode('utf-8')

        # Download button
        st.write('### Download Predictions')
        pred_download = st.download_button(
            "Download Predictions",
            pred_csv,
            "grad_rate_preds.csv",
            "text/csv",
            key='download-course-csv'
            )
        



    elif st.session_state['button_pressed'] and len(files_read_in) >=15 and missing_cols==False and st.session_state['option']=='Second term (first year)':
        # Generate and store munged features
        # on which to run model
        retention = prepare_retention(retention, sat, act, college_gpa, hs_gpa, ap_ib_aice, 
                        distance, residency, rank, zip_code, scholarships, course_desig,
                        income, parent_education, sap)

        munged_df = prepare_full_year(retention)

        # Check for Missing Values
        na_df = pd.DataFrame({'Feature': munged_df.columns.tolist(), 'Prop_NA':(munged_df.isna().sum()/munged_df.shape[0])}).reset_index(drop=True)

        # Check for fully NA columns
        if (na_df['Prop_NA'] == 1).any():
            fully_na_features = na_df.loc[na_df['Prop_NA'] == 1]
            st.write('### WARNING: all values are missing for the following columns:')
            st.write(fully_na_features.rename(columns = {'Prop_NA': 'Percent_Missing'}))

        # Warn user about Features missing more than 50%
        elif (na_df['Prop_NA'] >= 0.5).any():
            half_na_features = na_df.loc[na_df['Prop_NA'] >= 0.5].sort_values(by = 'Prop_NA', ascending=False)
            st.write('### WARNING: at least 50% of values are missing for the following columns:')
            st.write(half_na_features.rename(columns = {'Prop_NA': 'Percent_Missing'}))
        
        # Generate and store predictions
        prediction_df = output_preds(munged_df,
            cat_vars_path='static/grad_rate_pickles/GradRate_full_year_cat_vars.pkl',
            num_vars_path='static/grad_rate_pickles/GradRate_full_year_num_vars.pkl',
            stats_path='static/grad_rate_pickles/GradRate_full_year_statistics.pkl',
            model_path='static/grad_rate_pickles/GradRate_full_year_model.pkl',
            model_type='forest',
            cats=['GENDER_M', 'IS_WHITE', 'IS_TRANSFER', 'SPRING_ADMIT', 
            'CONTRACT_1_GRADE', 'FTIC_RETURNED_FOR_SPRING', 'CONTRACT_2_GRADE', 
            'SAP_GOOD', 'ISP_PASSED', 'IN_STATE', 'AP_IB_AICE_FLAG'])
        
        # Display predictions
        st.write('## Predictions')
        st.write(prediction_df)


        # Convert preds to csv and download
        pred_csv = prediction_df.to_csv(index=False).encode('utf-8')

        # Download button
        st.write('### Download Predictions')
        pred_download = st.download_button(
            "Download Predictions",
            pred_csv,
            "grad_rate_preds.csv",
            "text/csv",
            key='download-course-csv'
            )



    return None












# =========================================================================================================== #

# FUNCTIONS
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(file_uploaded):
    if file_uploaded.name.rsplit('.', 1)[1] == 'csv':
        return pd.read_csv(file_uploaded, sep=',', encoding='utf-8')
    else:
        return pd.read_excel(file_uploaded)


def prepare_retention(retention, sat, act, college_gpa, hs_gpa, ap_ib_aice, 
    distance, residency, rank, zip_code, scholarships, course_desig,
    income, parent_education, sap=None):
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(int)

    # Filter out second baccalaureate students
    retention = retention.loc[retention['ADMIT_TYPE'] != "S"]

    # Initialize column of predicted second term for each student
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(str)
    retention['NEXT_TERM'] = np.where(retention.ADMIT_TERM.str.endswith("8"), retention.ADMIT_TERM.astype(int) + 93, np.nan)

    # Subset students after 200208
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(int)

    retention = retention.rename(columns={'UNIV_ID':'N_NUMBER'})

    # Merge SAT
    final_sat = prepare_sat(sat, act)
    retention = retention.merge(final_sat, how='left', on='N_NUMBER')

    # Merge GPA
    final_col_gpa = prepare_col_gpa(college_gpa)
    retention = retention.merge(final_col_gpa, how='left',on='N_NUMBER')

    final_hs_gpa = prepare_gpa(hs_gpa)
    retention = retention.merge(final_hs_gpa, how='left',on='N_NUMBER')

    # Merge AP/IB/AICE ap_ib_aice
    taken_advanced = prepare_tests(ap_ib_aice)
    retention['AP_IB_AICE_FLAG'] = 0
    retention.loc[retention.N_NUMBER.isin(taken_advanced), 'AP_IB_AICE_FLAG'] = 1

    # Merge distances from NCF
    distance = prepare_google_dist(distance)
    retention = retention.merge(distance, how='left', on='N_NUMBER')

    # Merge In/Out-State residency
    residency = prepare_residency(residency)
    retention = retention.merge(residency, how='left', left_on=['N_NUMBER','ADMIT_TERM'], right_on=['N_NUMBER','TERM_ATTENDED']).drop(columns='TERM_ATTENDED')

    # Merge county data
    current_path = os.getcwd()

    county_zip_path = os.path.join(current_path, 'data/COUNTY_ZIP.csv')
    with open(county_zip_path, 'rb') as handle:
        county_zip = pd.read_csv(handle)

    education_path = os.path.join(current_path, 'data/Education.csv')
    with open(education_path, 'rb') as handle:
        education = pd.read_csv(handle)

    unemployment_path = os.path.join(current_path, 'data/Unemployment.xlsx')
    with open(unemployment_path, 'rb') as handle:
        unemployment = pd.read_excel(unemployment_path, sheet_name = "Unemployment Med HH Income", skiprows = 4)


    zip_code = prepare_zips(zip_code, county_zip, education, unemployment)
    retention = retention.merge(zip_code, how='left', on='N_NUMBER')

    # Merge HS rank
    final_rank = prepare_rank(rank)
    retention = retention.merge(final_rank, how='left', on='N_NUMBER')

    # Merge scholarships
    retention = prepare_scholarships(retention, scholarships)

    # Merge course designations
    retention = prepare_course_desig(retention, course_desig, term=st.session_state['option'])



    retention.GPA_HIGH_SCHOOL.fillna(retention.College_GPA, inplace=True)
    retention.loc[(retention.ADMIT_TYPE=='T') & ~retention.College_GPA.isna(), 'GPA_HIGH_SCHOOL'] = retention['College_GPA']
    retention.drop(columns='College_GPA', inplace=True)
    retention.rename(columns={'GPA_HIGH_SCHOOL':'GPA'}, inplace=True)
    retention = retention.fillna({'TOTAL_FUNDS':0, 'UNSUB_FUNDS':0})
    retention['Admit_Age'] = (round(retention.ADMIT_TERM,-2) - round(retention.BIRTH_DATE,-4)/100)/100
    retention['SPRING_ADMIT'] = (retention.ADMIT_TERM.astype(str).str[-2:] == '01')
    retention.replace({'RACE_MASTER':{
        7:1, 6:0, 5:0, 4:0,
        3:0, 2:0, 1:0, 8:0,
        9:0
        }}, inplace=True)
    retention.rename(columns={'RACE_MASTER':'IS_WHITE'}, inplace=True)



    # Merge income
    income = prepare_income(income)
    retention = pd.merge(retention, income, left_on=['N_NUMBER', 'ADMIT_TERM'], right_on = ['SPRIDEN_ID','TERM'], how = 'left').drop(columns = ['SPRIDEN_ID','TERM'])

    # Merge parent education
    parent_education = prepare_parent_edu(parent_education)
    retention = retention.merge(parent_education, how='left', left_on='N_NUMBER', right_on='SPRIDEN_ID').drop(columns='SPRIDEN_ID')


    if st.session_state['option'] == "Second term (first year)":
        # Merge SAP
        sap = prepare_sap(sap)
        retention = pd.merge(retention, sap[['TERM','N_NUMBER','SAP_GOOD']], how = 'left', left_on = ['N_NUMBER', 'NEXT_TERM'], right_on = ['N_NUMBER', 'TERM']).rename(columns={'SAPCODE':'SAP_NEXT_TERM'}).drop(columns='TERM')


    if st.session_state['option'] == 'Second term (first year)':
        retention = retention.rename(columns = {'RETURNED_FOR_SPRING' : 'FTIC_RETURNED_FOR_SPRING'})
        retention = retention[['N_NUMBER', 'GENDER_MASTER', 'IS_WHITE', 'ADMIT_TYPE',
                                'TEST_SCORE_N', 'SAT_MATH', 'GPA', 'IN_STATE', 'AP_IB_AICE_FLAG',
                                'dist_from_ncf', 'rank_percentile', 'TOTAL_FUNDS', 'UNSUB_FUNDS',
                                'FTIC_RETURNED_FOR_SPRING', 
                                'Percent of adults with a high school diploma only, 2015-19',
                                'Percent of adults with less than a high school diploma, 2015-19',
                                'COUNTY_UNEMPLOYMENT_RATE', 'ISP_PASSED', 'NUM_NONGRADABLE_TAKEN_1',
                                'NUM_NONGRADABLE_TAKEN_2', 'CONTRACT_1_GRADE', 'CONTRACT_2_GRADE',
                                'CREDITS_TAKEN_1', 'SAT_RATE_1', 'AVG_COURSE_LEVEL_1',
                                'DIVS_Humanities_1', 'DIVS_Natural_Science_1', 'DIVS_Social_Sciences_1',
                                'DIVS_Other_1', 'DIVS_Interdivisional_1', 'CREDITS_TAKEN_2',
                                'SAT_RATE_2', 'AVG_COURSE_LEVEL_2', 'DIVS_Humanities_2',
                                'DIVS_Natural_Science_2', 'DIVS_Social_Sciences_2', 'DIVS_Other_2',
                                'DIVS_Interdivisional_2', 'Admit_Age', 'SPRING_ADMIT', 'PARENTS_INCOME',
                                'STUDENT_INCOME', 'FAMILY_CONTRIB', 'FatherHIGrade',
                                'MotherHIGrade', 'SAP_GOOD']]

    elif st.session_state['option'] == 'First term':
        retention = retention[['N_NUMBER', 'GENDER_MASTER', 'IS_WHITE', 'ADMIT_TYPE',
                                'TEST_SCORE_N', 'SAT_MATH', 'GPA', 'IN_STATE', 'AP_IB_AICE_FLAG',
                                'dist_from_ncf', 'rank_percentile', 'TOTAL_FUNDS', 'UNSUB_FUNDS',
                                'Percent of adults with a high school diploma only, 2015-19',
                                'Percent of adults with less than a high school diploma, 2015-19',
                                'COUNTY_UNEMPLOYMENT_RATE', 'NUM_NONGRADABLE_TAKEN_1',
                                'CONTRACT_1_GRADE', 'CREDITS_TAKEN_1', 'SAT_RATE_1',
                                'AVG_COURSE_LEVEL_1', 'DIVS_Humanities_1', 'DIVS_Natural_Science_1',
                                'DIVS_Social_Sciences_1', 'DIVS_Other_1', 'DIVS_Interdivisional_1',
                                'Admit_Age', 'SPRING_ADMIT', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                'FAMILY_CONTRIB', 'FatherHIGrade', 'MotherHIGrade']]


    return retention




def prepare_sat(sat, act):
    # Prepare SAT
    sat = sat.loc[sat.TEST_SCORE_TYP.isin(['M','RW']) & (sat.TEST_REQ_CD == 'S2')]

    # group by test and student ID and take max score of each subtest
    sat = sat[['N_NUMBER','TEST_REQ_CD','TEST_SCORE_TYP', 'TEST_SCORE_N']].groupby(['N_NUMBER','TEST_REQ_CD','TEST_SCORE_TYP'],as_index=False).agg({'TEST_SCORE_N':'max'})

    # Get just math scores for analysis
    sat_math = sat.loc[sat.TEST_SCORE_TYP=='M']

    sat_math = sat_math.rename(columns={'TEST_SCORE_N':'SAT_MATH'})[['N_NUMBER', 'SAT_MATH']]

    # Group by student ID and sum subscores
    sat = sat[['N_NUMBER','TEST_SCORE_N']].groupby('N_NUMBER', as_index=False).agg({'TEST_SCORE_N':'sum'})

    sat = sat.merge(sat_math, how='left', on='N_NUMBER')

    sat['TEST_SCORE_N'] = sat['TEST_SCORE_N']/2

    # Prepare ACT
    act = act[['UNIV_ID','ACT_ENGLISH','ACT_MATH','ACT_READING','ACT_SCIENCE']]

    act['ACT_Comp'] = round((act.ACT_ENGLISH + act.ACT_MATH + act.ACT_READING + act.ACT_SCIENCE)/4)

    act = act[['UNIV_ID','ACT_Comp', 'ACT_MATH']].rename(columns = {'UNIV_ID':'N_NUMBER'})

    vals = {
    36 : 1590, 35 : 1540, 34 : 1500,
    33 : 1460, 32 : 1430, 31 : 1400, 
    30 : 1370, 29 : 1340, 28 : 1310,
    27 : 1280, 26 : 1240, 25 : 1210,
    24 : 1180, 23 : 1140, 22 : 1110,
    21 : 1080, 20 : 1040, 19 : 1010,
    18 : 970, 17 : 930, 16 : 890,
    15 : 850, 14 : 800, 13 : 760,
    12 : 710, 11 : 670, 10 : 630,
    9 : 590
    }

    encodings = {'ACT_Comp': vals, 'ACT_MATH':vals}

    act = act.replace(encodings)

    act['ACT_Comp'] = act['ACT_Comp']/2
    act['ACT_MATH'] = act['ACT_MATH']/2

    act = act.rename(columns={'ACT_Comp':'TEST_SCORE_N',
                         'ACT_MATH':'SAT_MATH'})

    final_sat = pd.concat([sat, act]).groupby('N_NUMBER', as_index=False).agg({'TEST_SCORE_N':'max',
                                                                                 'SAT_MATH':'max'})

    return final_sat

def prepare_col_gpa(college_gpa):
    college_gpa = college_gpa[['N_NUMBER', 'COLLEGE_DATE', 'GPA_CODE', 'GPA']]
    fccol = college_gpa.loc[college_gpa.GPA_CODE == 'FCCOL']

    ccol = college_gpa.loc[college_gpa.GPA_CODE == 'CCOL']

    ccol = ccol.loc[~ccol.N_NUMBER.isin( list(set(fccol.N_NUMBER.values)) )]

    final_col_gpa = pd.concat([ccol, fccol], axis=0)

    final_col_gpa = final_col_gpa[['N_NUMBER','GPA']].rename(columns={'GPA':'College_GPA'})

    return final_col_gpa

def prepare_gpa(hs_gpa):
    final_hs_gpa = hs_gpa[['UNIV_ID','GPA_HIGH_SCHOOL']].rename(columns={'UNIV_ID':'N_NUMBER'})

    final_hs_gpa = final_hs_gpa.replace({0:np.nan, 9.8:np.nan})

    return final_hs_gpa

def prepare_tests(ap_ib_aice):
    ap_ib_aice = ap_ib_aice.loc[ap_ib_aice.TEST_DESC.str.startswith('AP') | ap_ib_aice.TEST_DESC.str.startswith('IB') | 
         ap_ib_aice.TEST_DESC.str.startswith('AICE')]

    taken_advanced = ap_ib_aice['N_NUMBER'].unique()

    return taken_advanced

def prepare_google_dist(distance):
    # Remove duplicate N Numbers
    distance = distance[['N_NUMBER', 'dist_from_ncf']].groupby('N_NUMBER', as_index=False).agg({'dist_from_ncf':'max'})

    return distance

def prepare_residency(residency):
    residency['IN_STATE'] = np.where(residency.RESIDENCY=='F',1,0)

    residency = residency[['N_NUMBER','TERM_ATTENDED','IN_STATE']]

    return residency

def prepare_zips(zip_code, county_zip, education, unemployment):
    zip_code['ZIP'] = zip_code['ZIP'].str.split('-').str[0]

    county_zip['zip'] = county_zip.zip.astype(str)

    zip_code = zip_code.merge(county_zip, how='left', left_on='ZIP', right_on='zip').drop(columns='zip')

    zip_code = zip_code.merge(education, how='left', left_on='county', right_on='FIPS Code')

    zip_code = zip_code[['N_NUMBER','ZIP',
       'Percent of adults with a high school diploma only, 2015-19',
       'Percent of adults with less than a high school diploma, 2015-19']]
    zip_code.loc[zip_code.duplicated(keep=False)].sort_values(by='N_NUMBER')

    zip_code = zip_code.groupby(['N_NUMBER', 'ZIP'], as_index=False).agg('mean')

    unemployment = unemployment[['FIPS_Code', 'Unemployment_rate_2019']]

    unemployment.rename(columns = {"Unemployment_rate_2019": "COUNTY_UNEMPLOYMENT_RATE"}, inplace = True)


    # Convert FIPS to ZIP
    unemployment = pd.merge(unemployment, county_zip, how = "left", left_on="FIPS_Code", right_on = "county")
    unemployment = unemployment[['zip', 'COUNTY_UNEMPLOYMENT_RATE']]

    unemployment = unemployment.dropna(subset=['zip'])

    unemployment = unemployment.groupby(['zip'], as_index=False).agg("mean")

    zip_code = pd.merge(zip_code, unemployment, how = 'left', left_on = ['ZIP'], right_on = ['zip']).drop(columns = ['zip'])

    zip_code = zip_code.groupby('N_NUMBER', as_index=False).agg('mean')

    return zip_code

def prepare_rank(rank):
    rank['rank_percentile'] = 1- (rank.HS_CLASS_RANK / rank.HS_CLASS_SIZE)

    rank = rank[['N_NUMBER', 'rank_percentile']].dropna()

    rank_dropped_dup = rank.drop_duplicates(subset=['N_NUMBER','rank_percentile'])

    final_rank = rank_dropped_dup.groupby('N_NUMBER', as_index=False).agg({'rank_percentile':'max'})

    return final_rank

def prepare_scholarships(retention, scholarships):
    scholarships = scholarships.rename(columns={'TermCode':'TERM', 'SPRIDEN_ID':'STUDENT_ID'})

    # Replace NA funds with zero
    scholarships.FORMATTED_PAID_AMT.fillna(0, inplace=True)

    # String match to extract unsubsizided funds
    unsub = scholarships.loc[scholarships.FundTitle.str.contains("Unsub", case = False)]

    # String match to extract subsidized funds
    scholarships = scholarships.loc[~scholarships.FundTitle.str.contains("Unsub", case = False)]

    # GroupBy ID/TERM to add up total funds awarded to each student, for each term
    scholarships = scholarships.groupby(["STUDENT_ID"])['FORMATTED_PAID_AMT'].agg(sum).reset_index(name='TOTAL_FUNDS')
    unsub = unsub.groupby(["STUDENT_ID"])['FORMATTED_PAID_AMT'].agg(sum).reset_index(name='TOTAL_FUNDS')

    # Subset records with non-zero funds
    scholarships = scholarships[scholarships['TOTAL_FUNDS'] > 0]
    unsub = unsub[unsub['TOTAL_FUNDS'] > 0]

    unsub.rename(columns={'TOTAL_FUNDS':'UNSUB_FUNDS'}, inplace=True)

    retention = pd.merge(retention, scholarships, left_on = ["N_NUMBER"], right_on = ["STUDENT_ID"], how = "left").drop(columns = ["STUDENT_ID"])
    retention = pd.merge(retention, unsub, left_on = ["N_NUMBER"], right_on = ["STUDENT_ID"], how = "left").drop(columns = ["STUDENT_ID"])

    return retention

def prepare_course_desig(retention, course_desig, term):
    course_desig.rename(columns = {'SQ_COUNT_STUDENT_ID' : 'ID'}, inplace=True)

    isps = course_desig.copy().loc[(course_desig.CRS_SUBJ == "ISP") & (course_desig.GRADABLE_INDICATOR == "Y")].reset_index(drop=True)

    # Replace TERMS ending in "1" with "2" (eg. 201801 -> 201802)
    course_desig.TERM = course_desig.TERM.astype(str)
    course_desig.TERM = course_desig.TERM.apply(lambda x: (x[:-1]+"1") if x.endswith("2") else x)
    course_desig.TERM = course_desig.TERM.astype(int)

    # Separate contracts out
    contract_desig = course_desig.loc[course_desig['CLASS_TITLE'].str.contains('Contract ')].rename(columns = {'SQ_COUNT_STUDENT_ID' : 'ID', 'ACAD_HIST_GRDE_DESC':'CONTRACT_GRADE'})

    make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
    contract_desig.loc[contract_desig.CONTRACT_GRADE.isin(make_unsat),'CONTRACT_GRADE'] = "Unsatisfactory"


    contract_desig.dropna(subset=['CONTRACT_GRADE'], inplace=True)
    contract_desig.drop_duplicates(subset = ['ID', 'TERM'], inplace=True)

    # Count non-gradable courses taken by each student/term AND ADD TO CONTRACT_DESIG DATAFRAME
    contract_desig = course_desig.loc[course_desig.GRADABLE_INDICATOR == "N"].groupby(["ID", "TERM"]).GRADABLE_INDICATOR.size().reset_index().rename(columns = {"GRADABLE_INDICATOR":"NUM_NONGRADABLE_TAKEN"}).merge(contract_desig, how = "right", left_on=['ID','TERM'], right_on=['ID','TERM'])

    # replace na NUM_NONGRADABLE_TAKEN with zero
    contract_desig.NUM_NONGRADABLE_TAKEN.fillna(0, inplace = True)

    # Subset relevant columns
    contract_desig = contract_desig[['ID','TERM','NUM_NONGRADABLE_TAKEN','CONTRACT_GRADE']]

    # FILTER OUT WITHDRAWN STUDENTS
    contract_desig = contract_desig.loc[contract_desig.CONTRACT_GRADE != "Withdrawn"]

    retention = pd.merge(retention, contract_desig, how = "left", left_on = ['N_NUMBER','ADMIT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

    if term == 'Second term (first year)':
        retention= pd.merge(retention, contract_desig, how = "left", left_on = ['N_NUMBER','NEXT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

        retention.rename(columns = {'CONTRACT_GRADE_x':'CONTRACT_1_GRADE','CONTRACT_GRADE_y':'CONTRACT_2_GRADE',
                            'NUM_NONGRADABLE_TAKEN_x':'NUM_NONGRADABLE_TAKEN_1',
                            'NUM_NONGRADABLE_TAKEN_y':'NUM_NONGRADABLE_TAKEN_2'}, inplace = True)
    else:
        retention.rename(columns = {'CONTRACT_GRADE':'CONTRACT_1_GRADE',
                            'NUM_NONGRADABLE_TAKEN':'NUM_NONGRADABLE_TAKEN_1'}, inplace = True)

    # Course desig
    # Subset to gradable courses
    course_desig = course_desig.loc[(course_desig.CRS_SUBJ != "NCF") & (course_desig.CRS_SUBJ != "ISP") & (course_desig.GRADABLE_INDICATOR=='Y')]

    # Subset needed columns
    course_desig = course_desig[['ID', 'TERM', 'PART_TERM', 'CRS_NUMB', 'CRS_DIVS_DESC',
                  'ACAD_HIST_GRDE_DESC']]

    # Replace course credits with floats
    course_desig = course_desig.replace({'PART_TERM':{'1':1,
                                  'M1':0.5,
                                  'M2':0.5,
                                  '1MC':0.5}})

    # Extract course_level from CRS_NUMB
    course_desig['COURSE_LEVEL'] = [int(str(x)[0]) for x in course_desig.CRS_NUMB.tolist()]
    # Group course_level 5 & 6 values in with 4
    course_desig['COURSE_LEVEL'] = np.where(course_desig['COURSE_LEVEL'].gt(4), 4, course_desig['COURSE_LEVEL'])

    course_desig = course_desig.drop(columns='CRS_NUMB')

    # Values to mark as unsat
    make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
    course_desig.loc[course_desig.ACAD_HIST_GRDE_DESC.isin(make_unsat),'ACAD_HIST_GRDE_DESC'] = "Unsatisfactory"

    # FILTER OUT RECORDS NOT IN SAT/UNSAT
    course_desig = course_desig.loc[course_desig.ACAD_HIST_GRDE_DESC.isin(['Satisfactory', 'Unsatisfactory'])]

    top_n = ['Humanities', 'Natural Science', 'Social Sciences', 'Other', 'Interdivisional']

    course_desig['CRS_DIVS_DESC'] = np.where(course_desig['CRS_DIVS_DESC'].isin(top_n), course_desig['CRS_DIVS_DESC'], "Other")

    # Encode CRS_DIVS_DESC as one-hot variables
    for n in top_n:
        
        dummy_colname = n.replace(" ", "_")
        dummy_colname = "DIVS_" + dummy_colname
        course_desig[dummy_colname] = np.where(course_desig['CRS_DIVS_DESC'] == n, 1, 0)

    course_desig = course_desig.drop(columns='CRS_DIVS_DESC')

    course_desig = course_desig.replace({'ACAD_HIST_GRDE_DESC':{'Satisfactory':1, 'Unsatisfactory':0}})

    course_desig = course_desig.groupby(['ID','TERM'],as_index=False).agg({'PART_TERM':'sum',
                                         'ACAD_HIST_GRDE_DESC':'mean',
                                         'COURSE_LEVEL':'mean',
                                         'DIVS_Humanities':'sum',
                                         'DIVS_Natural_Science':'sum',
                                         'DIVS_Social_Sciences':'sum',
                                         'DIVS_Other':'sum',
                                         'DIVS_Interdivisional':'sum'})
                                                                      

    course_desig = course_desig.rename(columns={'ACAD_HIST_GRDE_DESC':'SAT_RATE',
                                               'COURSE_LEVEL':'AVG_COURSE_LEVEL',
                                               'PART_TERM':'CREDITS_TAKEN'})

    retention = retention.merge(course_desig, how='left', left_on=['N_NUMBER','ADMIT_TERM'], right_on=['ID','TERM']).drop(columns=['ID','TERM'])
    
    if term == 'Second term (first year)':
        retention = retention.merge(course_desig, how='left', left_on=['N_NUMBER','NEXT_TERM'], right_on=['ID','TERM'], suffixes = ["_1", "_2"]).drop(columns=['ID','TERM'])
    else:
        suffixed_cols = retention[['CREDITS_TAKEN', 'SAT_RATE', 'AVG_COURSE_LEVEL', 'DIVS_Humanities', 'DIVS_Natural_Science', 'DIVS_Social_Sciences', 'DIVS_Other', 'DIVS_Interdivisional']].add_suffix("_1")
        retention = retention.drop(columns = ['CREDITS_TAKEN', 'SAT_RATE', 'AVG_COURSE_LEVEL', 'DIVS_Humanities', 'DIVS_Natural_Science', 'DIVS_Social_Sciences', 'DIVS_Other', 'DIVS_Interdivisional'])
        retention = pd.concat([retention, suffixed_cols], axis=1)

    retention = retention.replace({'CONTRACT_1_GRADE':{'Satisfactory':1, 'Unsatisfactory':0}, 'CONTRACT_2_GRADE':{'Satisfactory':1, 'Unsatisfactory':0}})

    
    # Only do this ISP munging when app is ran for the full year
    if term == 'Second term (first year)':
        # ISP munging
        # Relabel Unsatisfactory outcomes
        make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
        isps.loc[isps.ACAD_HIST_GRDE_DESC.isin(make_unsat),'ACAD_HIST_GRDE_DESC'] = "Unsatisfactory"

        # FILTER OUT RECORDS NOT IN SAT/UNSAT
        isps = isps.loc[isps.ACAD_HIST_GRDE_DESC.isin(['Satisfactory', 'Unsatisfactory'])]

        isps = isps[['ID', 'TERM', 'ACAD_HIST_GRDE_DESC']]
        isps.rename(columns = {"ACAD_HIST_GRDE_DESC":"ISP_PASSED"}, inplace = True)
        isps.ISP_PASSED = np.where(isps.ISP_PASSED == "Satisfactory", 1, 0)

        spr_admit_ids = retention.loc[~retention.ADMIT_TERM.astype(str).str.endswith('08'), 'N_NUMBER'].tolist()
        
        isps.TERM = np.where((isps.TERM%10 != 8) & (~isps.ID.isin(spr_admit_ids)), isps.TERM-94, isps.TERM)

        isps = isps.groupby(['ID', 'TERM'], as_index=False).agg('max').sort_values(by=['ID','TERM'])
        isps = isps.drop_duplicates(subset='ID', keep='first').reset_index(drop=True)

        retention = pd.merge(retention, isps, how = 'left', left_on = ['N_NUMBER','ADMIT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

    return retention

def prepare_income(income):
    income['DEMO_TIME_FRAME'] = income.DEMO_TIME_FRAME.apply(lambda x: str(x)[:4] + "08").astype("int64")
    income = income.rename(columns={"DEMO_TIME_FRAME": "TERM"})
    income.dropna(subset=['PARENTS_INCOME','STUDENT_INCOME','FAMILY_CONTRIB'], inplace=True)
    income = income[['SPRIDEN_ID', 'TERM','PARENTS_INCOME','STUDENT_INCOME','FAMILY_CONTRIB']]

    return income

def prepare_parent_edu(parent_education):
    parent_education['FatherHIGrade'] = parent_education['FatherHIGrade'].astype(str).str.extract('(\d)').astype(float)
    parent_education['MotherHIGrade'] = parent_education['MotherHIGrade'].astype(str).str.extract('(\d)').astype(float)

    parent_education = parent_education.replace({'FatherHIGrade':{3:4, 4:3},
                    'MotherHIGrade':{3:4, 4:3}})

    parent_education = parent_education.drop(columns='First_Gen_flag')

    return parent_education

def prepare_sap(sap):
    sap['TERM'] = sap.TERM.astype(str)

    replacements = {'GOOD**':'GOOD', 'WARN2':'WARN'}
    sap['TERM'] = sap.TERM.apply(lambda x: (x[:-1]+"1") if x.endswith("2") else x)

    sap['TERM'] = sap.TERM.astype(int)

    sap['SAP_GOOD'] = np.where(sap.SAPCODE=='GOOD', 1, 0)

    sap = sap[['TERM','N_NUMBER','SAP_GOOD']]

    return sap




# ============================================ #






def prepare_first_term(retention):


    # # Drop Collinear Predictors
    retention = retention.drop(columns = ['DIVS_Social_Sciences_1'])



    # Drop Spring Admits
    retention = retention.loc[~retention.SPRING_ADMIT].reset_index(drop=True)

    # Drop Spring Features
    retention.drop(columns = ['SPRING_ADMIT'], inplace = True)

    # Fill na scholarships with zero
    retention = retention.fillna({'UNSUB_FUNDS':0})

    # Replace 9.8 GPA with NA
    retention.replace({'GPA':{9.8:np.nan}}, inplace=True)

    retention = retention.replace({'failed_to_grad':{True:1, False:0},
                                   'ADMIT_TYPE':{'T':1,'F':0},
                                   'GENDER_MASTER':{'M':1,'F':0}}
                                  )
    retention.rename(columns={'ADMIT_TYPE':'IS_TRANSFER', 'GENDER_MASTER':'GENDER_M'}, inplace=True)



    # Cap outliers at avg+-3*IQR
    treatoutliers(retention, columns=['GPA', 'dist_from_ncf', 'TOTAL_FUNDS', 
                                      'Percent of adults with a high school diploma only, 2015-19',
                                      'Percent of adults with less than a high school diploma, 2015-19',
                                      'COUNTY_UNEMPLOYMENT_RATE', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                      'FAMILY_CONTRIB'], factor=3)




    return retention





def prepare_full_year(retention):

    # For students who did NOT return in Spring, fill Spring data with zeroes
    retention.loc[retention.FTIC_RETURNED_FOR_SPRING==0, ['CREDITS_TAKEN_2', 'SAT_RATE_2', 'AVG_COURSE_LEVEL_2',
           'DIVS_Humanities_2', 'DIVS_Natural_Science_2', 'DIVS_Social_Sciences_2',
           'DIVS_Other_2', 'DIVS_Interdivisional_2', 'NUM_NONGRADABLE_TAKEN_2', 'CONTRACT_2_GRADE']] = 0

    retention = retention.fillna({'UNSUB_FUNDS':0})

    retention = retention.replace({'failed_to_grad':{True:1, False:0},
                                   'SPRING_ADMIT':{True:1, False:0},
                                   'ADMIT_TYPE':{'T':1,'F':0},
                                   'GENDER_MASTER':{'M':1,'F':0}}
                                  )
    retention.rename(columns={'ADMIT_TYPE':'IS_TRANSFER', 'GENDER_MASTER':'GENDER_M'}, inplace=True)

    retention.loc[retention['SPRING_ADMIT']==1, 'NUM_NONGRADABLE_TAKEN_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'CONTRACT_2_GRADE'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'CREDITS_TAKEN_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'SAT_RATE_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'AVG_COURSE_LEVEL_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'DIVS_Humanities_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'DIVS_Natural_Science_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'DIVS_Other_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'DIVS_Interdivisional_2'] = 0
    retention.loc[retention['SPRING_ADMIT']==1, 'DIVS_Social_Sciences_2'] = 0

    retention = retention.drop(columns = ['DIVS_Natural_Science_2', 'DIVS_Natural_Science_1',
                                          'SAT_RATE_2', 'AVG_COURSE_LEVEL_2'])

    # =================================== #
    # Cap large outliers
    treatoutliers(retention, columns=['GPA', 'dist_from_ncf', 'TOTAL_FUNDS', 
                                      'Percent of adults with a high school diploma only, 2015-19',
                                      'Percent of adults with less than a high school diploma, 2015-19',
                                      'COUNTY_UNEMPLOYMENT_RATE', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                      'FAMILY_CONTRIB'], factor=3)

    # -------------- #
    # REMOVE THIS!!!
    # retention.drop(columns=['failed_to_grad'],inplace=True)

    # -------------- #

    return retention



def output_preds(munged_df, cat_vars_path, num_vars_path, stats_path, model_path, cats, model_type, scaler_path=None):

    # Take IDs for prediction output
    predictions = munged_df[['N_NUMBER']]
    # DF to run model on
    munged_df = munged_df.drop(columns='N_NUMBER')

    # ================================ #
    # Read in pickled imputers
    current_path = os.getcwd()

    imputer = MissForest(criterion=("mse","gini"), oob_score=True, random_state=22, verbose=0)

    num_vars_path = os.path.join(current_path, num_vars_path)
    with open(num_vars_path, 'rb') as handle:
        num_vars = pickle.load(handle)

    cat_vars_path = os.path.join(current_path, cat_vars_path)
    with open(cat_vars_path, 'rb') as handle:
        cat_vars = pickle.load(handle)

    stats_path = os.path.join(current_path, stats_path)
    with open(stats_path, 'rb') as handle:
        statistics = pickle.load(handle)

    imputer.num_vars_ = num_vars
    imputer.cat_vars_ = cat_vars
    imputer.statistics_ = statistics


    # Imputing
    test_imputed = imputer.transform(munged_df)
    munged_df = pd.DataFrame(data = test_imputed,
        columns = munged_df.columns)


    # ================================ #
    # Scaling numerical features
    # (only if model is ridge regression)

    x_num = munged_df.drop(columns=cats)
    x_cat = munged_df[cats]

    num_cols = x_num.columns
    
    if scaler_path!=None:
        # Read in pickled scaler
        scaler_path = os.path.join(current_path, scaler_path)
        with open(scaler_path, 'rb') as handle:
            scl = pickle.load(handle)

        x_num = pd.DataFrame(scl.transform(x_num), columns=num_cols)

        munged_df = pd.concat([x_num, x_cat], axis=1)

    # ================================ #

    # Read in pickeled models
    model_path = os.path.join(current_path, model_path)
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)

    # Predicting
    if model_type=='ridge':
        d = model.decision_function(munged_df)
        d_2d = np.c_[-d, d]
        preds = softmax(d_2d)
    elif model_type=='forest':
        preds = model.predict_proba(munged_df)

    # Take prob of leaving
    preds = [item[1] for item in preds]

    # Add to prediction df
    predictions['Prob of NOT Grad. on Time'] = preds

    predictions = predictions.sort_values(by='Prob of NOT Grad. on Time', ascending=False).reset_index(drop=True)

    return  predictions



def treatoutliers(df, columns=None, factor=3, method='IQR', treament='cap'):
    """
    Removes the rows from self.df whose value does not lies in the specified standard deviation
    :param columns:
    :param in_stddev:
    :return:
    """
#     if not columns:
#         columns = self.mandatory_cols_ + self.optional_cols_ + [self.target_col]
    if not columns:
        columns = df.columns
    
    for column in columns:
        if method == 'STD':
            permissable_std = factor * df[column].std()
            col_mean = df[column].mean()
            floor, ceil = col_mean - permissable_std, col_mean + permissable_std
        elif method == 'IQR':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
        
        if treament == 'remove':
            df = df[(df[column] >= floor) & (df[column] <= ceil)]
        elif treament == 'cap':
            df[column] = df[column].clip(floor, ceil)
            
    return None






if __name__ == "__main__":
    main()

