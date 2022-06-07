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
    retention_file = st.sidebar.file_uploader("Upload Retention file:", key=1)
    if retention_file:
         # Can be used wherever a "file-like" object is accepted:
         retention = load_data(retention_file)
         files_read_in['Retention'] = retention.columns

    # SAP file upload depends on current time being run
    if st.session_state['option']=='Second term (first year)':
        sap_file = st.sidebar.file_uploader("Upload SAP file:", key=1)
        if sap_file:
            # Can be used wherever a "file-like" object is accepted:
            sap = load_data(sap_file)
            files_read_in['SAP'] = sap.columns



    # ========================== #




    # Dict of needed columns to check if user inputted all necessary columns

    cols_needed = dict()

    if st.session_state['option']=='Second term (first year)':
        cols_needed['Retention'] = []
    elif st.session_state['option']=='First term':
        cols_needed['Retention'] = []

    cols_needed['Course Designations'] = []

    cols_needed['HS GPA'] = []

    cols_needed['Scholarships'] = []

    cols_needed['Income'] = []

    cols_needed['Parent Education'] = []

    cols_needed['HS Rank'] = []

    cols_needed['SAT/ACT'] = []

    cols_needed['AP/IB/AICE'] = []

    cols_needed['Zip'] = []

    cols_needed['Google Distance'] = []

    if st.session_state['option']=='Second term (first year)':
        cols_needed['SAP'] = []






    # ========================== #

    # Analysis button
    run_analysis = st.sidebar.button('Run analysis')

    if run_analysis:
        st.session_state.button_pressed = True

    # ========================= #

    # Check for missing columns from data upload

    missing_cols = False

    if st.session_state['button_pressed']:
        for k in files_read_in.keys(): # Iterate thru each dataset
            missing_col_list = []
            for col in cols_needed[k]: # Iterate thru each col in dataset
                if col not in files_read_in[k]: # Check if needed col not in file
                    missing_col_list.append(col) 
                    missing_cols = True
            if len(missing_col_list) > 0:
                st.markdown('#### Columns missing from '+str(k)+':')
                st.markdown(missing_col_list)
                st.markdown('Please add these columns to the respective dataset.')

    # ========================= #

    # Write the dataset upload schema if any file is not uploaded
    # or the "run analysis" button is not pressed
    if not retention_file:
        st.markdown("### Dataset Upload Schemas")
        st.markdown('''Please upload the following datasets, with at least the 
            specified columns (Note: Spelling, spacing, and capitalization is important).''')
        table_schemas = open("Table_Schemas.txt", "r")
        st.markdown(table_schemas.read())








    # =============================================== #

    # Code to run after all files uploaded and user hit "Run Analysis" button


    if st.session_state['button_pressed'] and retention_file and missing_cols==False and st.session_state['option']=='First term':
        # Generate and store munged features
        # on which to run model
        munged_df = prepare_first_term(retention)

        st.write(munged_df)
        
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
            "retention_preds.csv",
            "text/csv",
            key='download-course-csv'
            )
        



    elif st.session_state['button_pressed'] and retention_file and missing_cols==False and st.session_state['option']=='Second term (first year)':
        # Generate and store munged features
        # on which to run model
        munged_df = prepare_full_year(retention)
        
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
            "retention_preds.csv",
            "text/csv",
            key='download-course-csv'
            )



    return None







# ======================== #

# Functions

def load_data(file_uploaded):
    if file_uploaded.name.split('.')[1] == 'csv':
        return pd.read_csv(file_uploaded, sep=',', encoding='utf-8')
    else:
        return pd.read_excel(file_uploaded)





def prepare_first_term(retention):

    retention = retention.drop(columns = ["NEXT_TERM", "FTIC_RETURNED_NEXT_FALL", "FTIC_RETURNED_FOR_SPRING",
        'CREDITS_TAKEN_2', 'SAT_RATE_2', 'AVG_COURSE_LEVEL_2', 'DIVS_Humanities_2', 'DIVS_Natural_Science_2', 
        'DIVS_Social_Sciences_2', 'DIVS_Other_2', 'DIVS_Interdivisional_2', 'NUM_NONGRADABLE_TAKEN_2', 
        'CONTRACT_2_GRADE', 'Art', 'Math_Science','Business','Social_Science', 'SAP_GOOD', 'ISP_PASSED',
        'NUM_VISITS', 'VIRTUAL_INTERACTIONS'])

    # # Drop Collinear Predictors
    retention = retention.drop(columns = ['FAMILY_INCOME', 'DIVS_Social_Sciences_1'])



    # Drop Spring Admits
    retention = retention.loc[~retention.SPRING_ADMIT].reset_index(drop=True)

    # Drop Spring Features
    retention.drop(columns = ['SPRING_ADMIT'], inplace = True)

    # Fill na scholarships with zero
    retention = retention.fillna({'UNSUB_FUNDS':0})

    retention = retention.dropna(subset=['SAT_RATE_1', 'CONTRACT_1_GRADE'])

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


    # -------------- #
    # REMOVE THIS!!!
    retention.drop(columns=['failed_to_grad'],inplace=True)

    # -------------- #

    return retention





def prepare_full_year(retention):
    retention = retention.drop(columns = ['Art', 'Math_Science', 'Business', 
        'Social_Science', 'NEXT_TERM', 'NUM_VISITS', 'VIRTUAL_INTERACTIONS'])

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

    retention = retention.dropna(subset=['SAT_RATE_1', 'CONTRACT_1_GRADE', 'SAT_RATE_2', 'CONTRACT_2_GRADE'])

    retention = retention.drop(columns = 'FTIC_RETURNED_NEXT_FALL')

    # =================================== #
    # Cap large outliers
    treatoutliers(retention, columns=['GPA', 'dist_from_ncf', 'TOTAL_FUNDS', 
                                      'Percent of adults with a high school diploma only, 2015-19',
                                      'Percent of adults with less than a high school diploma, 2015-19',
                                      'COUNTY_UNEMPLOYMENT_RATE', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                      'FAMILY_INCOME', 'FAMILY_CONTRIB'], factor=3)

    # -------------- #
    # REMOVE THIS!!!
    retention.drop(columns=['failed_to_grad'],inplace=True)

    # -------------- #

    return retention



def output_preds(munged_df, cat_vars_path, num_vars_path, stats_path, model_path, cats, model_type, scaler_path=None):

    munged_df['UNIV_ID'] = 'PLACEHOLDER'

    # Take IDs for prediction output
    predictions = munged_df[['UNIV_ID']]
    # DF to run model on
    munged_df = munged_df.drop(columns='UNIV_ID')

    # ================================ #
    # Read in pickled imputers
    current_path = os.getcwd()

    # imputer = MissForest(criterion=("mse","gini"), oob_score=True, random_state=22, verbose=0)

    # num_vars_path = os.path.join(current_path, num_vars_path)
    # with open(num_vars_path, 'rb') as handle:
    #     num_vars = pickle.load(handle)

    # cat_vars_path = os.path.join(current_path, cat_vars_path)
    # with open(cat_vars_path, 'rb') as handle:
    #     cat_vars = pickle.load(handle)

    # stats_path = os.path.join(current_path, stats_path)
    # with open(stats_path, 'rb') as handle:
    #     statistics = pickle.load(handle)

    # imputer.num_vars_ = num_vars
    # imputer.cat_vars_ = cat_vars
    # imputer.statistics_ = statistics


    # # Imputing
    # test_imputed = imputer.transform(munged_df)
    # munged_df = pd.DataFrame(data = test_imputed,
    #     columns = munged_df.columns)

    from sklearn.impute import KNNImputer

    imp_knn = KNNImputer(n_neighbors=5)
    test_imputed = imp_knn.fit_transform(munged_df)
    munged_df = pd.DataFrame(data = test_imputed,
        columns = munged_df.columns)
 
    st.write(munged_df)
    st.write(sklearn.__version__)

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

    st.write(munged_df)
    # Read in pickeled models
    model_path = os.path.join(current_path, model_path)
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)
    st.write(model)
    st.write(len(munged_df.columns))

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

    predictions = predictions.sort_values(by='Prob of NOT Grad. on Time', ascending=False)

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

