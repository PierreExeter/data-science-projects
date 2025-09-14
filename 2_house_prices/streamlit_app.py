import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm, shapiro, anderson, kstest, normaltest, f_oneway, kruskal, pearsonr, spearmanr
import streamlit as st



# load the data
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)
feat_desc = pd.read_csv("data/features_description.csv", header=0)

train_clean = pd.read_csv("data/train_cleaned.csv", header=0)
test_clean = pd.read_csv("data/test_cleaned.csv", header=0)

# update data types
train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF']] = train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF']].astype('float64')
test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']] = test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']].fillna(0).astype('int64')

# define qualitative and quantitative columns
quantitative = [
    'LotFrontage', 
    'LotArea', 
    'YearBuilt', 
    'YearRemodAdd', 
    'MasVnrArea', 
    'BsmtFinSF1', 
    'BsmtFinSF2', 
    'BsmtUnfSF', 
    'TotalBsmtSF', 
    '1stFlrSF', 
    '2ndFlrSF', 
    'LowQualFinSF', 
    'GrLivArea', 
    'GarageYrBlt', 
    'GarageArea', 
    'WoodDeckSF', 
    'OpenPorchSF', 
    'EnclosedPorch', 
    '3SsnPorch', 
    'ScreenPorch', 
    'PoolArea', 
    'MiscVal']
    
quantitative_clean = [
 'LotFrontage',
 'LotArea',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'House_Age',
 'House_Age_Squared',
 'Garage_Age',
 'Garage_Age_Squared',
 'Remodel_Age',
 'Remodel_Age_Squared']
 

qualitative = [
 'MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition',
 'MSSubClass',
 'OverallQual',
 'OverallCond',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'MoSold',
 'YrSold']
 
qualitative_clean = [
 'MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition',
 'MSSubClass',
 'OverallQual',
 'OverallCond',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'MoSold',
 'YrSold',
 '2ndFlrSF_cat',
 '3SsnPorch_cat',
 'BsmtFinSF1_cat',
 'BsmtFinSF2_cat',
 'BsmtUnfSF_cat',
 'EnclosedPorch_cat',
 'GarageArea_cat',
 'LowQualFinSF_cat',
 'MasVnrArea_cat',
 'MiscVal_cat',
 'OpenPorchSF_cat',
 'PoolArea_cat',
 'ScreenPorch_cat',
 'TotalBsmtSF_cat',
 'WoodDeckSF_cat',
 'House_Decade_Built',
 'Garage_Decade_Built',
 'Remodel_Decade_Built']

all_features = qualitative + quantitative
all_features_clean = qualitative_clean + quantitative_clean
all_features.sort()
all_features_clean.sort()

def plot_univariate(df, col, ax=None, color='#1f77b4', alpha=1.0, label=None, is_shaded=False):
    """ plot histogram of column col on specified axes with customizable color and transparency """
    if ax is None:
        ax = plt.gca()
    
    if col in qualitative or col in qualitative_clean:
        # For categorical variables - handle both solid and shaded properly
        value_counts = df[col].value_counts().sort_index()
        x_pos = np.arange(len(value_counts))
        
        if is_shaded:
            # Shaded version for categorical
            ax.bar(x_pos, value_counts.values, alpha=alpha, color=color, 
                   label=label, edgecolor='none', width=0.8)
        else:
            # Solid version for categorical
            ax.bar(x_pos, value_counts.values, alpha=alpha, color=color, 
                   label=label, edgecolor='black', linewidth=1, width=0.6)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(value_counts.index.astype(str))
        
    else:
        # For numerical variables
        if is_shaded:
            sns.histplot(df[col], kde=True, ax=ax, color=color, alpha=alpha, 
                        label=label, fill=True, stat='density')
        else:            
            sns.histplot(df[col], kde=True, ax=ax, color=color, alpha=alpha, 
                        label=label, stat='density')
        
        # Add normal distribution for reference
        mu, sigma = df[col].mean(), df[col].std()
        x = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='normal distribution')


    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel(col)
    
    if label:
        ax.legend()

        
        
def plot_bivariate(df, col, ax=None):
    """ plot scatter or boxplot of SalePrice vs col on specified axes """

    if ax is None:
        ax = plt.gca()
        
    if col in qualitative or col in qualitative_clean:
        # for categorical variables
        sns.boxplot(x=col, y='SalePrice', data=df, ax=ax)
    else:
        # for numerical variables
        sns.regplot(x=col, y='SalePrice', data=df, line_kws={'color': 'red'}, ax=ax)

    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel(col)


def test_normality(df, col, alpha=0.05):
    """ return True if feature col is normality distributed """

    clean_data = df[col].dropna()
    shapiro_stat, shapiro_p = shapiro(clean_data)
    return shapiro_p > alpha
    


def correlation_categorical(df, cat_col, alpha=0.05):
    """
    Check relationship between categorical and 'SalePrice' (numerical) 
    using ANOVA (if normal distribution) and kruskal wallis (non normally distributed)
    """
    
    # Group data by categories
    groups = [group[1]['SalePrice'].values for group in df.groupby(cat_col)]
    
    if len(groups) < 2:
        return None, {"error": "Need at least 2 categories"}
    
    # anova
    f_stat, p_value_anova = f_oneway(*groups)
    
    # kruskal wallis
    h_stat, p_value_kw = kruskal(*groups)
    
    return p_value_anova < alpha, p_value_kw < alpha


def correlation_numerical(df, col, alpha=0.05):
    """
    Check linear correlation between two numeric features using Pearson correlation
    """
    
    data = df[[col, 'SalePrice']].dropna()
    
    # Pearson (Linear Relationship, variables are normally distributed)
    corr, p_value_pearson = pearsonr(data[col], data['SalePrice'])
    
    # Spearman (Monotonic Relationship)
    corr, p_value_spearman = spearmanr(data[col], data['SalePrice'])
    
    return p_value_pearson < alpha, p_value_spearman < alpha
 



def calc_metrics(df, col):
    """ calculate metrics for dataframe df and feature col """

    results = {}

    if col in quantitative or col == 'SalePrice':
        results['max'] = df[col].max()
        results['min'] = df[col].min()
        results['mean'] = df[col].dropna().mean()
        results['skew'] = df[col].dropna().skew()
        results['kurt'] = df[col].dropna().kurt()
        results['normal'] = test_normality(df, col)
        results['correlation_pearson'], results['correlation_spearman'] = correlation_numerical(df, col)
        results['zeros_%'] = ((df[col] == 0).sum() / len(df[col]) * 100).round(1)
        results['NaN_%'] = (df[col].isna().sum() / len(df[col]) * 100).round(1)
         
    else:
    	# for qualitative features
        results['mode'] = df[col].mode().iloc[0]
        results['zeros_%'] = ((df[col] == 0).sum() / len(df[col]) * 100).round(1)
        results['NaN_%'] = (df[col].isna().sum() / len(df[col]) * 100).round(1)
        results['correlation_anova'], results['correlation_kw'] = correlation_categorical(df, col)

    if col != 'SalePrice':
        results['test_zeros_%'] = ((test[col] == 0).sum() / len(test[col]) * 100).round(1)
        results['test_NaN_%'] = (test[col].isna().sum() / len(test[col]) * 100).round(1)

    return results
   
  
   

### STREAMLIT APP


st.set_page_config(
    # Title and icon for the browser's tab bar:
    page_title="House Prices Dashboard",
    page_icon="🏠",
    # Make the content take up the width of the page:
    layout="wide")


st.title("House Prices Dashboard")


st.header('Target feature : SalePrice')

results = calc_metrics(train, 'SalePrice')

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:      
        st.header('Orginal data')
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train, 'SalePrice', ax=ax, color='skyblue', is_shaded=False, label='SalePrice')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with cols[1]:      
        st.header('Clean data')
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train_clean, 'SalePrice', ax=ax, color='skyblue', is_shaded=False, label='SalePrice')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        
with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
            
    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(train['SalePrice'], dist="norm", plot=ax)
        ax.set_title('Probability plot - train set')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cols[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(train_clean['SalePrice'], dist="norm", plot=ax)
        ax.set_title('Probability plot - train set')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(5, gap="medium")
    
    with cols[0]:
        st.metric("Max", f"{results['max']}")
        st.metric("Min", f"{results['min']}")
    
    with cols[1]:
        st.metric("Mean", f"{results['mean']:0.0f}")
        st.metric("Feature type", 'Quantitative')
    
    with cols[2]:
        st.metric("Skewness", f"{results['skew']:0.1f}")
        st.metric("Kurtosis", f"{results['kurt']:0.1f}")
        st.metric("Normally distributed", results['normal'])
    
    with cols[3]:
        st.metric("missing values : 0 (%)", f"{results['zeros_%']:0.1f}")
        st.metric("missing values : NaN (%)", f"{results['NaN_%']:0.1f}")

    with cols[4]:
        st.metric("Correlation with SalePrice (Pearson)", f"{results['correlation_pearson'][0]}") 
        st.metric("Correlation with SalePrice (Spearman)", f"{results['correlation_spearman'][0][0]}")

        

st.header('Select the features to analyze')        
with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")

    with cols[0]:
        selected_col = st.selectbox('from the original data set:', all_features)
    with cols[1]:
        selected_col_clean = st.selectbox('from the cleaned data set:', all_features_clean)


if selected_col in feat_desc['Feature'].values:
    selected_feature_description = feat_desc[feat_desc['Feature'] == selected_col]['Description'].iloc[0]


st.metric("Feature description: ", selected_feature_description)

st.header('Univariate Analysis')

train_color = '#1f77b4'  # Blue
test_color = '#ff7f0e'   # Orange

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:
        st.header('Orginal data')
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train, selected_col, ax=ax, color=train_color, alpha=0.8, label='Train Original', is_shaded=False)
        plt.title('Train set - original')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cols[1]:
        st.header('Clean data')
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train_clean, selected_col_clean, ax=ax, color=train_color, alpha=0.8, label='Train Clean', is_shaded=False)
        plt.title('Train set - clean')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)      

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(test, selected_col, ax=ax, color=test_color, alpha=0.6, label='Test Original', is_shaded=True)
        plt.title('Train set - Original')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cols[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(test_clean, selected_col_clean, ax=ax, color=test_color, alpha=0.6, label='Test Clean', is_shaded=True)
        plt.title('Test set - clean')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)   


if selected_col in quantitative:
        
    with st.container(horizontal=True, gap="medium"):
        cols = st.columns(2, gap="medium")
    
        with cols[0]:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(train[selected_col], dist="norm", plot=ax)
            ax.set_title('Probability plot - train set')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(train_clean[selected_col_clean], dist="norm", plot=ax)
            ax.set_title('Probability plot - train set - clean')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)     


    with st.container(horizontal=True, gap="medium"):
        cols = st.columns(2, gap="medium")
    
        with cols[0]:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(test[selected_col], dist="norm", plot=ax)
            ax.set_title('Probability plot - test set')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with cols[1]:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(test_clean[selected_col_clean], dist="norm", plot=ax)
            ax.set_title('Probability plot - test set - clean')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)     
    
st.header('Bivariate Analysis')


with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_bivariate(train, selected_col, ax=ax)
        plt.title('Train set - original')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with cols[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_bivariate(train_clean, selected_col_clean, ax=ax)
        plt.title('Train set - Clean')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


results = calc_metrics(train, selected_col)


if selected_col in quantitative:

    with st.container(horizontal=True, gap="medium"):
        cols = st.columns(5, gap="medium")
    
        with cols[0]:
            st.metric("Max", f"{results['max']}")
            st.metric("Min", f"{results['min']}")
    
        with cols[1]:
            st.metric("Mean", f"{results['mean']:0.0f}")
            st.metric("Feature type", 'Quantitative')
    
        with cols[2]:
            st.metric("Skewness", f"{results['skew']:0.1f}")
            st.metric("Kurtosis", f"{results['kurt']:0.1f}")
            st.metric("Normally distributed", results['normal'])
    
        with cols[3]:
            st.metric("missing values (train): 0 (%)", f"{results['zeros_%']:0.1f}")
            st.metric("missing values (train): NaN (%)", f"{results['NaN_%']:0.1f}")
            st.metric("missing values (test): 0 (%)", f"{results['test_zeros_%']:0.1f}")
            st.metric("missing values (test): NaN (%)", f"{results['test_NaN_%']:0.1f}")

        with cols[4]:
            st.metric("Correlation with SalePrice (Pearson)", f"{results['correlation_pearson']}") 
            st.metric("Correlation with SalePrice (Spearman)", f"{results['correlation_spearman']}")
            
else:    
    with st.container(horizontal=True, gap="medium"):
        cols = st.columns(3, gap="medium")
    
        with cols[0]:
            st.metric("Most common value", f"{results['mode']}")
            st.metric("Feature type", 'Qualitative')
            
        with cols[1]:
            st.metric("missing values (train): 0 (%)", f"{results['zeros_%']:0.1f}")
            st.metric("missing values (train): NaN (%)", f"{results['NaN_%']:0.1f}")
            st.metric("missing values (test): 0 (%)", f"{results['test_zeros_%']:0.1f}")
            st.metric("missing values (test): NaN (%)", f"{results['test_NaN_%']:0.1f}")   
             
        with cols[2]:
            st.metric("Correlation with SalePrice (ANOVA)", f"{results['correlation_anova']}") 
            st.metric("Correlation with SalePrice (Kruskal-Wallis)", f"{results['correlation_kw']}")
 
