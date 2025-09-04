import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, shapiro, anderson, kstest, normaltest
import streamlit as st


# TODO: add count of NaN, zeros and correlation coeff for categorical features (anova or kruskal wallis)

# load the data
train = pd.read_csv("data/train.csv", header=0)
test = pd.read_csv("data/test.csv", header=0)

# update data types
train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF']] = train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF']].astype('float64')
test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']] = test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']].fillna(0).astype('int64')

# define qualitative and quantitative columns
quantitative = train.select_dtypes(include=['float', 'int']).columns.tolist()
qualitative = train.select_dtypes(include=['object']).columns.tolist()

quantitative.remove('SalePrice')
quantitative.remove('Id')

# these features should be qualitative, not quantitative.
additional_qualitative = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath',
                          'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                          'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

for col in additional_qualitative:
    quantitative.remove(col)

qualitative = qualitative + additional_qualitative

all_features = qualitative + quantitative
all_features.sort()

def plot_univariate(df, col, ax=None, color='#1f77b4', alpha=1.0, label=None, is_shaded=False):
    """ plot histogram of column col on specified axes with customizable color and transparency """
    if ax is None:
        ax = plt.gca()
    
    if col in qualitative:
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
        
    if col in qualitative:
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
        results['correlation'] = df[[col, 'SalePrice']].corr().iloc[0, 1]
      
    else:
        results['mode'] = df[col].mode().iloc[0]

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
    cols = st.columns(3, gap="medium")
    
    with cols[0]:      
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train, 'SalePrice', ax=ax, color='skyblue', is_shaded=False, label='SalePrice')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with cols[1]:
    
        st.metric("Max", f"{results['max']}")
        st.metric("Min", f"{results['min']}")
        st.metric("Mean", f"{results['mean']:0.0f}")
    
    with cols[2]:
    
        st.metric("Skewness", f"{results['skew']:0.1f}")
        st.metric("Kurtosis", f"{results['kurt']:0.1f}")
        st.metric("Normally distributed", results['normal'])
        st.metric("Correlation coeff with SalePrice", results['correlation'])
    

        
st.header('Univariate Analysis')

selected_col = st.selectbox('Select a feature to analyze:', all_features)

train_color = '#1f77b4'  # Blue
test_color = '#ff7f0e'   # Orange

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(train, selected_col, ax=ax, color=train_color, alpha=0.8, label='Train', is_shaded=False)
        plt.title('Train set')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with cols[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_univariate(test, selected_col, ax=ax, color=test_color, alpha=0.6, label='Test', is_shaded=True)
        plt.title('Test set')
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
            stats.probplot(test[selected_col], dist="norm", plot=ax)
            ax.set_title('Probability plot - test set')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)     

    
        

st.header('Bivariate Analysis')

metrics_results = calc_metrics(train, selected_col)

with st.container(horizontal=True, gap="medium"):
    cols = st.columns(2, gap="medium")
    
    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_bivariate(train, selected_col, ax=ax)
        plt.title('Train set')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    with cols[1]:
        if selected_col in quantitative:
            
            st.metric("Max", f"{metrics_results['max']}")
            st.metric("Min", f"{metrics_results['min']}")
            st.metric("Mean", f"{metrics_results['mean']:0.0f}")
            st.metric("Skewness", f"{metrics_results['skew']:0.1f}")
            st.metric("Kurtosis", f"{metrics_results['kurt']:0.1f}")
            st.metric("Normally distributed", metrics_results['normal'])
            st.metric("Correlation coeff with SalePrice", metrics_results['correlation'])
            
        else:
            st.metric("Mode", f"{metrics_results['mode']}")
        
            
        
 
