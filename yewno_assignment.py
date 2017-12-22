import pandas as pd
import datetime
import statsmodels.api as sm
from pytrends.request import TrendReq


_rootPath = 'C:\\Users\\shawn\\OneDrive\\Documents\\Career\\Yewno Project\\'
_inputfile = 'LaborForceUnemployment.xlsx'
_statecodes = 'statecodes.csv' # from https://download.bls.gov/pub/time.series/sm/sm.state
_nationcode = 'LNU04000000'

_statesabr = states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


def linearmodel(actualdata,inputdata,rollingwindow=12,usegoogle = True, stdize = False):
    # Run regression model across geographies and return statistics as well as fitted values
    timerange = inputdata.index.intersection(actualdata.index) # Subset to data that are available
    timerange = timerange[rollingwindow-1:]
    X = pd.get_dummies(pd.Series(timerange.month,timerange)) # Create monthly dummies

    statsdf = {}
    fitted = {}
    for geo in actualdata.columns:
        if 'MA-12Month' in X.columns: # Add moving average
            X = X.drop('MA-12Month',axis=1)
        X.insert(0,'MA-12Month',actualdata[geo].rolling(rollingwindow).mean().shift())

        if usegoogle: # Add Google data
            if 'Google' in X.columns:
                X = X.drop('Google', axis=1)
            X.insert(1,'Google',inputdata[geo])
        y = actualdata[geo].reindex(timerange)

        if stdize:
            X['Google'] = (X['Google'] - X['Google'].mean()) / X['Google'].std()
        model = sm.OLS(y,X)
        results = model.fit()
        stats = results.params
        fitted[geo] = (X*results.params).sum(axis=1) # Calculate fitted values
        stats = stats.append(results.tvalues.rename(index = lambda x:str(x)+'_Tstat'))
        stats['R2'] = results.rsquared
        statsdf[geo] = stats
    statsdf = pd.DataFrame(statsdf)
    colorder = statsdf.columns.drop('US').insert(0,'US')
    statsdf = statsdf[colorder]
    fitted = pd.DataFrame(fitted)
    fitted = fitted[colorder]
    return statsdf, fitted

def explain_unemployment(rollingwindow=12):
    # Main function
    statecodes = pd.read_csv(_rootPath + _statecodes)
    statemap = dict(zip(statecodes['Code'].apply(lambda x:'LAUST'+str(x).zfill(2)+'0000000000003'),statecodes['State'])) # Create mapping
    statemap[_nationcode] = 'US' # Add US/nation-wide to mapping
    actualdata = pd.read_excel(_rootPath + _inputfile, 'BLS Data Series', header=3, index_col=0)
    actualdata = actualdata.T
    actualdata = actualdata.rename(columns=statemap)
    statesnamemap = dict(zip(_statesabr.values(),_statesabr.keys()))
    actualdata = actualdata.rename(columns=statesnamemap) # Change from state names to abbreviations
    actualdata = actualdata.drop(['PR','DC'],axis=1) # Removing Puerto Rice and DC
    actualdata = actualdata.dropna(how='all')

    pytrend = TrendReq() # Google Trends API
    inputdata = {}
    for geo in actualdata.columns:
        pytrend.build_payload(kw_list=['unemployment'], timeframe='all', geo=geo if geo == 'US' else 'US-'+geo) # Search for unemployment per geography
        inputdata[geo] = pytrend.interest_over_time()['unemployment']
    inputdata = pd.DataFrame(inputdata)
    inputdata = inputdata.shift().dropna(how='all') # Lag data as it is saved as first of month when should be last

    sampledata = {} # Sample plot
    sampledata['Unemployment Rate'] = actualdata['US']
    sampledata['Unemployment Google Trends'] = inputdata['US']
    sampledata = pd.DataFrame(sampledata)
    sampledata.plot(secondary_y=['Unemployment Google Trends'])

    basestatsdf, basefitted = linearmodel(actualdata, inputdata, rollingwindow, usegoogle=False) # Base model
    statsdf, fitted = linearmodel(actualdata, inputdata, rollingwindow, usegoogle=True,stdize=True) # With google

    statsdf.loc['Google_Tstat'].plot(kind='bar')
    plt.tight_layout()

    sampledata = {}
    sampledata['Unemployment Rate'] = actualdata['US']
    sampledata['Fitted (Base Model)'] = basefitted['US']
    sampledata['Fitted (Including Google)'] = fitted['US']
    sampledata = pd.DataFrame(sampledata)
    sampledata.plot()

    return basestatsdf, statsdf

