#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pytrends plotly pandas vaderSentiment')


# In[3]:


import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px

#import libraries


# In[16]:


# Define candy companies and their top brands

company_brands = {
    "Mars Wrigley": ["m&m's", "snickers", "twix"],
    "Hershey": ["reese's", "hershey's", "kitkat"],
    "Ferrero": ["butterfinger", "crunch", "trolli"]
}

# Flatten the brand list for Google Trends search

brands = [brand for company in company_brands.values() for brand in company]
brands


# In[18]:


from pytrends.request import TrendReq
import pandas as pd
import time
import random
import os

# ✅ Companies and top brands
company_brands = {
    "Mars Wrigley": ["m&m's", "snickers", "twix"],
    "Hershey": ["reese's", "kitkat", "hershey's"],
    "Ferrero": ["butterfinger", "crunch", "trolli"]
}

# ✅ Google Trends setup
pytrends = TrendReq(hl='en-US', tz=360)

# Folder for cached results
os.makedirs("trend_data", exist_ok=True)
all_data = []

# Loop through each brand
for company, brands in company_brands.items():
    for brand in brands:
        file_path = f"trend_data/{brand.replace(' ', '_')}.csv"

        # Skip if already fetched
        if os.path.exists(file_path):
            print(f"📁 Cached: {brand} already exists. Skipping.")
            df = pd.read_csv(file_path)
            all_data.append(df)
            continue

        success = False
        retries = 0

        while not success and retries < 3:
            try:
                print(f"🔍 Fetching data for: {brand} ({company}) [Attempt {retries+1}]")
                pytrends.build_payload([brand], timeframe='2024-10-01 2024-10-31', geo='US')
                df = pytrends.interest_by_region(resolution='STATE', inc_low_vol=True, inc_geo_code=False)
                df = df.reset_index().rename(columns={'geoName': 'state', brand: 'popularity_score'})
                df['brand'] = brand
                df['company'] = company

                # Save and append
                df.to_csv(file_path, index=False)
                all_data.append(df)
                success = True
                print(f"✅ {brand} fetched successfully and cached.\n")

                # Random sleep to prevent 429
                sleep_time = random.uniform(25, 60)
                print(f"🕒 Sleeping {round(sleep_time,1)} seconds...\n")
                time.sleep(sleep_time)

            except Exception as e:
                retries += 1
                print(f"⚠️ Error fetching {brand}: {e}")
                if retries < 3:
                    wait = random.uniform(60, 120)
                    print(f"⏳ Waiting {round(wait,1)} seconds before retry...\n")
                    time.sleep(wait)
                else:
                    print(f"❌ Skipping {brand} after {retries} failed attempts.\n")

# Combine all data
if all_data:
    popularity_df = pd.concat(all_data, ignore_index=True)
    popularity_df.to_csv("brand_popularity_by_state.csv", index=False)
    print(f"\n✅ All data combined! Rows: {len(popularity_df)}, States: {popularity_df['state'].nunique()}")
else:
    print("❌ No data fetched.")


# In[19]:


import pandas as pd

popularity_df = pd.read_csv("brand_popularity_by_state.csv")

print(popularity_df['state'].nunique())  # should be 50
print(popularity_df['brand'].unique())   # should show 9 brands
print(popularity_df.head())


# In[20]:


df = pd.read_csv("brand_popularity_by_state.csv")
df.head()


# In[21]:


popularity_df.groupby('brand').head(2)


# In[22]:


from IPython.display import FileLink

# Create a clickable download link
FileLink("brand_popularity_by_state.csv")


# In[23]:


# Check number of states and brands
print("Unique states:", popularity_df['state'].nunique())  # Should be 50
print("Brands:", popularity_df['brand'].unique())         # Should be 9 brands

# Preview 2 rows per brand
popularity_df.groupby('brand').head(2)


# In[24]:


import plotly.express as px

selected_brand = "reese's"  # change to any brand

fig = px.choropleth(
    popularity_df[popularity_df['brand'] == selected_brand],
    locations='state',
    locationmode="USA-states",
    color='popularity_score',
    scope="usa",
    color_continuous_scale="Oranges",
    title=f"{selected_brand.title()} Popularity by State (Halloween 2024)"
)
fig.show()


# In[25]:


# Average popularity by company per state
company_state_df = (
    popularity_df.groupby(['state', 'company'])['popularity_score']
    .mean()
    .reset_index()
)

# Find top company per state
top_company_state = (
    company_state_df.loc[company_state_df.groupby('state')['popularity_score'].idxmax()]
)

fig = px.choropleth(
    top_company_state,
    locations='state',
    locationmode="USA-states",
    color='company',
    scope="usa",
    title="Top Candy Company by State (Halloween 2024)"
)
fig.show()


# In[27]:


top_brands = popularity_df.groupby('brand')['popularity_score'].mean().sort_values(ascending=False)
top_brands.head(5).plot(kind='bar', title="Top 5 Candy Brands by Popularity (Halloween 2024)")


# In[28]:


top_companies = popularity_df.groupby('company')['popularity_score'].mean().sort_values(ascending=False)
top_companies.plot(kind='bar', title="Top Candy Companies by Popularity (Halloween 2024)")


# In[29]:


import plotly.express as px

# Mapping full state names → abbreviations
us_state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL',
    'Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE',
    'Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC',
    'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
    'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# Add abbreviation column
popularity_df['state_code'] = popularity_df['state'].map(us_state_abbrev)


# In[30]:


selected_brand = "reese's"

fig = px.choropleth(
    popularity_df[popularity_df['brand'] == selected_brand],
    locations='state_code',       # Use state abbreviations
    locationmode="USA-states",
    color='popularity_score',
    scope="usa",
    color_continuous_scale="Oranges",
    title=f"{selected_brand.title()} Popularity by State (Halloween 2024)"
)
fig.show()


# In[31]:


company_state_df = popularity_df.groupby(['state_code', 'company'])['popularity_score'].mean().reset_index()

# Find top company per state
top_company_state = company_state_df.loc[company_state_df.groupby('state_code')['popularity_score'].idxmax()]

fig = px.choropleth(
    top_company_state,
    locations='state_code',
    locationmode="USA-states",
    color='company',
    scope="usa",
    title="Top Candy Company by State (Halloween 2024)"
)
fig.show()


# In[32]:


# Mapping full state names → abbreviations
us_state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL',
    'Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE',
    'Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC',
    'North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
    'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

popularity_df['state_code'] = popularity_df['state'].map(us_state_abbrev)


# In[33]:


import plotly.express as px

selected_brand = "reese's"

fig = px.choropleth(
    popularity_df[popularity_df['brand'] == selected_brand],
    locations='state_code',          # state abbreviations
    locationmode="USA-states",
    color='popularity_score',
    hover_name='state',              # shows state name on hover
    hover_data={
        'brand': True,
        'company': True,
        'popularity_score': True,
        'state_code': False          # hide the code itself
    },
    scope="usa",
    color_continuous_scale="Oranges",
    title=f"{selected_brand.title()} Popularity by State (Halloween 2024)"
)
fig.show()


# In[34]:


# Average popularity by company per state
company_state_df = popularity_df.groupby(['state_code','company'])['popularity_score'].mean().reset_index()

# Top company per state
top_company_state = company_state_df.loc[company_state_df.groupby('state_code')['popularity_score'].idxmax()]

fig = px.choropleth(
    top_company_state,
    locations='state_code',
    locationmode="USA-states",
    color='company',
    hover_name='state_code',
    hover_data={
        'company': True,
        'popularity_score': True
    },
    scope="usa",
    title="Top Candy Company by State (Halloween 2024)"
)
fig.show()


# In[35]:


import plotly.express as px

selected_brand = "reese's"

fig = px.choropleth(
    popularity_df[popularity_df['brand'] == selected_brand],
    locations='state_code',          # still need codes for map
    locationmode="USA-states",
    color='popularity_score',
    hover_name='state',              # show full state name
    hover_data={
        'brand': True,
        'company': True,
        'popularity_score': True,
        'state_code': False          # hide the code itself
    },
    scope="usa",
    color_continuous_scale="Oranges",
    title=f"{selected_brand.title()} Popularity by State (Halloween 2024)"
)
fig.show()


# In[36]:


# Average popularity by company per state
company_state_df = popularity_df.groupby(['state','state_code','company'])['popularity_score'].mean().reset_index()

# Top company per state
top_company_state = company_state_df.loc[company_state_df.groupby('state_code')['popularity_score'].idxmax()]

fig = px.choropleth(
    top_company_state,
    locations='state_code',       # still needed for map
    locationmode="USA-states",
    color='company',
    hover_name='state',           # show full state name
    hover_data={
        'company': True,
        'popularity_score': True
    },
    scope="usa",
    title="Top Candy Company by State (Halloween 2024)"
)
fig.show()


# In[37]:


# Group by state, company, brand and calculate average popularity (in case of multiple entries)
statewise_df = (
    popularity_df.groupby(['state','company','brand'])['popularity_score']
    .mean()
    .reset_index()
)

# Sort by state and company
statewise_df = statewise_df.sort_values(['state','company','brand'])
statewise_df.head(10)


# In[38]:


# Pivot: index = state, columns = company + brand, values = popularity_score
report_df = statewise_df.pivot_table(
    index='state',
    columns=['company','brand'],
    values='popularity_score'
)

# Optional: fill missing values with 0
report_df = report_df.fillna(0)
report_df.head()


# In[39]:


report_df.to_csv("statewise_brand_company_report.csv")


# In[40]:


from IPython.display import FileLink

# Create a clickable download link
FileLink("statewise_brand_company_report.csv")


# In[41]:


# Add total popularity per company per state
company_totals = statewise_df.groupby(['state','company'])['popularity_score'].sum().reset_index()
company_totals.head()


# In[42]:


import pandas as pd

# Sum popularity scores per company per state
company_totals = popularity_df.groupby(['state','company'])['popularity_score'].sum().reset_index()

# Add state codes for plotting
company_totals['state_code'] = company_totals['state'].map(us_state_abbrev)


# In[43]:


# Function to create hover text
def make_hover_text(df):
    hover_text = []
    for state, group in df.groupby('state'):
        state_texts = []
        for company, comp_group in group.groupby('company'):
            brands_text = ", ".join([f"{b}({round(p,1)})" for b,p in zip(comp_group['brand'], comp_group['popularity_score'])])
            state_texts.append(f"{company}: {brands_text}")
        hover_text.append((state, "<br>".join(state_texts)))
    return dict(hover_text)

hover_dict = make_hover_text(popularity_df)


# In[44]:


import plotly.express as px

# Pick a company to plot (or all via color)
fig = px.choropleth(
    company_totals,
    locations='state_code',
    locationmode='USA-states',
    color='popularity_score',
    scope='usa',
    hover_name='state',
    hover_data={'company':True, 'popularity_score':True},
    color_continuous_scale='Viridis',
    title="Statewise Candy Company Popularity Totals (Halloween 2024)"
)

# Add custom hover text for brands
for i, d in enumerate(fig.data):
    state = company_totals.iloc[i]['state']
    d['hovertemplate'] = hover_dict[state] + "<extra></extra>"

fig.show()


# In[49]:


# Prepare hover text for each state + company combination
company_totals = popularity_df.groupby(['state','state_code','company']).agg({
    'popularity_score':'sum',
    'brand': lambda x: ", ".join(x)  # combine all brands under this company
}).reset_index()

# Custom hover text
company_totals['hover_text'] = company_totals.apply(
    lambda row: f"State: {row['state']}<br>Company: {row['company']}<br>Brands: {row['brand']}<br>Total Popularity: {row['popularity_score']}",
    axis=1
)


# In[50]:


import plotly.express as px

fig = px.choropleth(
    company_totals,
    locations='state_code',
    locationmode='USA-states',
    color='popularity_score',
    scope='usa',
    color_continuous_scale='Viridis',
    title="Statewise Candy Company Popularity Totals (Halloween 2024)",
    custom_data=['hover_text']  # pass hover text through custom_data
)

# Update traces to use our hover_text
fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

fig.show()


# In[54]:


# Combine all brands under each company per state
hover_all = popularity_df.groupby(['state','state_code','company']).agg({
    'brand': lambda x: ", ".join(x),
    'popularity_score': 'sum'
}).reset_index()

# Create a dictionary: state → all companies info with state name
state_hover_dict = {}
for state, group in hover_all.groupby('state'):
    lines = [f"State: {state}"]  # always show state name first
    for _, row in group.iterrows():
        lines.append(f"{row['company']}: {row['brand']} ({row['popularity_score']})")
    state_hover_dict[state] = "<br>".join(lines)


# In[55]:


# One row per state for the map
map_df = hover_all.groupby(['state','state_code']).agg({
    'popularity_score':'sum'  # total popularity across all companies
}).reset_index()

# Add the combined hover text
map_df['hover_text'] = map_df['state'].map(state_hover_dict)


# In[56]:


import plotly.express as px

fig = px.choropleth(
    map_df,
    locations='state_code',
    locationmode='USA-states',
    color='popularity_score',
    scope='usa',
    color_continuous_scale='Viridis',
    title="Statewise Candy Brand Performance (All Companies, Halloween 2024)",
    custom_data=['hover_text']  # use our combined tooltip
)

# Hover template shows state name + company + brands
fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

fig.show()


# In[57]:


# Aggregate brands per company per state
hover_all = popularity_df.groupby(['state','state_code','company']).agg({
    'brand': lambda x: ", ".join(x),
    'popularity_score': 'sum'
}).reset_index()

# Build hover text including state name
state_hover_dict = {}
for state, group in hover_all.groupby('state'):
    lines = [f"State: {state}"]  # state name
    for _, row in group.iterrows():
        lines.append(f"{row['company']}: {row['brand']} ({row['popularity_score']})")
    state_hover_dict[state] = "<br>".join(lines)


# In[58]:


# Find the company with the highest popularity per state
dominant_company = hover_all.loc[hover_all.groupby('state')['popularity_score'].idxmax()]

# Keep only state_code, dominant company, and popularity score for coloring
dominant_company_map = dominant_company[['state','state_code','company','popularity_score']].copy()

# Add hover text
dominant_company_map['hover_text'] = dominant_company_map['state'].map(state_hover_dict)


# In[59]:


import plotly.express as px

# Choose discrete color per company
fig = px.choropleth(
    dominant_company_map,
    locations='state_code',
    locationmode='USA-states',
    color='company',                 # color by dominant company
    scope='usa',
    title="Candy Company Dominance by State (Halloween 2024)",
    custom_data=['hover_text']       # hover text
)

# Show state name + all companies in hover
fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

fig.show()


# In[62]:


import pandas as pd

# Use the same popularity_df
chart_df = popularity_df[['state','company','brand','popularity_score']].copy()

# Optional: sort for consistent stacking
chart_df = chart_df.sort_values(['state','company','brand'])


# In[66]:


# Define base colors for each company
company_colors = {
    'Mars Wrigley': 'Blues',
    'Hershey': 'Reds',
    'Ferrero': 'Greens'
}


# In[67]:


import plotly.express as px

# Map each brand to a shade of the company's color
brand_colors = {}
for company in chart_df['company'].unique():
    brands = chart_df[chart_df['company']==company]['brand'].unique()
    n = len(brands)
    # Get n shades from the company base color
    color_scale = px.colors.sequential.__getattribute__(company_colors[company])[:n]
    for i, brand in enumerate(brands):
        brand_colors[f"{company} - {brand}"] = color_scale[i]


# In[68]:


import plotly.graph_objects as go

fig = go.Figure()

for company in chart_df['company'].unique():
    for brand in chart_df[chart_df['company']==company]['brand'].unique():
        df = chart_df[(chart_df['company']==company) & (chart_df['brand']==brand)]
        fig.add_trace(
            go.Bar(
                x=df['state'],
                y=df['popularity_score'],
                name=f"{company} - {brand}",
                marker_color=brand_colors[f"{company} - {brand}"],
                hovertemplate="State: %{x}<br>%{name}<br>Popularity: %{y}<extra></extra>"
            )
        )

fig.update_layout(
    barmode='stack',
    title="Statewise Candy Popularity by Company & Brand",
    xaxis_title="State",
    yaxis_title="Popularity Score",
    height=600,
    width=1200,
    xaxis_tickangle=-45,
    legend_title_text='Company - Brand',
)
fig.show()


# In[69]:


top_companies = popularity_df.groupby('company')['popularity_score'].mean().sort_values(ascending=False)
top_companies.plot(kind='bar', title="Top Candy Companies by Popularity (Halloween 2024)")


# In[70]:


top_brands = popularity_df.groupby('brand')['popularity_score'].mean().sort_values(ascending=False)
top_brands.head(5).plot(kind='bar', title="Top 5 Candy Brands by Popularity (Halloween 2024)")


# In[71]:



# Sum popularity per company and brand

company_brand_totals = popularity_df.groupby(['company','brand'])['popularity_score'].sum().reset_index()
company_brand_totals


# In[72]:


# Aggregate total popularity per company and brand
company_brand_totals = (
    popularity_df.groupby(['company','brand'])['popularity_score']
    .sum()
    .reset_index()
    .sort_values(by='popularity_score', ascending=False)  # sort descending
)

company_brand_totals


# In[73]:


company_totals = company_brand_totals.groupby('company')['popularity_score'].sum().sort_values(ascending=False)


# In[74]:


company_brand_totals['company'] = pd.Categorical(
    company_brand_totals['company'],
    categories=company_totals.index,  # companies ordered by total popularity
    ordered=True
)


# In[75]:


import plotly.express as px

fig = px.bar(
    company_brand_totals,
    x='company',
    y='popularity_score',
    color='brand',
    title="Total Candy Popularity by Company & Brand (Halloween 2024)",
    labels={'popularity_score':'Total Popularity', 'company':'Company'},
    hover_data=['brand','popularity_score']
)

fig.update_layout(
    barmode='stack',
    xaxis_title="Company",
    yaxis_title="Total Popularity",
    height=600,
    width=900
)

fig.show()


# In[76]:


# For each company, sort brands descending by popularity
company_brand_totals_sorted = (
    company_brand_totals
    .sort_values(['company','popularity_score'], ascending=[True, True])  # ascending=True puts largest at bottom in stack
)


# In[77]:


# Total popularity per company
company_totals = company_brand_totals.groupby('company')['popularity_score'].sum().sort_values(ascending=False)

# Set company as categorical for X-axis order
company_brand_totals_sorted['company'] = pd.Categorical(
    company_brand_totals_sorted['company'],
    categories=company_totals.index,
    ordered=True
)


# In[78]:


import plotly.express as px

fig = px.bar(
    company_brand_totals_sorted,
    x='company',
    y='popularity_score',
    color='brand',
    title="Total Candy Popularity by Company & Brand (Halloween 2024)",
    labels={'popularity_score':'Total Popularity', 'company':'Company'},
    hover_data=['brand','popularity_score']
)

fig.update_layout(
    barmode='stack',      # stacked bars
    xaxis_title="Company",
    yaxis_title="Total Popularity",
    height=600,
    width=900
)

fig.show()


# In[79]:


pytrends.build_payload([brand], timeframe='today 12-m')


# In[81]:


timeframe = '2025-09-01 2025-10-31'  # September 1 – October 31, 2025


# In[82]:


brand_totals = popularity_df.groupby('brand')['popularity_score'].sum().sort_values(ascending=False)
brand_totals.plot(kind='bar', title="Top Candy Brands Nationwide")


# In[83]:


import plotly.express as px
fig = px.bar(popularity_df, x='brand', y='popularity_score', color='company',
             title="Top Brands by Company (Total Popularity)")
fig.show()


# In[91]:


from pytrends.request import TrendReq
import pandas as pd
import time
import random
import os

# ✅ Companies and top brands
company_brands = {
    "Mars Wrigley": ["m&m's", "snickers", "twix"],
    "Hershey": ["reese's", "kitkat", "hershey's"],
    "Ferrero": ["butterfinger", "crunch", "trolli"]
}

# ✅ Google Trends setup
pytrends = TrendReq(hl='en-US', tz=360)

# Folder for cached results
os.makedirs("trend_data", exist_ok=True)
all_data = []

# Loop through each brand
for company, brands in company_brands.items():
    for brand in brands:
        file_path = f"trend_data/{brand.replace(' ', '_')}.csv"

        # Skip if already fetched
        if os.path.exists(file_path):
            print(f"📁 Cached: {brand} already exists. Skipping.")
            df = pd.read_csv(file_path)
            all_data.append(df)
            continue

        success = False
        retries = 0

        while not success and retries < 3:
            try:
                print(f"🔍 Fetching data for: {brand} ({company}) [Attempt {retries+1}]")
                pytrends.build_payload([brand], timeframe='2024-10-01 2025-10-31', geo='US')
                df = pytrends.interest_by_region(resolution='STATE', inc_low_vol=True, inc_geo_code=False)
                df = df.reset_index().rename(columns={'geoName': 'state', brand: 'popularity_score'})
                df['brand'] = brand
                df['company'] = company

                # Save and append
                df.to_csv(file_path, index=False)
                all_data.append(df)
                success = True
                print(f"✅ {brand} fetched successfully and cached.\n")

                # Random sleep to prevent 429
                sleep_time = random.uniform(25, 60)
                print(f"🕒 Sleeping {round(sleep_time,1)} seconds...\n")
                time.sleep(sleep_time)

            except Exception as e:
                retries += 1
                print(f"⚠️ Error fetching {brand}: {e}")
                if retries < 3:
                    wait = random.uniform(60, 120)
                    print(f"⏳ Waiting {round(wait,1)} seconds before retry...\n")
                    time.sleep(wait)
                else:
                    print(f"❌ Skipping {brand} after {retries} failed attempts.\n")

# Combine all data
if all_data:
    popularity_df = pd.concat(all_data, ignore_index=True)
    popularity_df.to_csv("brand_popularity_by_state.csv", index=False)
    print(f"\n✅ All data combined! Rows: {len(popularity_df)}, States: {popularity_df['state'].nunique()}")
else:
    print("❌ No data fetched.")


# In[92]:


# ✅ Save detailed CSV for Tableau
popularity_df.to_csv("brand_popularity_by_state_2025.csv", index=False)
print("✅ Detailed CSV saved as 'brand_popularity_by_state_2025.csv'")


# In[93]:


from IPython.display import FileLink
FileLink("brand_popularity_by_state_2025.csv")


# In[94]:


#Sentiment Analysis per candy brand


# In[97]:


# Install if not already installed
get_ipython().system('pip install nltk plotly pandas')

# Imports
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')


# In[98]:


# Example: small dummy dataset to test
df_tweets = pd.DataFrame({
    'state': ['Alabama','Alabama','Alaska','Alaska','Arizona','Arizona'],
    'brand': ["m&m's","snickers","m&m's","snickers","m&m's","snickers"],
    'text': [
        "I love m&m's so much!",
        "Snickers are okay",
        "M&m's are terrible this year",
        "I hate snickers",
        "M&m's taste amazing",
        "Snickers are the best!"
    ]
})


# In[99]:


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df_tweets['clean_text'] = df_tweets['text'].apply(clean_text)


# In[100]:


sid = SentimentIntensityAnalyzer()

# Compound score from -1 to +1
df_tweets['sentiment'] = df_tweets['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[101]:


df_tweets[['state','brand','sentiment']]


# In[117]:



import pandas as pd
import numpy as np

# Load your popularity data
popularity_df = pd.read_csv("brand_popularity_by_state_2025.csv")

# Simulate sentiment: random number between -1 (negative) and +1 (positive)
np.random.seed(42)  # for reproducibility
sentiment_df = popularity_df[['state','brand']].copy()
sentiment_df['sentiment'] = np.random.uniform(-1, 1, len(sentiment_df))

# Save to CSV (optional)
sentiment_df.to_csv("state_brand_sentiment.csv", index=False)
print("✅ Simulated sentiment saved as 'state_brand_sentiment.csv'")


# In[118]:


df = popularity_df.merge(sentiment_df, on=['state','brand'], how='left')


# In[119]:


get_ipython().system('pip install pandas plotly numpy')


# In[120]:


import pandas as pd
import numpy as np

# Load candy popularity by state (your CSV)
popularity_df = pd.read_csv("brand_popularity_by_state_2025.csv")

# Check first rows
popularity_df.head()


# In[121]:


# Random sentiment between -1 (negative) and +1 (positive)
np.random.seed(42)
sentiment_df = popularity_df[['state','brand']].copy()
sentiment_df['sentiment'] = np.random.uniform(-1, 1, len(sentiment_df))

# Merge with popularity
df = popularity_df.merge(sentiment_df, on=['state','brand'], how='left')


# In[122]:


# Sum popularity by company per state
state_company = df.groupby(['state','company'])['popularity_score'].sum().reset_index()

# Dominant company per state
state_dominance = state_company.loc[state_company.groupby('state')['popularity_score'].idxmax()]
state_dominance = state_dominance.rename(columns={'company':'dominant_company'})


# In[123]:


# Hover text showing all brands + sentiment per state
hover_text = df.groupby('state').apply(
    lambda x: '<br>'.join([f"{row['brand']} ({row['company']}): {row['sentiment']:.2f}" 
                           for idx,row in x.iterrows()])
).reset_index(name='hover_text')

# Merge hover text with dominance info
state_map = state_dominance.merge(hover_text, on='state', how='left')


# In[124]:


# Map each company to a number
company_to_code = {comp: i for i, comp in enumerate(state_map['dominant_company'].unique())}
state_map['company_code'] = state_map['dominant_company'].map(company_to_code)


# In[130]:


us_state_to_code = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
    'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# Add state_code column
state_map['state_code'] = state_map['state'].map(us_state_to_code)


# In[126]:


df.to_csv("halloween_candy_dashboard.csv", index=False)
print("✅ Dashboard-ready CSV saved as 'halloween_candy_dashboard.csv'")


# In[128]:


from IPython.display import FileLink
FileLink("halloween_candy_dashboard.csv")


# In[131]:


import plotly.express as px

company_colors = {
    "Mars Wrigley": "blue",
    "Hershey": "orange",
    "Ferrero": "green"
}

fig = px.choropleth(
    state_map,
    locations='state_code',          # <- must use state codes
    locationmode='USA-states',
    color='dominant_company',
    hover_name='state',
    hover_data=['hover_text','dominant_company'],
    scope='usa',
    color_discrete_map=company_colors
)

fig.update_layout(
    title_text="Halloween 2025 Candy Company Dominance by State",
    geo=dict(lakecolor='white')
)

fig.show()


# In[132]:


def sentiment_label(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_label'] = df['sentiment'].apply(sentiment_label)


# In[133]:


# Create hover text showing brand + sentiment category
hover_text = df.groupby('state').apply(
    lambda x: '<br>'.join([f"{row['brand']} ({row['company']}): {row['sentiment_label']}" 
                           for idx,row in x.iterrows()])
).reset_index(name='hover_text')

# Merge with dominant company and state codes
state_map = state_dominance.merge(hover_text, on='state', how='left')
state_map['state_code'] = state_map['state'].map(us_state_to_code)


# In[134]:


import plotly.express as px

company_colors = {
    "Mars Wrigley": "blue",
    "Hershey": "orange",
    "Ferrero": "green"
}

fig = px.choropleth(
    state_map,
    locations='state_code',
    locationmode='USA-states',
    color='dominant_company',
    hover_name='state',
    hover_data=['hover_text','dominant_company'],
    scope='usa',
    color_discrete_map=company_colors
)

fig.update_layout(
    title_text="Halloween 2025 Candy Company Dominance by State",
    geo=dict(lakecolor='white')
)

fig.show()


# In[135]:


us_state_to_code = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
    'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

df['state_code'] = df['state'].map(us_state_to_code)


# In[136]:


sentiment_colors = {
    "Positive": "green",
    "Neutral": "lightgray",
    "Negative": "red"
}


# In[137]:


import plotly.express as px
import os

# Folder to save maps
os.makedirs("brand_sentiment_maps", exist_ok=True)

brands = df['brand'].unique()

for brand in brands:
    df_brand = df[df['brand'] == brand]

    fig = px.choropleth(
        df_brand,
        locations='state_code',
        locationmode='USA-states',
        color='sentiment_label',           # Positive/Negative/Neutral
        hover_name='state',
        hover_data=['company','popularity_score','sentiment'],
        scope='usa',
        color_discrete_map=sentiment_colors,
        labels={'sentiment_label':'Sentiment'}
    )

    fig.update_layout(title_text=f"Halloween 2025 Sentiment for {brand}")
    
    # Save each map as HTML
    fig.write_html(f"brand_sentiment_maps/{brand}_sentiment_map.html")
    print(f"✅ Map saved for {brand}")


# In[142]:


import os
os.getcwd()


# In[144]:


import webbrowser

webbrowser.open("brand_sentiment_maps/m&m's_sentiment_map.html")


# In[145]:


# Example check
df[['state','state_code','brand','sentiment_label','company','popularity_score']].head()


# In[148]:


sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['sentiment_code'] = df['sentiment_label'].map(sentiment_map)


# In[149]:


import plotly.graph_objects as go

brands = df['brand'].unique()
fig = go.Figure()

colors = ["red", "lightgray", "green"]  # Negative, Neutral, Positive

for i, brand in enumerate(brands):
    df_brand = df[df['brand'] == brand]
    
    fig.add_trace(go.Choropleth(
        locations=df_brand['state_code'],
        z=df_brand['sentiment_code'],   # now actual sentiment
        locationmode='USA-states',
        text=[f"{row['state']}<br>{row['brand']} ({row['company']}): {row['sentiment_label']}" 
              for idx,row in df_brand.iterrows()],
        hoverinfo='text',
        colorscale=[ [0,"red"], [0.5,"lightgray"], [1,"green"] ],  # Negative, Neutral, Positive
        zmin=0,
        zmax=2,
        showscale=False,
        visible=True if i==0 else False
    ))

# Dropdown menu
dropdown_buttons = []
for i, brand in enumerate(brands):
    visibility = [False]*len(brands)
    visibility[i] = True
    dropdown_buttons.append(
        dict(label=brand,
             method="update",
             args=[{"visible": visibility},
                   {"title":f"Halloween 2025 Sentiment for {brand}"}])
    )

fig.update_layout(
    updatemenus=[dict(active=0, buttons=dropdown_buttons, x=0.1, y=1.15)],
    title="Halloween 2025 Candy Brand Sentiment by State",
    geo=dict(scope='usa', lakecolor='white')
)

fig.show()


# In[187]:


import plotly.graph_objects as go

brands = df['brand'].unique()
fig = go.Figure()

colors = ["red", "lightgray", "green"]  # Negative, Neutral, Positive
sentiment_labels = ["Negative", "Neutral", "Positive"]

# Add choropleth traces for each brand
for i, brand in enumerate(brands):
    df_brand = df[df['brand'] == brand]
    
    fig.add_trace(go.Choropleth(
        locations=df_brand['state_code'],
        z=df_brand['sentiment_code'],   # sentiment code
        locationmode='USA-states',
        text=[f"{row['state']}<br>{row['brand']} ({row['company']}): {row['sentiment_label']}" 
              for idx,row in df_brand.iterrows()],
        hoverinfo='text',
        colorscale=[[0,"red"], [0.5,"lightgray"], [1,"green"]],  # Negative, Neutral, Positive
        zmin=0,
        zmax=2,
        showscale=False,
        visible=True if i==0 else False
    ))

# Add manual legend using invisible scatter traces
for color, label in zip(colors, sentiment_labels):
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup='sentiment',
        showlegend=True,
        name=label
    ))

# Dropdown menu
dropdown_buttons = []
for i, brand in enumerate(brands):
    visibility = [False]*len(brands) + [True]*3  # Keep legend always visible
    visibility[i] = True
    dropdown_buttons.append(
        dict(label=brand,
             method="update",
             args=[{"visible": visibility},
                   {"title":f"Halloween 2025 Sentiment for {brand}"}])
    )

fig.update_layout(
    updatemenus=[dict(active=0, buttons=dropdown_buttons, x=0.1, y=1.15)],
    title="Halloween 2025 Candy Brand Sentiment by State",
    geo=dict(scope='usa', lakecolor='white')
)

fig.show()


# In[183]:


fig.write_html("candy_sentiment_Sneha_Banerjee.html", include_plotlyjs='cdn', full_html=True)


# In[184]:


fig.write_html("candy_sentiment.html")


# In[185]:


import os
print(os.getcwd())


# In[186]:


from IPython.display import FileLink

FileLink("candy_sentiment.html")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




