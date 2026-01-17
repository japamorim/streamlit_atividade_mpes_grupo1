#GRUPO 1 - JOÃO ANTONIO AMORIM; Washington Barbosa; Helivelton Barbosa; Fernanda Montenegro; Carlos Henrique; André Santos

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Beer Consumption Analysis", layout="wide")
st.title("🍺 Beer Consumption Analysis - São Paulo")

# Download latest version
@st.cache_data
def load_data():
    df = pd.read_csv( './Consumo_cerveja.csv')
    return df

df = load_data()

# Show missing values heatmap (interactive)
st.header("1. Missing Values Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Cleaning")
    missing_before = df.isnull().astype(int)
    fig_missing = px.imshow(missing_before.T, 
                            color_continuous_scale='viridis',
                            labels=dict(x="Row", y="Column", color="Missing"),
                            aspect="auto")
    fig_missing.update_layout(height=300)
    st.plotly_chart(fig_missing, use_container_width=True)

# Clean data
df = df.dropna(subset=[df.columns[0]]).copy()

with col2:
    st.subheader("After Cleaning")
    missing_after = df.isnull().astype(int)
    fig_missing2 = px.imshow(missing_after.T, 
                             color_continuous_scale='viridis',
                             labels=dict(x="Row", y="Column", color="Missing"),
                             aspect="auto")
    fig_missing2.update_layout(height=300)
    st.plotly_chart(fig_missing2, use_container_width=True)

# Data preprocessing
df['Data'] = pd.to_datetime(df['Data'])
df['dia_nome'] = df['Data'].dt.day_name()

dias_pt = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
df['Dia_Semana'] = df['Data'].dt.day_name().map(dias_pt)

temp_col = df.columns[1]
consumption_col = df.columns[6]
fds = df.columns[5]
dia = df.columns[8]

df[temp_col] = df[temp_col].str.replace(',', '.').astype(float)

# Interactive Scatter Plot: Temperature vs Consumption
st.header("2. Day of Week(mon ->sum) vs Consumption (colored by Temperature)")

fig_scatter = px.scatter(df, 
                         x=dia, 
                         y=consumption_col,
                         color=temp_col,
                         color_continuous_scale='RdYlBu_r',
                         hover_data=['Data', 'dia_nome'],
                         title='Day vs Consumption (colored by Temperature)')

# Add trend line
z = np.polyfit(df[dia], df[consumption_col], 1)
p = np.poly1d(z)
x_line = np.linspace(df[dia].min(), df[dia].max(), 100)

fig_scatter.add_trace(go.Scatter(x=x_line, y=p(x_line), 
                                  mode='lines', 
                                  name='Trend',
                                  line=dict(color='black', dash='dash', width=2)))

fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# Interactive Box Plot
st.header("3. Consumption by Day of Week (Box Plot)")

fig_box = px.box(df, 
                 x='Dia_Semana', 
                 y=consumption_col,
                 color='Dia_Semana',
                 title='Beer Consumption Distribution by Day of Week',
                 labels={'Dia_Semana': 'Day of Week'})

fig_box.update_layout(height=500, showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# Bar Chart - Total consumption by day
st.header("4. Total Consumption by Day of Week")

valores = df.groupby('dia_nome')[consumption_col].sum().reset_index()
valores_sorted = valores.sort_values(consumption_col)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sorted by Consumption")
    fig_bar1 = px.bar(valores_sorted, 
                      x='dia_nome', 
                      y=consumption_col,
                      color=consumption_col,
                      color_continuous_scale='Blues',
                      title='Total Beer Consumption by Day (Sorted)')
    fig_bar1.update_layout(height=400, yaxis_range=[600, 1700])
    st.plotly_chart(fig_bar1, use_container_width=True)

with col2:
    st.subheader("By Day Order")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    valores['dia_nome'] = pd.Categorical(valores['dia_nome'], categories=day_order, ordered=True)
    valores_ordered = valores.sort_values('dia_nome')
    
    fig_bar2 = px.bar(valores_ordered, 
                      x='dia_nome', 
                      y=consumption_col,
                      color=consumption_col,
                      color_continuous_scale='Greens',
                      title='Total Beer Consumption by Day (Weekly Order)')
    fig_bar2.update_layout(height=400)
    st.plotly_chart(fig_bar2, use_container_width=True)

# Correlation Heatmap
st.header("5. Correlation Heatmap")

numeric_df = df.select_dtypes(include=['float64', 'int64', 'bool'])
corr_matrix = numeric_df.corr()

fig_heatmap = px.imshow(corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        title='Correlation Matrix',
                        aspect='auto')

fig_heatmap.update_layout(height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Additional Interactive Features
st.header("6. Interactive Data Explorer")

# Time series
st.subheader("Consumption Over Time")
fig_time = px.line(df, x='Data', y=consumption_col, 
                   title='Beer Consumption Over Time',
                   hover_data=[temp_col, 'dia_nome'])
fig_time.update_layout(height=400)
st.plotly_chart(fig_time, use_container_width=True)

# Sidebar filters
st.sidebar.header("Filters")
selected_days = st.sidebar.multiselect(
    "Select Days of Week",
    options=df['dia_nome'].unique(),
    default=df['dia_nome'].unique()
)

temp_range = st.sidebar.slider(
    "Temperature Range",
    float(df[temp_col].min()),
    float(df[temp_col].max()),
    (float(df[temp_col].min()), float(df[temp_col].max()))
)

# Filtered data visualization
filtered_df = df[
(df['dia_nome'].isin(selected_days)) &
(df[temp_col] >= temp_range[0]) &
(df[temp_col] <= temp_range[1])
]

st.subheader("Filtered Data Scatter Plot")
fig_filtered = px.scatter(filtered_df,
                          x=temp_col,
                          y=consumption_col,
                          color='dia_nome',
                          hover_data=['Data'],
                          title=f'Temperature vs Consumption (Filtered: {len(filtered_df)} records)')
st.plotly_chart(fig_filtered, use_container_width=True)

# Show data table
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)
