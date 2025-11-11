import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create metric display card"""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

def create_bar_chart(df, x, y, title, color=None):
    """Create bar chart using plotly"""
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(template="plotly_white")
    return fig

def create_line_chart(df, x, y, title, color=None):
    """Create line chart using plotly"""
    fig = px.line(df, x=x, y=y, title=title, color=color)
    fig.update_layout(template="plotly_white")
    return fig

def create_pie_chart(df, names, values, title):
    """Create pie chart using plotly"""
    fig = px.pie(df, names=names, values=values, title=title)
    return fig