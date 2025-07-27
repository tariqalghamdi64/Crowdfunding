import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 Kickstarter Project Clustering Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample Kickstarter data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic sample data
    data = {
        'usd_goal_real': np.random.exponential(5000, n_samples),
        'usd_pledged_real': np.random.exponential(3000, n_samples),
        'main_category': np.random.choice(['Technology', 'Art', 'Music', 'Film', 'Games', 'Publishing'], n_samples),
        'days_duration': np.random.randint(15, 60, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create pledge ratio
    df['pledge_goal_ratio'] = df['usd_pledged_real'] / df['usd_goal_real']
    
    # Add some realistic patterns
    df.loc[df['main_category'] == 'Technology', 'usd_goal_real'] *= 2
    df.loc[df['main_category'] == 'Art', 'pledge_goal_ratio'] *= 1.2
    
    return df

def train_clustering_model(df):
    """Train clustering model on the data"""
    try:
        # Select numerical features for clustering
        features = ['usd_goal_real', 'usd_pledged_real', 'pledge_goal_ratio', 'days_duration']
        
        # Prepare data
        X = df[features].fillna(0).values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=100)
        labels = model.fit_predict(X_scaled)
        
        return model, labels, scaler, features
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

def get_cluster_label(cluster_id):
    """Get human-readable cluster labels"""
    cluster_labels = {
        0: "🎯 Standard Projects",
        1: "💎 Premium Projects", 
        2: "🚀 High Achievers",
        3: "⚠️ Struggling Projects",
        4: "❌ Failure Projects",
        5: "🌟 Emerging Stars",
        6: "🎨 Niche Projects",
        7: "⚖️ Balanced Projects"
    }
    return cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

def main():
    st.markdown('<h1 class="main-header">🎯 Kickstarter Project Clustering Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Overview", "🎯 Cluster Analysis", "📈 Visualizations"]
    )
    
    # Load sample data
    with st.spinner("🔄 Loading sample data..."):
        df = create_sample_data()
        st.success("✅ Sample data loaded successfully!")
    
    # Train model
    with st.spinner("🤖 Training clustering model..."):
        model, labels, scaler, features = train_clustering_model(df)
        if model is not None:
            df['Cluster'] = labels
            df['Cluster_Label'] = df['Cluster'].apply(get_cluster_label)
            st.success("✅ Model trained successfully!")
        else:
            st.error("❌ Failed to train model.")
            return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Overview":
        show_data_overview(df)
    elif page == "🎯 Cluster Analysis":
        show_cluster_analysis(df)
    elif page == "📈 Visualizations":
        show_visualizations(df)

def show_home_page(df):
    """Display the home page with overview"""
    st.markdown("## 🏠 Welcome to Kickstarter Project Analysis!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Projects", f"{len(df):,}")
    
    with col2:
        st.metric("🎯 Clusters", "8")
    
    with col3:
        avg_goal = df['usd_goal_real'].mean()
        st.metric("💰 Avg Goal", f"${avg_goal:,.0f}")
    
    with col4:
        success_rate = (df['pledge_goal_ratio'] >= 1).mean() * 100
        st.metric("✅ Success Rate", f"{success_rate:.1f}%")
    
    # Cluster overview
    st.markdown("## 🎯 Cluster Overview")
    
    cluster_summary = df.groupby('Cluster_Label').agg({
        'usd_goal_real': ['count', 'mean'],
        'pledge_goal_ratio': 'mean',
        'days_duration': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['Count', 'Avg Goal ($)', 'Avg Success Ratio', 'Avg Duration (Days)']
    cluster_summary = cluster_summary.sort_values('Count', ascending=False)
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Quick insights
    st.markdown("## 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Performing Clusters")
        top_clusters = df.groupby('Cluster_Label')['pledge_goal_ratio'].mean().sort_values(ascending=False).head(3)
        for i, (cluster, ratio) in enumerate(top_clusters.items(), 1):
            st.markdown(f"{i}. **{cluster}**: {ratio:.2f}x goal")
    
    with col2:
        st.markdown("### 📈 Largest Clusters")
        largest_clusters = df['Cluster_Label'].value_counts().head(3)
        for i, (cluster, count) in enumerate(largest_clusters.items(), 1):
            percentage = (count / len(df)) * 100
            st.markdown(f"{i}. **{cluster}**: {count:,} projects ({percentage:.1f}%)")

def show_data_overview(df):
    """Display data overview and basic statistics"""
    st.markdown("## 📊 Data Overview")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Information")
        st.write(f"**Total Projects:** {len(df):,}")
        st.write(f"**Features:** {len(df.columns)}")
        st.write(f"**Categories:** {df['main_category'].nunique()}")
    
    with col2:
        st.markdown("### 📈 Key Statistics")
        st.write(f"**Average Goal:** ${df['usd_goal_real'].mean():,.0f}")
        st.write(f"**Average Pledged:** ${df['usd_pledged_real'].mean():,.0f}")
        st.write(f"**Success Rate:** {(df['pledge_goal_ratio'] >= 1).mean() * 100:.1f}%")
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Category distribution
    st.markdown("### 📂 Category Distribution")
    category_counts = df['main_category'].value_counts()
    fig = px.bar(x=category_counts.index, y=category_counts.values, 
                 title="Projects by Category")
    st.plotly_chart(fig, use_container_width=True)

def show_cluster_analysis(df):
    """Display detailed cluster analysis"""
    st.markdown("## 🎯 Cluster Analysis")
    
    # Cluster selection
    selected_cluster = st.selectbox(
        "Select a cluster to analyze:",
        options=sorted(df['Cluster'].unique()),
        format_func=lambda x: get_cluster_label(x)
    )
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    # Cluster overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Projects", f"{len(cluster_data):,}")
    
    with col2:
        avg_goal = cluster_data['usd_goal_real'].mean()
        st.metric("💰 Avg Goal", f"${avg_goal:,.0f}")
    
    with col3:
        avg_ratio = cluster_data['pledge_goal_ratio'].mean()
        st.metric("📈 Success Ratio", f"{avg_ratio:.2f}")
    
    with col4:
        success_rate = (cluster_data['pledge_goal_ratio'] >= 1).mean() * 100
        st.metric("✅ Success Rate", f"{success_rate:.1f}%")
    
    # Detailed statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Numerical Features")
        numerical_stats = cluster_data[['usd_goal_real', 'usd_pledged_real', 'pledge_goal_ratio', 'days_duration']].describe()
        st.dataframe(numerical_stats, use_container_width=True)
    
    with col2:
        st.markdown("### 📂 Category Distribution")
        category_dist = cluster_data['main_category'].value_counts()
        fig = px.bar(x=category_dist.index, y=category_dist.values, 
                     title=f"Categories in {get_cluster_label(selected_cluster)}")
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    """Display comprehensive visualizations"""
    st.markdown("## 📈 Visualizations")
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization:",
        ["📊 Cluster Distribution", "💰 Goal Analysis", "📈 Success Analysis", "⏱️ Duration Analysis"]
    )
    
    if viz_option == "📊 Cluster Distribution":
        st.markdown("### 📊 Cluster Distribution")
        
        cluster_counts = df['Cluster_Label'].value_counts()
        
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Distribution of Projects Across Clusters"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart
        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Number of Projects by Cluster",
            labels={'x': 'Cluster', 'y': 'Number of Projects'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    elif viz_option == "💰 Goal Analysis":
        st.markdown("### 💰 Goal Analysis")
        
        fig = px.box(
            df,
            x='Cluster_Label',
            y='usd_goal_real',
            title="Goal Amount Distribution by Cluster"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average goals
        avg_goals = df.groupby('Cluster_Label')['usd_goal_real'].mean().sort_values(ascending=False)
        fig_avg = px.bar(
            x=avg_goals.index,
            y=avg_goals.values,
            title="Average Goal Amount by Cluster",
            labels={'x': 'Cluster', 'y': 'Average Goal ($)'}
        )
        fig_avg.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_avg, use_container_width=True)
    
    elif viz_option == "📈 Success Analysis":
        st.markdown("### 📈 Success Analysis")
        
        # Success ratio distribution
        fig = px.box(
            df,
            x='Cluster_Label',
            y='pledge_goal_ratio',
            title="Success Ratio Distribution by Cluster"
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="100% Success Line")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rates
        success_rates = df.groupby('Cluster_Label')['pledge_goal_ratio'].apply(
            lambda x: (x >= 1).mean() * 100
        ).sort_values(ascending=False)
        
        fig_success = px.bar(
            x=success_rates.index,
            y=success_rates.values,
            title="Success Rate by Cluster (≥100% of Goal)",
            labels={'x': 'Cluster', 'y': 'Success Rate (%)'}
        )
        fig_success.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_success, use_container_width=True)
    
    elif viz_option == "⏱️ Duration Analysis":
        st.markdown("### ⏱️ Duration Analysis")
        
        fig = px.box(
            df,
            x='Cluster_Label',
            y='days_duration',
            title="Campaign Duration Distribution by Cluster"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average duration
        avg_duration = df.groupby('Cluster_Label')['days_duration'].mean().sort_values(ascending=False)
        fig_duration = px.bar(
            x=avg_duration.index,
            y=avg_duration.values,
            title="Average Campaign Duration by Cluster",
            labels={'x': 'Cluster', 'y': 'Average Duration (Days)'}
        )
        fig_duration.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_duration, use_container_width=True)

if __name__ == "__main__":
    main() 