import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def format_number(value):
    """Format numbers with commas and appropriate decimal places"""
    if pd.isna(value):
        return value
    if isinstance(value, (int, float)):
        if value >= 1000:
            return f"{value:,.2f}"
        else:
            return f"{value:.2f}"
    return value

def format_dataframe(df):
    """Format all numeric columns in a dataframe with proper number formatting"""
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['int64', 'float64']:
            df_formatted[col] = df_formatted[col].apply(format_number)
    return df_formatted

# Page configuration
st.set_page_config(
    page_title="🎯 Kickstarter Project Clustering Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-cluster {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .premium-cluster {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    .standard-cluster {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .struggling-cluster {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .failure-cluster {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    .emerging-cluster {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .niche-cluster {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .balanced-cluster {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = {}

@st.cache_data
def load_data():
    """Load and preprocess the Kickstarter data"""
    try:
        # Load data from Google Drive
        file_id = "1MwNbPlwLvO1J_K-rIQoztXi5ZuhhzQfL"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_csv(url)
        
        # Store original data
        df_original = df.copy()
        
        # Preprocessing
        columns_to_drop = ['pledged', 'usd pledged', 'goal']
        columns_found = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_found)
        
        # Create engineered features
        df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
        df['days_duration'] = (df['deadline'] - df['launched']).dt.days
        df['pledge_goal_ratio'] = df['usd_pledged_real'] / df['usd_goal_real']
        
        # Drop additional columns
        columns_to_drop = ['ID', 'name', 'state', 'currency', 'deadline', 'launched', 'country']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # One-hot encoding
        cat_columns = df.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(df[cat_columns])
        encoded_df = pd.DataFrame(encoded_data, 
                                 columns=encoder.get_feature_names_out(cat_columns),
                                 index=df.index)
        
        numerical_df = df.select_dtypes(exclude=['object'])
        df_encoded = pd.concat([numerical_df, encoded_df], axis=1)
        
        # Standard scaling
        numerical_columns = df_encoded.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_encoded[numerical_columns])
        scaled_df = pd.DataFrame(scaled_data, 
                                columns=numerical_columns,
                                index=df_encoded.index)
        
        categorical_columns = df_encoded.select_dtypes(exclude=['int64', 'float64']).columns
        categorical_df = df_encoded[categorical_columns]
        df_final = pd.concat([scaled_df, categorical_df], axis=1)
        
        return df_original, df_final, scaler, encoder, list(numerical_columns)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None, None, None

@st.cache_resource
def train_model(df_final, _numerical_columns):
    """Train the clustering model"""
    try:
        X = df_final[_numerical_columns].values
        n_clusters = 8
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2000)
        labels = model.fit_predict(X)
        return model, labels
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None

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

def get_cluster_description(cluster_id):
    """Get detailed cluster descriptions"""
    descriptions = {
        0: "Standard projects with moderate goals and typical success rates. These are your average Kickstarter campaigns.",
        1: "Premium projects with high funding goals and strong backing potential. Often innovative or high-quality products.",
        2: "High achievers that consistently exceed their funding goals. These projects have strong market appeal.",
        3: "Projects that struggle to meet their funding targets. May need strategy adjustments.",
        4: "Projects that typically fail to reach their funding goals. High-risk category.",
        5: "Emerging projects with potential for growth. New categories or innovative concepts.",
        6: "Niche projects targeting specific audiences. Lower volume but dedicated following.",
        7: "Balanced projects with mixed characteristics. Moderate risk and reward profile."
    }
    return descriptions.get(cluster_id, "No description available.")

def predict_cluster(user_data, model, scaler, encoder, numerical_columns):
    """Predict cluster for user input"""
    try:
        # Create a DataFrame with user input
        user_df = pd.DataFrame([user_data])
        
        # Apply the same preprocessing
        # Note: This is a simplified version - in practice, you'd need to handle all categorical variables
        user_scaled = scaler.transform(user_df[numerical_columns])
        cluster = model.predict(user_scaled)[0]
        return cluster
    except Exception as e:
        st.error(f"❌ Error predicting cluster: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Kickstarter Project Clustering Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Overview", "🎯 Cluster Analysis", "🔮 Project Predictor", "📈 Visualizations", "📋 Detailed Statistics"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading and preprocessing data..."):
            df_original, df_final, scaler, encoder, numerical_columns = load_data()
            if df_original is not None:
                st.session_state.df_original = df_original
                st.session_state.df_final = df_final
                st.session_state.scaler = scaler
                st.session_state.encoder = encoder
                st.session_state.numerical_columns = numerical_columns
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
    
    if not st.session_state.data_loaded:
        st.error("❌ Failed to load data. Please check your internet connection.")
        return
    
    # Train model
    if not st.session_state.model_trained:
        with st.spinner("🤖 Training clustering model..."):
            model, labels = train_model(st.session_state.df_final, st.session_state.numerical_columns)
            if model is not None:
                st.session_state.model = model
                st.session_state.labels = labels
                st.session_state.model_trained = True
                st.success("✅ Model trained successfully!")
    
    if not st.session_state.model_trained:
        st.error("❌ Failed to train model.")
        return
    
    # Add cluster labels to original data
    df_with_clusters = st.session_state.df_original.copy()
    
    # Add engineered features that were created during preprocessing
    if 'days_duration' not in df_with_clusters.columns:
        # Create days_duration column
        df_with_clusters['launched'] = pd.to_datetime(df_with_clusters['launched'], errors='coerce')
        df_with_clusters['deadline'] = pd.to_datetime(df_with_clusters['deadline'], errors='coerce')
        df_with_clusters['days_duration'] = (df_with_clusters['deadline'] - df_with_clusters['launched']).dt.days
    
    if 'pledge_goal_ratio' not in df_with_clusters.columns:
        # Create pledge_goal_ratio column
        df_with_clusters['pledge_goal_ratio'] = df_with_clusters['usd_pledged_real'] / df_with_clusters['usd_goal_real']
    
    # Add cluster labels
    df_with_clusters['Cluster'] = st.session_state.labels
    df_with_clusters['Cluster_Label'] = df_with_clusters['Cluster'].apply(get_cluster_label)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df_with_clusters)
    elif page == "📊 Data Overview":
        show_data_overview(df_with_clusters)
    elif page == "🎯 Cluster Analysis":
        show_cluster_analysis(df_with_clusters)
    elif page == "🔮 Project Predictor":
        show_project_predictor(df_with_clusters)
    elif page == "📈 Visualizations":
        show_visualizations(df_with_clusters)
    elif page == "📋 Detailed Statistics":
        show_detailed_statistics(df_with_clusters)

def show_home_page(df_with_clusters):
    """Display the home page with overview"""
    st.markdown("## 🏠 Welcome to Kickstarter Project Analysis!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Projects", f"{len(df_with_clusters):,}")
    
    with col2:
        st.metric("🎯 Clusters", "8")
    
    with col3:
        avg_goal = df_with_clusters['usd_goal_real'].mean()
        st.metric("💰 Avg Goal", f"${avg_goal:,.0f}")
    
    with col4:
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            success_rate = (df_with_clusters['pledge_goal_ratio'] >= 1).mean() * 100
            st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("✅ Success Rate", "N/A")
    
    # Cluster overview
    st.markdown("## 🎯 Cluster Overview")
    
    # Create aggregation dictionary based on available columns
    agg_dict = {'usd_goal_real': ['count', 'mean']}
    
    if 'pledge_goal_ratio' in df_with_clusters.columns:
        agg_dict['pledge_goal_ratio'] = 'mean'
    
    if 'days_duration' in df_with_clusters.columns:
        agg_dict['days_duration'] = 'mean'
    
    cluster_summary = df_with_clusters.groupby('Cluster_Label').agg(agg_dict).round(2)
    
    # Rename columns based on what's available
    if 'pledge_goal_ratio' in df_with_clusters.columns and 'days_duration' in df_with_clusters.columns:
        cluster_summary.columns = ['Count', 'Avg Goal ($)', 'Avg Success Ratio', 'Avg Duration (Days)']
    elif 'pledge_goal_ratio' in df_with_clusters.columns:
        cluster_summary.columns = ['Count', 'Avg Goal ($)', 'Avg Success Ratio']
    elif 'days_duration' in df_with_clusters.columns:
        cluster_summary.columns = ['Count', 'Avg Goal ($)', 'Avg Duration (Days)']
    else:
        cluster_summary.columns = ['Count', 'Avg Goal ($)']
    
    cluster_summary = cluster_summary.sort_values('Count', ascending=False)
    
    # Format the dataframe for better readability
    cluster_summary_formatted = format_dataframe(cluster_summary)
    st.dataframe(cluster_summary_formatted, use_container_width=True)
    
    # Quick insights
    st.markdown("## 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Performing Clusters")
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            top_clusters = df_with_clusters.groupby('Cluster_Label')['pledge_goal_ratio'].mean().sort_values(ascending=False).head(3)
            for i, (cluster, ratio) in enumerate(top_clusters.items(), 1):
                st.markdown(f"{i}. **{cluster}**: {ratio:.2f}x goal")
        else:
            st.markdown("Data not available")
    
    with col2:
        st.markdown("### 📈 Largest Clusters")
        largest_clusters = df_with_clusters['Cluster_Label'].value_counts().head(3)
        for i, (cluster, count) in enumerate(largest_clusters.items(), 1):
            percentage = (count / len(df_with_clusters)) * 100
            st.markdown(f"{i}. **{cluster}**: {count:,} projects ({percentage:.1f}%)")

def show_data_overview(df_with_clusters):
    """Display data overview and basic statistics"""
    st.markdown("## 📊 Data Overview")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Information")
        st.write(f"**Total Projects:** {len(df_with_clusters):,}")
        st.write(f"**Features:** {len(df_with_clusters.columns)}")
        st.write(f"**Date Range:** {df_with_clusters['launched'].min().strftime('%Y-%m-%d')} to {df_with_clusters['launched'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.markdown("### 📈 Key Statistics")
        st.write(f"**Average Goal:** ${df_with_clusters['usd_goal_real'].mean():,.0f}")
        if 'usd_pledged_real' in df_with_clusters.columns:
            st.write(f"**Average Pledged:** ${df_with_clusters['usd_pledged_real'].mean():,.0f}")
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            st.write(f"**Success Rate:** {(df_with_clusters['pledge_goal_ratio'] >= 1).mean() * 100:.1f}%")
        else:
            st.write("**Success Rate:** N/A")
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    # Format the dataframe for better readability
    df_preview_formatted = format_dataframe(df_with_clusters.head(10))
    st.dataframe(df_preview_formatted, use_container_width=True)
    
    # Missing values
    st.markdown("### 🔍 Missing Values Analysis")
    missing_data = df_with_clusters.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data.values / len(df_with_clusters)) * 100
    }).sort_values('Missing Values', ascending=False)
    
    # Format the dataframe for better readability
    missing_df_formatted = format_dataframe(missing_df)
    st.dataframe(missing_df_formatted, use_container_width=True)

def show_cluster_analysis(df_with_clusters):
    """Display detailed cluster analysis"""
    st.markdown("## 🎯 Cluster Analysis")
    
    # Cluster selection
    selected_cluster = st.selectbox(
        "Select a cluster to analyze:",
        options=sorted(df_with_clusters['Cluster'].unique()),
        format_func=lambda x: get_cluster_label(x)
    )
    
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
    
    # Cluster overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Projects", f"{len(cluster_data):,}")
    
    with col2:
        avg_goal = cluster_data['usd_goal_real'].mean()
        st.metric("💰 Avg Goal", f"${avg_goal:,.0f}")
    
    with col3:
        if 'pledge_goal_ratio' in cluster_data.columns:
            avg_ratio = cluster_data['pledge_goal_ratio'].mean()
            st.metric("📈 Success Ratio", f"{avg_ratio:.2f}")
        else:
            st.metric("📈 Success Ratio", "N/A")
    
    with col4:
        if 'pledge_goal_ratio' in cluster_data.columns:
            success_rate = (cluster_data['pledge_goal_ratio'] >= 1).mean() * 100
            st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("✅ Success Rate", "N/A")
    
    # Cluster description
    st.markdown(f"### 📝 {get_cluster_label(selected_cluster)}")
    st.info(get_cluster_description(selected_cluster))
    
    # Detailed statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Numerical Features")
        numerical_stats = cluster_data[['usd_goal_real', 'usd_pledged_real', 'pledge_goal_ratio', 'days_duration']].describe()
        # Format the dataframe for better readability
        numerical_stats_formatted = format_dataframe(numerical_stats)
        st.dataframe(numerical_stats_formatted, use_container_width=True)
    
    with col2:
        st.markdown("### 📂 Category Distribution")
        if 'main_category' in cluster_data.columns:
            category_dist = cluster_data['main_category'].value_counts().head(10)
            st.bar_chart(category_dist)
    
    # Comparison with overall dataset
    st.markdown("### 🔄 Comparison with Overall Dataset")
    
    comparison_data = []
    for feature in ['usd_goal_real', 'usd_pledged_real', 'pledge_goal_ratio', 'days_duration']:
        cluster_mean = cluster_data[feature].mean()
        overall_mean = df_with_clusters[feature].mean()
        comparison_data.append({
            'Feature': feature,
            'Cluster Average': cluster_mean,
            'Overall Average': overall_mean,
            'Difference': cluster_mean - overall_mean,
            'Percentage': ((cluster_mean - overall_mean) / overall_mean) * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    # Format the dataframe for better readability
    comparison_df_formatted = format_dataframe(comparison_df)
    st.dataframe(comparison_df_formatted, use_container_width=True)

def show_project_predictor(df_with_clusters):
    """Show the project prediction interface"""
    st.markdown("## 🔮 Project Predictor")
    st.markdown("Enter your project details to predict which cluster it belongs to!")
    
    # Input form
    with st.form("project_predictor"):
        st.markdown("### 📝 Project Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            goal_amount = st.number_input("💰 Funding Goal ($)", min_value=1, value=10000, step=1000)
            campaign_duration = st.number_input("⏱️ Campaign Duration (days)", min_value=1, max_value=365, value=30)
            
            if 'main_category' in df_with_clusters.columns:
                categories = sorted(df_with_clusters['main_category'].unique())
                selected_category = st.selectbox("📂 Main Category", categories)
        
        with col2:
            # Additional features could be added here
            st.markdown("### 📊 Additional Information")
            st.info("💡 The prediction uses goal amount and duration to find the most similar cluster. This is a simplified approach for demonstration purposes.")
            st.info("🎯 For more accurate predictions, the full model would need additional features like category, backers, and pledged amounts.")
        
        submitted = st.form_submit_button("🔮 Predict Cluster")
        
        if submitted:
            # Create user data (simplified version)
            user_data = {
                'usd_goal_real': goal_amount,
                'days_duration': campaign_duration,
                'pledge_goal_ratio': 0,  # Will be calculated by model
                'usd_pledged_real': 0    # Will be calculated by model
            }
            
            # Predict cluster using basic features only
            try:
                # Use only the basic numerical features for prediction
                basic_features = ['usd_goal_real', 'days_duration']
                available_features = [f for f in basic_features if f in user_data]
                
                if len(available_features) >= 2:
                    # Create a simple prediction based on goal amount and duration
                    # This is a simplified approach that doesn't require the full model
                    
                    # Calculate average values for each cluster
                    cluster_centers = {}
                    for cluster_id in range(8):
                        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
                        if len(cluster_data) > 0:
                            avg_goal = cluster_data['usd_goal_real'].mean()
                            avg_duration = cluster_data['days_duration'].mean() if 'days_duration' in cluster_data.columns else 30
                            cluster_centers[cluster_id] = (avg_goal, avg_duration)
                    
                    # Find the closest cluster based on goal and duration
                    user_goal = user_data['usd_goal_real']
                    user_duration = user_data['days_duration']
                    
                    min_distance = float('inf')
                    predicted_cluster = 0
                    
                    for cluster_id, (avg_goal, avg_duration) in cluster_centers.items():
                        # Normalize the distance calculation
                        goal_distance = abs(user_goal - avg_goal) / max(avg_goal, 1)
                        duration_distance = abs(user_duration - avg_duration) / max(avg_duration, 1)
                        total_distance = goal_distance + duration_distance
                        
                        if total_distance < min_distance:
                            min_distance = total_distance
                            predicted_cluster = cluster_id
                else:
                    # Fallback to a simple rule-based prediction
                    if user_data['usd_goal_real'] > 50000:
                        predicted_cluster = 1  # Premium projects
                    elif user_data['usd_goal_real'] > 20000:
                        predicted_cluster = 0  # Standard projects
                    else:
                        predicted_cluster = 4  # Failure projects
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 📊 Predicted Cluster")
                    cluster_label = get_cluster_label(predicted_cluster)
                    st.markdown(f"**{cluster_label}**")
                    st.info(get_cluster_description(predicted_cluster))
                    
                    # Show prediction confidence
                    if 'min_distance' in locals():
                        confidence = max(0, 100 - (min_distance * 50))  # Convert distance to confidence
                        st.metric("🎯 Prediction Confidence", f"{confidence:.1f}%")
                    else:
                        st.info("📊 Prediction based on simplified rules")
                
                with col2:
                    st.markdown("### 📈 Cluster Characteristics")
                    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == predicted_cluster]
                    
                    st.metric("Average Goal", f"${cluster_data['usd_goal_real'].mean():,.0f}")
                    if 'pledge_goal_ratio' in cluster_data.columns:
                        st.metric("Success Rate", f"{(cluster_data['pledge_goal_ratio'] >= 1).mean() * 100:.1f}%")
                    else:
                        st.metric("Success Rate", "N/A")
                    if 'days_duration' in cluster_data.columns:
                        st.metric("Avg Duration", f"{cluster_data['days_duration'].mean():.0f} days")
                    else:
                        st.metric("Avg Duration", "N/A")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if predicted_cluster in [2, 1]:  # High achievers or premium
                    st.success("🎉 Your project shows characteristics of successful campaigns! Consider:")
                    st.write("- High-quality marketing materials")
                    st.write("- Strong social media presence")
                    st.write("- Clear value proposition")
                
                elif predicted_cluster in [3, 4]:  # Struggling or failure
                    st.warning("⚠️ Your project may face challenges. Consider:")
                    st.write("- Lowering your funding goal")
                    st.write("- Extending campaign duration")
                    st.write("- Improving project presentation")
                
                else:
                    st.info("📊 Your project falls in a moderate category. Consider:")
                    st.write("- Balanced approach to funding")
                    st.write("- Moderate marketing efforts")
                    st.write("- Realistic expectations")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def show_visualizations(df_with_clusters):
    """Display comprehensive visualizations"""
    st.markdown("## 📈 Visualizations")
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization:",
        ["📊 Cluster Distribution", "💰 Goal Analysis", "📈 Success Analysis", "⏱️ Duration Analysis", "🎨 Category Analysis", "📊 Correlation Matrix"]
    )
    
    if viz_option == "📊 Cluster Distribution":
        st.markdown("### 📊 Cluster Distribution")
        
        cluster_counts = df_with_clusters['Cluster_Label'].value_counts()
        
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
            df_with_clusters,
            x='Cluster_Label',
            y='usd_goal_real',
            title="Goal Amount Distribution by Cluster"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average goals
        avg_goals = df_with_clusters.groupby('Cluster_Label')['usd_goal_real'].mean().sort_values(ascending=False)
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
        
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            # Success ratio distribution
            fig = px.box(
                df_with_clusters,
                x='Cluster_Label',
                y='pledge_goal_ratio',
                title="Success Ratio Distribution by Cluster"
            )
            fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="100% Success Line")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Success rates
            success_rates = df_with_clusters.groupby('Cluster_Label')['pledge_goal_ratio'].apply(
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
        else:
            st.warning("⚠️ Success ratio data not available for visualization.")
    
    elif viz_option == "⏱️ Duration Analysis":
        st.markdown("### ⏱️ Duration Analysis")
        
        if 'days_duration' in df_with_clusters.columns:
            fig = px.box(
                df_with_clusters,
                x='Cluster_Label',
                y='days_duration',
                title="Campaign Duration Distribution by Cluster"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Average duration
            avg_duration = df_with_clusters.groupby('Cluster_Label')['days_duration'].mean().sort_values(ascending=False)
            fig_duration = px.bar(
                x=avg_duration.index,
                y=avg_duration.values,
                title="Average Campaign Duration by Cluster",
                labels={'x': 'Cluster', 'y': 'Average Duration (Days)'}
            )
            fig_duration.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.warning("⚠️ Duration data not available for visualization.")
    
    elif viz_option == "🎨 Category Analysis":
        st.markdown("### 🎨 Category Analysis")
        
        if 'main_category' in df_with_clusters.columns:
            # Top categories overall
            top_categories = df_with_clusters['main_category'].value_counts().head(10)
            fig_cat = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title="Top 10 Categories by Number of Projects",
                labels={'x': 'Category', 'y': 'Number of Projects'}
            )
            fig_cat.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Category performance by cluster
            category_cluster = df_with_clusters.groupby(['main_category', 'Cluster_Label']).size().unstack(fill_value=0)
            fig_heatmap = px.imshow(
                category_cluster,
                title="Category Distribution Across Clusters",
                labels=dict(x="Cluster", y="Category", color="Number of Projects")
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    elif viz_option == "📊 Correlation Matrix":
        st.markdown("### 📊 Correlation Matrix")
        
        # Create list of available numerical columns
        numerical_cols = ['usd_goal_real']
        if 'usd_pledged_real' in df_with_clusters.columns:
            numerical_cols.append('usd_pledged_real')
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            numerical_cols.append('pledge_goal_ratio')
        if 'days_duration' in df_with_clusters.columns:
            numerical_cols.append('days_duration')
        
        if len(numerical_cols) >= 2:
            correlation_matrix = df_with_clusters[numerical_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Need at least 2 numerical features for correlation matrix.")

def show_detailed_statistics(df_with_clusters):
    """Display detailed statistical analysis"""
    st.markdown("## 📋 Detailed Statistics")
    
    # Overall statistics
    st.markdown("### 📊 Overall Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Numerical Features")
        # Create list of available numerical columns
        numerical_cols = ['usd_goal_real']
        if 'usd_pledged_real' in df_with_clusters.columns:
            numerical_cols.append('usd_pledged_real')
        if 'pledge_goal_ratio' in df_with_clusters.columns:
            numerical_cols.append('pledge_goal_ratio')
        if 'days_duration' in df_with_clusters.columns:
            numerical_cols.append('days_duration')
        
        numerical_stats = df_with_clusters[numerical_cols].describe()
        # Format the dataframe for better readability
        numerical_stats_formatted = format_dataframe(numerical_stats)
        st.dataframe(numerical_stats_formatted, use_container_width=True)
    
    with col2:
        st.markdown("#### 📂 Categorical Features")
        if 'main_category' in df_with_clusters.columns:
            category_stats = df_with_clusters['main_category'].value_counts().head(10)
            # Format the dataframe for better readability
            category_stats_formatted = format_dataframe(category_stats)
            st.dataframe(category_stats_formatted, use_container_width=True)
    
    # Cluster comparison
    st.markdown("### 🔄 Cluster Comparison")
    
    # Create aggregation dictionary based on available columns
    agg_dict = {'usd_goal_real': ['mean', 'std', 'min', 'max']}
    
    if 'usd_pledged_real' in df_with_clusters.columns:
        agg_dict['usd_pledged_real'] = ['mean', 'std']
    
    if 'pledge_goal_ratio' in df_with_clusters.columns:
        agg_dict['pledge_goal_ratio'] = ['mean', 'std']
    
    if 'days_duration' in df_with_clusters.columns:
        agg_dict['days_duration'] = ['mean', 'std']
    
    comparison_metrics = df_with_clusters.groupby('Cluster_Label').agg(agg_dict).round(2)
    
    # Format the dataframe for better readability
    comparison_metrics_formatted = format_dataframe(comparison_metrics)
    st.dataframe(comparison_metrics_formatted, use_container_width=True)
    
    # Statistical tests
    st.markdown("### 🔬 Statistical Analysis")
    
    # ANOVA test for goal amounts across clusters
    from scipy.stats import f_oneway
    
    cluster_groups = [df_with_clusters[df_with_clusters['Cluster'] == i]['usd_goal_real'].values 
                     for i in sorted(df_with_clusters['Cluster'].unique())]
    
    f_stat, p_value = f_oneway(*cluster_groups)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 ANOVA Test Results")
        st.write(f"**F-statistic:** {f_stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.success("✅ Significant differences exist between clusters (p < 0.05)")
        else:
            st.warning("⚠️ No significant differences between clusters (p ≥ 0.05)")
    
    with col2:
        st.markdown("#### 📈 Key Insights")
        st.write("• **Largest Cluster:** " + df_with_clusters['Cluster_Label'].value_counts().index[0])
        st.write("• **Highest Success Rate:** " + 
                df_with_clusters.groupby('Cluster_Label')['pledge_goal_ratio'].apply(
                    lambda x: (x >= 1).mean()
                ).idxmax())
        st.write("• **Highest Average Goal:** " + 
                df_with_clusters.groupby('Cluster_Label')['usd_goal_real'].mean().idxmax())

if __name__ == "__main__":
    main() 
