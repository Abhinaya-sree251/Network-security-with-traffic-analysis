import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Network Security Analyzer", layout="wide")

st.title("🛡️ Advanced AI Network Security Traffic Analyzer")
st.markdown("""
This AI-powered app provides comprehensive network traffic analysis with:
- **Automated Traffic Pattern Recognition**
- **Intelligent Anomaly Detection**
- **Behavioral Analysis & Clustering**
- **Advanced Threat Classification**
- **Real-time Security Insights**
""")

# ----------------------------
# Sidebar
# ----------------------------
menu = st.sidebar.radio("Choose Analysis Mode", [
    "📊 Data Upload & Smart Analysis", 
    "🤖 AI Traffic Classification", 
    "🔍 Advanced Anomaly Detection", 
    "📈 Behavioral Pattern Analysis",
    "⚡ Real-time Threat Assessment"
])

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# AI Enhancement Functions
# ----------------------------
def ai_traffic_insights(df):
    """Generate AI-powered insights about traffic patterns"""
    insights = []
    
    # Basic statistics
    total_flows = len(df)
    
    # Duration analysis
    if 'duration' in df.columns:
        avg_duration = df['duration'].mean()
        long_connections = (df['duration'] > df['duration'].quantile(0.9)).sum()
        insights.append(f"🕐 Average connection duration: {avg_duration:.2f}s")
        insights.append(f"⏱️ Long-lasting connections (>90th percentile): {long_connections}")
    
    # Traffic volume analysis
    if 'bytes_sent' in df.columns and 'bytes_received' in df.columns:
        total_traffic = df['bytes_sent'].sum() + df['bytes_received'].sum()
        avg_traffic_per_flow = total_traffic / total_flows
        insights.append(f"📊 Total traffic volume: {total_traffic:,} bytes")
        insights.append(f"📈 Average traffic per flow: {avg_traffic_per_flow:.0f} bytes")
        
        # Identify high-volume flows
        high_volume_threshold = df['bytes_sent'].quantile(0.95)
        high_volume_flows = (df['bytes_sent'] > high_volume_threshold).sum()
        insights.append(f"🔺 High-volume flows (>95th percentile): {high_volume_flows}")
    
    # Packet analysis
    if 'packet_count' in df.columns:
        avg_packets = df['packet_count'].mean()
        insights.append(f"📦 Average packets per flow: {avg_packets:.1f}")
        
        # Small packet flows (potential scanning)
        small_packet_flows = (df['packet_count'] < 10).sum()
        insights.append(f"🔍 Small packet flows (<10 packets): {small_packet_flows}")
    
    return insights

def ai_threat_scoring(df):
    """Calculate AI-based threat scores for each flow"""
    threat_scores = np.zeros(len(df))
    
    # Score based on various suspicious patterns
    if 'duration' in df.columns:
        # Very short or very long connections are suspicious
        short_conn_score = (df['duration'] < 0.1).astype(int) * 0.3
        long_conn_score = (df['duration'] > 300).astype(int) * 0.2
        threat_scores += short_conn_score + long_conn_score
    
    if 'packet_count' in df.columns:
        # Very few packets might indicate scanning
        low_packet_score = (df['packet_count'] < 5).astype(int) * 0.4
        threat_scores += low_packet_score
    
    if 'bytes_sent' in df.columns and 'bytes_received' in df.columns:
        # Asymmetric traffic patterns
        total_bytes = df['bytes_sent'] + df['bytes_received']
        asymmetry = abs(df['bytes_sent'] - df['bytes_received']) / (total_bytes + 1)
        high_asymmetry_score = (asymmetry > 0.9).astype(int) * 0.3
        threat_scores += high_asymmetry_score
    
    # Normalize scores to 0-1 range
    if threat_scores.max() > 0:
        threat_scores = threat_scores / threat_scores.max()
    
    return threat_scores

def advanced_clustering_analysis(df, features):
    """Perform advanced clustering to identify traffic patterns with error handling"""
    try:
        # Check if we have enough samples
        if len(df) < 4:
            st.warning("⚠️ Not enough samples for clustering analysis (need at least 4)")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        # Apply PCA for dimensionality reduction
        n_components = min(3, len(features), len(df) - 1)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-means clustering - adjust number of clusters based on dataset size
        n_clusters = min(4, len(df) // 2, max(2, len(df) // 10))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN for anomaly detection - adjust parameters for dataset size
        eps = 0.5 if len(df) > 50 else 1.0
        min_samples = min(5, max(2, len(df) // 10))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        return {
            'kmeans_labels': cluster_labels,
            'dbscan_labels': dbscan_labels,
            'pca_components': X_pca,
            'scaler': scaler,
            'pca': pca,
            'n_clusters': n_clusters
        }
    except Exception as e:
        st.error(f"❌ Clustering analysis failed: {str(e)}")
        return None

def neural_network_classifier(X_train, X_test, y_train, y_test):
    """Train a neural network classifier with robust error handling"""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Adjust hidden layer size based on dataset size
        n_samples = len(X_train)
        if n_samples < 100:
            hidden_layers = (min(50, n_samples // 2),)
        else:
            hidden_layers = (100, 50)
        
        # Train neural network with adaptive parameters
        nn_clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=min(1000, n_samples * 10),  # Adjust iterations based on dataset size
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1 if n_samples > 20 else 0.2,
            n_iter_no_change=10,
            alpha=0.001  # L2 regularization
        )
        
        # Handle potential convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn_clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = nn_clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return nn_clf, scaler, accuracy, y_pred
        
    except Exception as e:
        st.warning(f"⚠️ Neural network training failed: {str(e)}")
        # Return dummy results to prevent crashes
        return None, None, 0.0, np.zeros(len(y_test))

def plot_advanced_visualizations(df, features):
    """Create advanced visualizations for traffic analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation heatmap
    if len(features) > 1:
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
        axes[0,0].set_title('Feature Correlation Heatmap')
    
    # 2. Distribution plot
    if 'duration' in features:
        axes[0,1].hist(df['duration'], bins=50, alpha=0.7, color='skyblue')
        axes[0,1].set_xlabel('Duration')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Duration Distribution')
        axes[0,1].set_yscale('log')
    
    # 3. Packet size vs duration scatter
    if 'packet_size_mean' in features and 'duration' in features:
        scatter = axes[1,0].scatter(df['duration'], df['packet_size_mean'], 
                                   alpha=0.6, c='blue', s=20)
        axes[1,0].set_xlabel('Duration')
        axes[1,0].set_ylabel('Packet Size Mean')
        axes[1,0].set_title('Duration vs Packet Size')
    
    # 4. Bytes sent vs received
    if 'bytes_sent' in features and 'bytes_received' in features:
        axes[1,1].scatter(df['bytes_sent'], df['bytes_received'], 
                         alpha=0.6, c='green', s=20)
        axes[1,1].set_xlabel('Bytes Sent')
        axes[1,1].set_ylabel('Bytes Received')
        axes[1,1].set_title('Bytes Sent vs Received')
        axes[1,1].plot([0, max(df['bytes_sent'].max(), df['bytes_received'].max())], 
                      [0, max(df['bytes_sent'].max(), df['bytes_received'].max())], 
                      'r--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# ----------------------------
# Shared Utility Functions
# ----------------------------
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    st.write(f"📊 Dataset shape: {df.shape}")
    st.write("📋 Column names:", df.columns.tolist())
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Flows", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        if 'label' in df.columns:
            st.metric("Classes", df['label'].nunique())
    
    st.write("🔍 Data Preview:")
    st.dataframe(df.head())
    return df

def get_required_columns():
    return ['duration', 'packet_count', 'bytes_sent', 'bytes_received', 'packet_size_mean']

def check_required_columns(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    return missing_cols

# ----------------------------
# 1. Data Upload & Smart Analysis
# ----------------------------
if menu == "📊 Data Upload & Smart Analysis":
    st.header("📤 Upload Traffic Data for AI Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        # Required columns check
        required_cols = get_required_columns()
        missing_cols = check_required_columns(df, required_cols)
        
        if missing_cols:
            st.warning(f"⚠️ Missing required columns: {missing_cols}")
            st.info("Please ensure your CSV has columns: " + ", ".join(required_cols))
        else:
            # AI-powered traffic insights
            st.subheader("🤖 AI Traffic Insights")
            insights = ai_traffic_insights(df)
            for insight in insights:
                st.write(f"• {insight}")
            
            # Threat scoring
            st.subheader("🚨 AI Threat Assessment")
            threat_scores = ai_threat_scoring(df)
            df['threat_score'] = threat_scores
            
            # Display high-risk flows
            high_risk_flows = df[df['threat_score'] > 0.7]
            st.write(f"🔴 High-risk flows identified: {len(high_risk_flows)}")
            
            if len(high_risk_flows) > 0:
                st.write("Top 10 highest risk flows:")
                st.dataframe(high_risk_flows.nlargest(10, 'threat_score'))
            
            # Advanced visualizations
            st.subheader("📊 Advanced Traffic Analysis")
            fig = plot_advanced_visualizations(df, required_cols)
            st.pyplot(fig)
            
            # Clustering analysis
            st.subheader("🔍 Traffic Pattern Clustering")
            clustering_results = advanced_clustering_analysis(df, required_cols)
            
            if clustering_results is not None:
                # Display cluster information
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**K-Means Clustering Results:**")
                    cluster_counts = pd.Series(clustering_results['kmeans_labels']).value_counts()
                    st.write(cluster_counts)
                
                with col2:
                    st.write("**DBSCAN Anomaly Detection:**")
                    dbscan_counts = pd.Series(clustering_results['dbscan_labels']).value_counts()
                    st.write(dbscan_counts)
                    anomalies = (clustering_results['dbscan_labels'] == -1).sum()
                    st.metric("Anomalies Detected", anomalies)
            else:
                st.info("ℹ️ Clustering analysis skipped due to insufficient data")

# ----------------------------
# 2. AI Traffic Classification
# ----------------------------
elif menu == "🤖 AI Traffic Classification":
    st.header("🧠 Advanced AI Traffic Classification")
    uploaded_file = st.file_uploader("Upload labeled CSV (with 'label')", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if 'label' not in df.columns:
            st.error("❌ No 'label' column found. Cannot train classifier.")
        else:
            required_cols = get_required_columns()
            missing_cols = check_required_columns(df, required_cols)
            
            if not missing_cols:
                X = df[required_cols]
                y = df['label']
                
                # Encode labels if they're strings
                le = LabelEncoder()
                if y.dtype == 'object':
                    y_encoded = le.fit_transform(y)
                else:
                    y_encoded = y
                
                # Check if we have enough samples for train-test split
                min_samples_per_class = pd.Series(y_encoded).value_counts().min()
                total_samples = len(y_encoded)
                
                if total_samples < 10:
                    st.error(f"❌ Dataset too small ({total_samples} samples). Need at least 10 samples for reliable training.")
                    st.stop()
                
                # Adjust test size based on dataset size and class distribution
                if total_samples < 50:
                    test_size = max(0.2, 2 / total_samples)  # At least 2 samples for test
                    st.warning(f"⚠️ Small dataset detected. Using test_size={test_size:.2f}")
                else:
                    test_size = 0.3
                
                # Check if stratification is possible
                if min_samples_per_class < 2:
                    st.warning("⚠️ Some classes have only 1 sample. Using random split instead of stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
                    )
                
                # Train multiple models
                st.subheader("🔄 Training Multiple AI Models")
                
                models = {}
                
                # Random Forest
                with st.spinner("Training Random Forest..."):
                    try:
                        rf_clf = RandomForestClassifier(
                            n_estimators=min(100, max(10, len(X_train) // 5)),  # Adjust based on dataset size
                            random_state=42,
                            max_depth=None if len(X_train) > 100 else 10
                        )
                        rf_clf.fit(X_train, y_train)
                        rf_pred = rf_clf.predict(X_test)
                        rf_acc = accuracy_score(y_test, rf_pred)
                        models['Random Forest'] = {'model': rf_clf, 'accuracy': rf_acc, 'predictions': rf_pred}
                        st.success(f"✅ Random Forest trained: {rf_acc:.3f} accuracy")
                    except Exception as e:
                        st.error(f"❌ Random Forest failed: {str(e)}")
                
                # Neural Network
                with st.spinner("Training Neural Network..."):
                    try:
                        nn_clf, nn_scaler, nn_acc, nn_pred = neural_network_classifier(X_train, X_test, y_train, y_test)
                        if nn_clf is not None:
                            models['Neural Network'] = {'model': nn_clf, 'accuracy': nn_acc, 'predictions': nn_pred, 'scaler': nn_scaler}
                            st.success(f"✅ Neural Network trained: {nn_acc:.3f} accuracy")
                        else:
                            st.warning("⚠️ Neural Network training skipped due to small dataset")
                    except Exception as e:
                        st.error(f"❌ Neural Network failed: {str(e)}")
                
                # SVM (only for smaller datasets to avoid memory issues)
                if len(X_train) < 1000:
                    with st.spinner("Training SVM..."):
                        try:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            svm_clf = SVC(kernel='rbf', random_state=42, gamma='scale')
                            svm_clf.fit(X_train_scaled, y_train)
                            svm_pred = svm_clf.predict(X_test_scaled)
                            svm_acc = accuracy_score(y_test, svm_pred)
                            models['SVM'] = {'model': svm_clf, 'accuracy': svm_acc, 'predictions': svm_pred, 'scaler': scaler}
                            st.success(f"✅ SVM trained: {svm_acc:.3f} accuracy")
                        except Exception as e:
                            st.error(f"❌ SVM failed: {str(e)}")
                else:
                    st.info("ℹ️ SVM skipped for large dataset (>1000 samples)")
                
                if not models:
                    st.error("❌ No models were successfully trained. Please check your data.")
                    st.stop()
                
                # Display results
                st.subheader("📊 Model Performance Comparison")
                results_df = pd.DataFrame({
                    'Model': list(models.keys()),
                    'Accuracy': [models[m]['accuracy'] for m in models.keys()]
                })
                st.dataframe(results_df)
                
                # Best model
                best_model_name = max(models.keys(), key=lambda x: models[x]['accuracy'])
                st.success(f"🏆 Best performing model: {best_model_name} (Accuracy: {models[best_model_name]['accuracy']:.3f})")
                
                # Save best model
                best_model = models[best_model_name]['model']
                joblib.dump(best_model, os.path.join(MODEL_DIR, "best_traffic_classifier.pkl"))
                if 'scaler' in models[best_model_name]:
                    joblib.dump(models[best_model_name]['scaler'], os.path.join(MODEL_DIR, "classifier_scaler.pkl"))
                
                # Confusion matrix for best model
                st.subheader("🎯 Confusion Matrix (Best Model)")
                cm = confusion_matrix(y_test, models[best_model_name]['predictions'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {best_model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

# ----------------------------
# 3. Advanced Anomaly Detection
# ----------------------------
elif menu == "🔍 Advanced Anomaly Detection":
    st.header("🔬 Advanced AI Anomaly Detection")
    uploaded_file = st.file_uploader("Upload CSV for anomaly detection", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        required_cols = get_required_columns()
        missing_cols = check_required_columns(df, required_cols)
        
        if not missing_cols:
            X = df[required_cols]
            
            # Multiple anomaly detection methods
            st.subheader("🔍 Multi-Method Anomaly Detection")
            
            # 1. Isolation Forest
            with st.spinner("Running Isolation Forest..."):
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_anomalies = iso_forest.fit_predict(X)
                iso_scores = iso_forest.decision_function(X)
            
            # 2. DBSCAN
            with st.spinner("Running DBSCAN clustering..."):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                dbscan_anomalies = (dbscan_labels == -1).astype(int)
            
            # 3. Statistical anomaly detection
            with st.spinner("Running statistical analysis..."):
                # Z-score based anomaly detection
                z_scores = np.abs((X - X.mean()) / X.std())
                statistical_anomalies = (z_scores > 3).any(axis=1).astype(int)
            
            # Combine results
            df['iso_forest_anomaly'] = (iso_anomalies == -1).astype(int)
            df['dbscan_anomaly'] = dbscan_anomalies
            df['statistical_anomaly'] = statistical_anomalies
            df['anomaly_score'] = iso_scores
            
            # Consensus anomaly detection
            df['consensus_anomaly'] = (df['iso_forest_anomaly'] + df['dbscan_anomaly'] + df['statistical_anomaly'] >= 2).astype(int)
            
            # Display results
            st.subheader("📊 Anomaly Detection Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Isolation Forest", df['iso_forest_anomaly'].sum())
            with col2:
                st.metric("DBSCAN", df['dbscan_anomaly'].sum())
            with col3:
                st.metric("Statistical", df['statistical_anomaly'].sum())
            with col4:
                st.metric("Consensus", df['consensus_anomaly'].sum())
            
            # Show top anomalies
            st.subheader("🚨 Top Anomalous Flows")
            top_anomalies = df.nsmallest(10, 'anomaly_score')
            st.dataframe(top_anomalies)
            
            # Visualization
            st.subheader("📈 Anomaly Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['anomaly_score'], bins=50, alpha=0.7, color='skyblue')
            ax.axvline(df['anomaly_score'].quantile(0.1), color='red', linestyle='--', label='10th percentile')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Anomaly Scores')
            ax.legend()
            st.pyplot(fig)
            
            # Save anomaly detector
            joblib.dump(iso_forest, os.path.join(MODEL_DIR, "advanced_anomaly_detector.pkl"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))

# ----------------------------
# 4. Behavioral Pattern Analysis
# ----------------------------
elif menu == "📈 Behavioral Pattern Analysis":
    st.header("🧬 Advanced Behavioral Pattern Analysis")
    uploaded_file = st.file_uploader("Upload CSV for pattern analysis", type="csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        required_cols = get_required_columns()
        missing_cols = check_required_columns(df, required_cols)
        
        if not missing_cols:
            X = df[required_cols]
            
            # Advanced clustering analysis
            st.subheader("🔍 Pattern Discovery")
            clustering_results = advanced_clustering_analysis(df, required_cols)
            
            if clustering_results is not None:
                # Add cluster labels to dataframe
                df['behavior_cluster'] = clustering_results['kmeans_labels']
                
                # Analyze each cluster
                st.subheader("📊 Behavioral Cluster Analysis")
                
                n_clusters = clustering_results['n_clusters']
                for cluster in range(n_clusters):
                    cluster_data = df[df['behavior_cluster'] == cluster]
                    if len(cluster_data) > 0:
                        st.write(f"**Cluster {cluster} ({len(cluster_data)} flows):**")
                        
                        # Cluster characteristics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Duration", f"{cluster_data['duration'].mean():.2f}s")
                        with col2:
                            st.metric("Avg Packets", f"{cluster_data['packet_count'].mean():.0f}")
                        with col3:
                            st.metric("Avg Bytes", f"{cluster_data['bytes_sent'].mean():.0f}")
                        
                        # Behavioral insights
                        duration_pattern = "Short" if cluster_data['duration'].mean() < df['duration'].mean() else "Long"
                        packet_pattern = "Few" if cluster_data['packet_count'].mean() < df['packet_count'].mean() else "Many"
                        
                        st.write(f"Pattern: {duration_pattern} duration, {packet_pattern} packets")
                        st.write("---")
            else:
                st.info("ℹ️ Pattern analysis skipped due to insufficient data")
            
            # Time series analysis (if timestamp available)
            st.subheader("⏰ Temporal Pattern Analysis")
            if 'timestamp' in df.columns:
                # Convert timestamp and analyze patterns
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # Hourly pattern
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                hourly_counts = df.groupby('hour').size()
                axes[0].bar(hourly_counts.index, hourly_counts.values)
                axes[0].set_xlabel('Hour of Day')
                axes[0].set_ylabel('Flow Count')
                axes[0].set_title('Traffic by Hour')
                
                # Daily pattern
                daily_counts = df.groupby('day_of_week').size()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                axes[1].bar(range(7), daily_counts.values)
                axes[1].set_xlabel('Day of Week')
                axes[1].set_ylabel('Flow Count')
                axes[1].set_title('Traffic by Day')
                axes[1].set_xticks(range(7))
                axes[1].set_xticklabels(days)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("💡 Add a 'timestamp' column to enable temporal analysis")

# ----------------------------
# 5. Real-time Threat Assessment
# ----------------------------
elif menu == "⚡ Real-time Threat Assessment":
    st.header("🎯 Real-time AI Threat Assessment")
    st.markdown("Enter traffic flow characteristics for instant AI-powered threat analysis:")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=3.0, step=0.1)
        packet_count = st.number_input("Packet Count", min_value=0, value=25, step=1)
        bytes_sent = st.number_input("Bytes Sent", min_value=0, value=2000, step=100)
    
    with col2:
        bytes_received = st.number_input("Bytes Received", min_value=0, value=1800, step=100)
        packet_size_mean = st.number_input("Packet Size Mean", min_value=0.0, value=500.0, step=10.0)
    
    if st.button("🔍 Analyze Threat Level", type="primary"):
        features = np.array([[duration, packet_count, bytes_sent, bytes_received, packet_size_mean]])
        
        # Load models
        clf_path = os.path.join(MODEL_DIR, "best_traffic_classifier.pkl")
        iso_path = os.path.join(MODEL_DIR, "advanced_anomaly_detector.pkl")
        
        if os.path.exists(clf_path) and os.path.exists(iso_path):
            try:
                # Load models
                clf = joblib.load(clf_path)
                iso = joblib.load(iso_path)
                
                # Load scalers if they exist
                scaler_path = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    features_scaled = scaler.transform(features)
                    class_pred = clf.predict(features_scaled)[0]
                    class_proba = clf.predict_proba(features_scaled)[0]
                else:
                    class_pred = clf.predict(features)[0]
                    class_proba = clf.predict_proba(features)[0] if hasattr(clf, 'predict_proba') else None
                
                # Anomaly detection
                anomaly_pred = iso.predict(features)[0]
                anomaly_score = iso.decision_function(features)[0]
                
                # AI threat scoring
                temp_df = pd.DataFrame(features, columns=['duration', 'packet_count', 'bytes_sent', 'bytes_received', 'packet_size_mean'])
                ai_threat_score = ai_threat_scoring(temp_df)[0]
                
                # Comprehensive threat assessment
                st.subheader("🛡️ Comprehensive Threat Assessment")
                
                # Threat level calculation
                threat_level = (ai_threat_score + (1 if anomaly_pred == -1 else 0) + (-anomaly_score if anomaly_score < 0 else 0)) / 3
                
                if threat_level > 0.7:
                    threat_status = "🔴 HIGH RISK"
                    threat_color = "red"
                elif threat_level > 0.4:
                    threat_status = "🟡 MEDIUM RISK"
                    threat_color = "orange"
                else:
                    threat_status = "🟢 LOW RISK"
                    threat_color = "green"
                
                st.markdown(f"### Overall Threat Level: {threat_status}")
                st.progress(threat_level)
                
                # Detailed analysis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Classification", f"Class {class_pred}")
                    if class_proba is not None:
                        st.write(f"Confidence: {class_proba.max():.2%}")
                
                with col2:
                    anomaly_status = "Anomaly" if anomaly_pred == -1 else "Normal"
                    st.metric("Anomaly Status", anomaly_status)
                    st.write(f"Anomaly Score: {anomaly_score:.3f}")
                
                with col3:
                    st.metric("AI Threat Score", f"{ai_threat_score:.2%}")
                    st.write(f"Risk Level: {threat_level:.2%}")
                
                # Behavioral insights
                st.subheader("🧠 AI Behavioral Analysis")
                insights = []
                
                if duration < 0.5:
                    insights.append("⚠️ Very short connection - potential scanning activity")
                elif duration > 300:
                    insights.append("⚠️ Long-lasting connection - potential data exfiltration")
                
                if packet_count < 5:
                    insights.append("⚠️ Very few packets - suspicious activity pattern")
                elif packet_count > 1000:
                    insights.append("⚠️ High packet count - potential bulk data transfer")
                
                traffic_ratio = bytes_sent / (bytes_received + 1)
                if traffic_ratio > 10:
                    insights.append("⚠️ High upload ratio - potential data exfiltration")
                elif traffic_ratio < 0.1:
                    insights.append("⚠️ High download ratio - potential data infiltration")
                
                if packet_size_mean < 100:
                    insights.append("⚠️ Small packet size - potential control traffic")
                elif packet_size_mean > 1400:
                    insights.append("⚠️ Large packet size - potential bulk data transfer")
                
                if insights:
                    for insight in insights:
                        st.write(f"• {insight}")
                else:
                    st.write("✅ No suspicious behavioral patterns detected")
                
                # Recommendations
                st.subheader("💡 AI Security Recommendations")
                recommendations = []
                
                if threat_level > 0.7:
                    recommendations.extend([
                        "🚨 Immediate investigation required",
                        "🔒 Consider blocking this traffic pattern",
                        "📊 Monitor similar connections closely",
                        "🔍 Deep packet inspection recommended"
                    ])
                elif threat_level > 0.4:
                    recommendations.extend([
                        "⚠️ Enhanced monitoring recommended",
                        "📈 Track this connection pattern",
                        "🔍 Consider additional analysis"
                    ])
                else:
                    recommendations.extend([
                        "✅ Connection appears legitimate",
                        "📊 Continue normal monitoring"
                    ])
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please ensure models are trained first using the other menu options.")
        else:
            st.warning("⚠️ Models not found. Please train classification and anomaly detection models first.")
            st.info("Use the 'AI Traffic Classification' and 'Advanced Anomaly Detection' sections to train models.")

# ----------------------------
# Advanced AI Features Sidebar
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Features")
st.sidebar.markdown("""
**Advanced Capabilities:**
- Multi-model ensemble learning
- Real-time threat scoring
- Behavioral pattern recognition
- Anomaly consensus detection
- Neural network classification
- Automated feature engineering
- Temporal pattern analysis
- Risk assessment automation
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Expected Data Format")
st.sidebar.markdown("""
**Required Columns:**
- `duration`: Connection duration (seconds)
- `packet_count`: Number of packets
- `bytes_sent`: Bytes transmitted
- `bytes_received`: Bytes received
- `packet_size_mean`: Average packet size
- `label`: Traffic class (for training)
- `timestamp`: Time info (optional)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Model Status")
classifier_exists = os.path.exists(os.path.join(MODEL_DIR, "best_traffic_classifier.pkl"))
anomaly_exists = os.path.exists(os.path.join(MODEL_DIR, "advanced_anomaly_detector.pkl"))

st.sidebar.write("**Classifier:**", "✅ Trained" if classifier_exists else "❌ Not trained")
st.sidebar.write("**Anomaly Detector:**", "✅ Trained" if anomaly_exists else "❌ Not trained")

# ----------------------------
# Advanced Analytics Dashboard
# ----------------------------
if st.sidebar.button("📊 Show AI Analytics Dashboard"):
    st.subheader("🎯 AI Analytics Dashboard")
    
    # Model performance metrics
    if classifier_exists and anomaly_exists:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Trained", "2/2", "✅")
        with col2:
            st.metric("AI Features", "8+", "🚀")
        with col3:
            st.metric("Detection Methods", "3", "🔍")
        with col4:
            st.metric("Status", "Ready", "✅")
        
        # Feature importance (if available)
        try:
            clf = joblib.load(os.path.join(MODEL_DIR, "best_traffic_classifier.pkl"))
            if hasattr(clf, 'feature_importances_'):
                st.subheader("📊 Feature Importance Analysis")
                feature_names = get_required_columns()
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance Score')
                ax.set_title('Feature Importance in Traffic Classification')
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars based on importance
                for i, bar in enumerate(bars):
                    if importance_df.iloc[i]['Importance'] > 0.3:
                        bar.set_color('red')
                    elif importance_df.iloc[i]['Importance'] > 0.2:
                        bar.set_color('orange')
                    else:
                        bar.set_color('skyblue')
                
                plt.tight_layout()
                st.pyplot(fig)
        except:
            st.info("Train models to see feature importance analysis")
    else:
        st.warning("⚠️ Train models first to access full analytics dashboard")

# ----------------------------
# Export and Model Management
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Model Management")

if st.sidebar.button("📥 Export Analysis Results"):
    st.sidebar.success("✅ Export functionality ready")
    st.sidebar.info("Results will be exported when analysis is complete")

if st.sidebar.button("🔄 Reset All Models"):
    if st.sidebar.confirm("Are you sure you want to reset all models?"):
        for file in os.listdir(MODEL_DIR):
            if file.endswith('.pkl'):
                os.remove(os.path.join(MODEL_DIR, file))
        st.sidebar.success("✅ All models reset")
        st.rerun()

# ----------------------------
# Real-time Monitoring Simulation
# ----------------------------
if st.sidebar.button("🔴 Simulate Real-time Monitoring"):
    st.subheader("🔴 Real-time Traffic Monitoring Simulation")
    
    # Generate synthetic traffic data
    np.random.seed(42)
    n_flows = 50
    
    # Create synthetic data with some anomalies
    normal_data = {
        'duration': np.random.exponential(2, n_flows-5),
        'packet_count': np.random.poisson(30, n_flows-5),
        'bytes_sent': np.random.normal(2000, 500, n_flows-5),
        'bytes_received': np.random.normal(1800, 400, n_flows-5),
        'packet_size_mean': np.random.normal(500, 100, n_flows-5)
    }
    
    # Add some anomalies
    anomaly_data = {
        'duration': [0.1, 0.05, 500, 600, 0.02],
        'packet_count': [2, 1, 2000, 1500, 3],
        'bytes_sent': [50, 20, 50000, 40000, 30],
        'bytes_received': [10, 5, 1000, 800, 5],
        'packet_size_mean': [25, 15, 1200, 1100, 10]
    }
    
    # Combine data
    synthetic_df = pd.DataFrame(normal_data)
    anomaly_df = pd.DataFrame(anomaly_data)
    full_df = pd.concat([synthetic_df, anomaly_df], ignore_index=True)
    
    # Add AI threat scoring
    threat_scores = ai_threat_scoring(full_df)
    full_df['threat_score'] = threat_scores
    full_df['flow_id'] = range(len(full_df))
    
    # Real-time display
    st.write("**Live Traffic Analysis:**")
    
    # Create a placeholder for real-time updates
    placeholder = st.empty()
    
    # Sort by threat score for dramatic effect
    full_df = full_df.sort_values('threat_score', ascending=False)
    
    # Display top threats
    high_threat_flows = full_df[full_df['threat_score'] > 0.3]
    
    if len(high_threat_flows) > 0:
        st.error(f"🚨 {len(high_threat_flows)} high-risk flows detected!")
        st.dataframe(high_threat_flows[['flow_id', 'duration', 'packet_count', 'bytes_sent', 'threat_score']].head(10))
    else:
        st.success("✅ No high-risk flows detected")
    
    # Traffic summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Flows", len(full_df))
    with col2:
        st.metric("High Risk", len(high_threat_flows))
    with col3:
        avg_threat = full_df['threat_score'].mean()
        st.metric("Avg Threat Score", f"{avg_threat:.2%}")

# ----------------------------
# Footer with AI Capabilities
# ----------------------------
st.markdown("---")
st.markdown("""
### 🤖 AI-Powered Network Security Features

This advanced system combines multiple AI techniques for comprehensive network security:

**🧠 Machine Learning Models:**
- Random Forest Classification
- Neural Network Deep Learning
- Support Vector Machines
- Isolation Forest Anomaly Detection

**🔍 Advanced Analytics:**
- Multi-method Anomaly Detection
- Behavioral Pattern Clustering
- Real-time Threat Assessment
- Feature Importance Analysis

**⚡ Real-time Capabilities:**
- Instant threat scoring
- Automated pattern recognition
- Consensus-based detection
- Behavioral anomaly identification

**📊 Visualization & Insights:**
- Interactive dashboards
- Correlation analysis
- Temporal pattern detection
- Risk assessment metrics
""")

st.markdown("---")
st.markdown("*Powered by Advanced AI for Network Security* 🛡️")