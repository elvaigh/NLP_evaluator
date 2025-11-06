import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import sqlite3
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Configure page
st.set_page_config(page_title="Prediction Submission System", layout="wide")

# Database setup
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('submissions.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create submissions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            description TEXT,
            accuracy REAL NOT NULL,
            f1_score REAL NOT NULL,
            submission_id TEXT UNIQUE NOT NULL
        )
    ''')
    
    # Create ground truth table
    c.execute('''
        CREATE TABLE IF NOT EXISTS ground_truth (
            id INTEGER PRIMARY KEY,
            target INTEGER NOT NULL
        )
    ''')
    
    conn.commit()
    return conn

# Initialize database connection
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()

# Load or create ground truth
def load_ground_truth():
    """Load ground truth from database or create if doesn't exist"""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    # Check if ground truth exists
    c.execute("SELECT COUNT(*) FROM ground_truth")
    count = c.fetchone()[0]

    if count == 0:
        # Create sample ground truth data
        test_df = pd.read_csv('gtruth.csv')
        ground_truth = pd.DataFrame({
            'id': list(test_df["id"]),
            'target': list(test_df["target"])
        })
        
        # Save to database
        ground_truth.to_sql('ground_truth', conn, if_exists='replace', index=False)
        return ground_truth
    else:
        # Load from database
        return pd.read_sql("SELECT * FROM ground_truth", conn)

if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = load_ground_truth()

# Database functions
def save_submission(username, description, accuracy, f1_score, submission_id):
    """Save submission to database"""
    conn = st.session_state.db_conn
    c = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        c.execute('''
            INSERT INTO submissions (username, timestamp, description, accuracy, f1_score, submission_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, timestamp, description, accuracy, f1_score, submission_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user_submissions(username):
    """Get all submissions for a user"""
    conn = st.session_state.db_conn
    query = """
        SELECT submission_id, timestamp, description, accuracy, f1_score
        FROM submissions
        WHERE username = ?
        ORDER BY timestamp DESC
    """
    return pd.read_sql(query, conn, params=(username,))

def get_all_submissions():
    """Get all submissions"""
    conn = st.session_state.db_conn
    query = "SELECT * FROM submissions ORDER BY timestamp DESC"
    return pd.read_sql(query, conn)

def get_leaderboard():
    """Get leaderboard with best submission per user"""
    conn = st.session_state.db_conn
    query = """
        SELECT username, MAX(f1_score) as f1_score, 
               MAX(accuracy) as accuracy,
               MAX(timestamp) as timestamp
        FROM submissions
        GROUP BY username
        ORDER BY f1_score DESC
    """
    return pd.read_sql(query, conn)

def get_submission_count():
    """Get total number of submissions"""
    conn = st.session_state.db_conn
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM submissions")
    return c.fetchone()[0]

def get_user_count():
    """Get total number of unique users"""
    conn = st.session_state.db_conn
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT username) FROM submissions")
    return c.fetchone()[0]

# Scoring functions
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def calculate_f1_score(y_true, y_pred):
    # tp = np.sum((y_true == 1) & (y_pred == 1))
    # fp = np.sum((y_true == 0) & (y_pred == 1))
    # fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = precision_score(y_true, y_pred, average="weighted") 
    recall = recall_score(y_true, y_pred, average="weighted")
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def evaluate_submission(submission_df, ground_truth_df):
    """Evaluate submission against ground truth"""
    try:
        # Merge on id
        merged = pd.merge(ground_truth_df, submission_df, on='id', how='inner', suffixes=('_true', '_pred'))
        
        if len(merged) != len(ground_truth_df):
            return None, f"Submission has {len(submission_df)} rows, expected {len(ground_truth_df)}"
        
        y_true = merged['target_true'].values
        y_pred = merged['target_pred'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = calculate_f1_score(y_true, y_pred)
        
        return {'accuracy': accuracy, 'f1_score': f1}, None
    except Exception as e:
        return None, str(e)

# Title and description
st.title("🎯 Prediction Submission System")
st.markdown("Submit your predictions, track your progress, and compete on the leaderboard!")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigation", ["Submit Prediction", "My Submissions", "Leaderboard", "Ground Truth Info"])

# User identification
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    with st.sidebar:
        st.markdown("---")
        st.subheader("👤 Login")
        username = st.text_input("Enter your username:", key="username_input")
        if st.button("Login"):
            if username:
                st.session_state.user_id = username
                st.rerun()
            else:
                st.error("Please enter a username")
else:
    st.sidebar.success(f"👤 Logged in as: **{st.session_state.user_id}**")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.rerun()

# Main content based on page selection
if page == "Submit Prediction":
    st.header("📤 Submit Your Prediction")
    
    if st.session_state.user_id is None:
        st.warning("⚠️ Please login in the sidebar to submit predictions.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Submission Format
            Your CSV file should contain exactly two columns:
            - `id`: Integer identifier for each prediction
            - `target`: Predicted value (int)
            
            Example:
            ```
            id,target
            1,0
            2,1
            3,5
            ...
            ```
            """)
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    submission_df = pd.read_csv(uploaded_file)
                    
                    st.subheader("Preview of your submission:")
                    st.dataframe(submission_df.head(10), use_container_width=True)
                    
                    # Validate format
                    if list(submission_df.columns) != ['id', 'target']:
                        st.error("❌ Invalid format! Columns must be exactly: 'id' and 'target'")
                    else:
                        st.success("✅ Format is correct!")
                        
                        # Add submission description
                        description = st.text_area("Submission description (optional):", 
                                                   placeholder="e.g., Random Forest with feature engineering v2")
                        
                        if st.button("🚀 Submit and Evaluate", type="primary"):
                            with st.spinner("Evaluating your submission..."):
                                scores, error = evaluate_submission(submission_df, st.session_state.ground_truth)
                                
                                if error:
                                    st.error(f"❌ Error: {error}")
                                else:
                                    # Generate submission ID
                                    submission_id = hashlib.md5(
                                        f"{st.session_state.user_id}{datetime.now()}".encode()
                                    ).hexdigest()[:8]
                                    
                                    # Save to database
                                    success = save_submission(
                                        st.session_state.user_id,
                                        description if description else "No description",
                                        scores['accuracy'],
                                        scores['f1_score'],
                                        submission_id
                                    )
                                    
                                    if success:
                                        st.success("✅ Submission saved successfully!")
                                        
                                        # Display results
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Accuracy", f"{scores['accuracy']:.4f}")
                                        with col_b:
                                            st.metric("F1 Score", f"{scores['f1_score']:.4f}")
                                        
                                        st.info(f"📋 Submission ID: `{submission_id}`")
                                        st.balloons()
                                    else:
                                        st.error("❌ Error saving submission. Please try again.")
                                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            user_subs = get_user_submissions(st.session_state.user_id)
            
            if len(user_subs) > 0:
                st.metric("Your Submissions", len(user_subs))
                best_f1 = user_subs['f1_score'].max()
                st.metric("Your Best F1 Score", f"{best_f1:.4f}")
                best_acc = user_subs['accuracy'].max()
                st.metric("Your Best Accuracy", f"{best_acc:.4f}")
            else:
                st.info("No submissions yet")

elif page == "My Submissions":
    st.header("📋 My Submission History")
    
    if st.session_state.user_id is None:
        st.warning("⚠️ Please login in the sidebar to view your submissions.")
    else:
        user_submissions = get_user_submissions(st.session_state.user_id)
        
        if len(user_submissions) == 0:
            st.info("You haven't made any submissions yet.")
        else:
            # Display table
            st.dataframe(
                user_submissions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                    "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f")
                }
            )
            
            # Show improvement chart
            if len(user_submissions) > 1:
                st.subheader("📈 Performance Over Time")
                chart_df = user_submissions.sort_values('timestamp', ascending=True).copy()
                chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(chart_df.set_index('timestamp')['accuracy'])
                    st.caption("Accuracy over submissions")
                with col2:
                    st.line_chart(chart_df.set_index('timestamp')['f1_score'])
                    st.caption("F1 Score over submissions")
            
            # Summary stats
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Submissions", len(user_submissions))
            with col2:
                st.metric("Best F1 Score", f"{user_submissions['f1_score'].max():.4f}")
            with col3:
                st.metric("Average F1 Score", f"{user_submissions['f1_score'].mean():.4f}")
            with col4:
                st.metric("Improvement", f"{(user_submissions['f1_score'].max() - user_submissions['f1_score'].min()):.4f}")

elif page == "Leaderboard":
    st.header("🏆 Leaderboard")
    
    submission_count = get_submission_count()
    
    if submission_count == 0:
        st.info("No submissions yet. Be the first to submit!")
    else:
        leaderboard = get_leaderboard()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Participants", get_user_count())
        with col2:
            st.metric("Total Submissions", submission_count)
        with col3:
            st.metric("Best Score", f"{leaderboard['f1_score'].max():.4f}")
        
        st.markdown("---")
        
        # Display leaderboard
        st.subheader("Top Performers")
        
        # Add rank
        leaderboard.insert(0, 'Rank', range(1, len(leaderboard) + 1))
        
        # Format display
        display_df = leaderboard[['Rank', 'username', 'f1_score', 'accuracy', 'timestamp']].copy()
        display_df.columns = ['Rank', 'Username', 'F1 Score', 'Accuracy', 'Last Submission']
        
        # Highlight current user
        def highlight_user(row):
            if st.session_state.user_id and row['Username'] == st.session_state.user_id:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_user, axis=1).format({
                'F1 Score': '{:.4f}',
                'Accuracy': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Show your rank if logged in
        if st.session_state.user_id:
            user_rank = leaderboard[leaderboard['username'] == st.session_state.user_id]
            if len(user_rank) > 0:
                rank = user_rank.index[0] + 1
                st.info(f"🎯 Your current rank: **#{rank}** out of {len(leaderboard)} participants")

else:  # Ground Truth Info
    st.header("ℹ️ Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        st.metric("Number of Samples", len(st.session_state.ground_truth))
        st.metric("Number of Features", 1)
        
        # Class distribution
        st.subheader("Target Distribution")
        class_counts = st.session_state.ground_truth['target'].value_counts()
        st.bar_chart(class_counts)
        
        # Show distribution percentages
        total = len(st.session_state.ground_truth)
        st.write(f"Class 0: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.1f}%)")
        st.write(f"Class 1: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.1f}%)")
        st.write(f"Class 5: {class_counts.get(5, 0)} ({class_counts.get(5, 0)/total*100:.1f}%)")
        st.write("...")
    with col2:
        st.subheader("Sample IDs")
        st.write(f"IDs range from {st.session_state.ground_truth['id'].min()} to {st.session_state.ground_truth['id'].max()}")
        
        st.subheader("Preview")
        st.dataframe(st.session_state.ground_truth.head(10), use_container_width=True)
    


# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Higher F1 Score is better. Focus on balancing precision and recall!")

# Database info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    total_subs = get_submission_count()
    total_users = get_user_count()
    st.write(f"Total submissions: **{total_subs}**")
    st.write(f"Total users: **{total_users}**")