import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from surprise import SVD

# Set page title and layout
st.set_page_config(page_title="Yelp Recommendation System", layout="wide")

# Custom CSS for a cute appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #484848;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .recommendation {
        background-color: #FFE4E1;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Yelp Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Find your next favorite place in Phoenix!</p>", unsafe_allow_html=True)

# Sidebar for model loading
with st.sidebar:
    st.markdown("### Model Settings")
    
    # Option to load model
    model_option = st.radio(
        "Choose model loading option:",
        ("Load pre-trained model", "Train new model (demo)")
    )
    
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                if model_option == "Load pre-trained model":
                    # Load the pre-trained model
                    model_file = st.sidebar.file_uploader("Upload model file", type=['pkl'])
                    if model_file:
                        model = pickle.load(model_file)
                        st.session_state['model'] = model
                        st.success("Model loaded successfully!")
                else:
                    # Create a simple demo model
                    model = SVD(n_factors=50, n_epochs=10)
                    st.session_state['model'] = model
                    st.success("Demo model created!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

# Main content area
tab1, tab2 = st.tabs(["Recommend", "Explore Data"])

with tab1:
    st.markdown("<h2 class='sub-header'>Get Recommendations</h2>", unsafe_allow_html=True)
    
    # User selection
    col1, col2 = st.columns(2)
    
    with col1:
        # In a real app, you'd load these from your dataset
        sample_users = ["A", "Aaron", "Adam", "Alex", "Amy", "Andrew", "Ashley", "Bob", "Brian", "Chris"]
        selected_user = st.selectbox("Select a user:", sample_users)
    
    with col2:
        # Categories from the dataset
        categories = ["Restaurants", "Shopping", "Beauty & Spas", "Health & Medical", "Home Services"]
        selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    
    # Location filter
    locations = ["Phoenix", "Scottsdale", "Tempe", "Chandler", "Gilbert", "Glendale", "Mesa"]
    selected_location = st.multiselect("Filter by location:", locations, default=["Phoenix"])
    
    # Rating filter
    min_rating = st.slider("Minimum rating:", 1.0, 5.0, 3.0, 0.5)
    
    if st.button("Get Recommendations"):
        if 'model' in st.session_state:
            with st.spinner("Finding recommendations..."):
                # Simulate recommendations (in a real app, you'd use your model)
                # This is just for demonstration
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"### Top Recommendations for {selected_user}")
                
                # Simulate model predictions
                businesses = [
                    {"name": "Pacific Seafood Buffet", "stars": 4.5, "categories": ["Buffets", "Restaurants"], "city": "Chandler"},
                    {"name": "Oregano's Pizza Bistro", "stars": 4.2, "categories": ["Pizza", "Italian", "Restaurants"], "city": "Gilbert"},
                    {"name": "Phoenicia Grill", "stars": 4.7, "categories": ["Mediterranean", "Restaurants"], "city": "Phoenix"},
                    {"name": "Hon Machi", "stars": 4.1, "categories": ["Sushi Bars", "Japanese", "Restaurants"], "city": "Tempe"},
                    {"name": "Greek Gyro Express", "stars": 3.9, "categories": ["Greek", "Restaurants"], "city": "Scottsdale"}
                ]
                
                # Filter by category if needed
                if selected_category != "All":
                    businesses = [b for b in businesses if selected_category in b["categories"]]
                
                # Filter by location
                if selected_location:
                    businesses = [b for b in businesses if b["city"] in selected_location]
                
                # Filter by rating
                businesses = [b for b in businesses if b["stars"] >= min_rating]
                
                # Sort by predicted rating
                businesses = sorted(businesses, key=lambda x: x["stars"], reverse=True)
                
                if businesses:
                    for i, business in enumerate(businesses, 1):
                        st.markdown(f"""
                        <div class='recommendation'>
                            <h4>{i}. {business['name']} - {business['stars']}⭐</h4>
                            <p>Categories: {', '.join(business['categories'])}</p>
                            <p>Location: {business['city']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations found matching your criteria. Try adjusting your filters.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please load a model first!")

with tab2:
    st.markdown("<h2 class='sub-header'>Explore Yelp Data</h2>", unsafe_allow_html=True)
    
    # Sample data exploration
    st.markdown("### Business Distribution by City")
    
    # Create sample data for visualization
    cities = ["Phoenix", "Scottsdale", "Tempe", "Chandler", "Gilbert", "Glendale", "Mesa"]
    business_counts = [450, 320, 280, 230, 180, 150, 120]
    
    city_df = pd.DataFrame({
        "City": cities,
        "Business Count": business_counts
    })
    
    st.bar_chart(city_df.set_index("City"))
    
    # Category distribution
    st.markdown("### Business Categories")
    
    categories = ["Restaurants", "Shopping", "Beauty & Spas", "Health & Medical", "Home Services", "Automotive", "Nightlife"]
    category_counts = [520, 380, 250, 220, 180, 150, 120]
    
    category_df = pd.DataFrame({
        "Category": categories,
        "Count": category_counts
    })
    
    st.bar_chart(category_df.set_index("Category"))
    
    # Rating distribution
    st.markdown("### Rating Distribution")
    
    ratings = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    rating_counts = [50, 80, 120, 200, 350, 450, 520, 380, 250]
    
    rating_df = pd.DataFrame({
        "Rating": ratings,
        "Count": rating_counts
    })
    
    st.line_chart(rating_df.set_index("Rating"))

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center'>Built with Streamlit and Collaborative Filtering</p>", unsafe_allow_html=True)
