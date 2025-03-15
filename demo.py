import streamlit as st
from datetime import date
import pandas as pd
import random
from pymongo import MongoClient
import os
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
from together import Together

# MongoDB Connection
client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["hotel_guests"]
new_bookings_collection = db["new_bookings"]

# Helper function to load Excel files
def load_excel_file(filename, rename_cols=None):
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename, engine='openpyxl')
            if rename_cols:
                df.rename(columns=rename_cols, inplace=True)
            return df
        else:
            st.error(f"❌ File not found: {filename}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {str(e)}")
        return pd.DataFrame()

# Initialize Pinecone
def init_pinecone():
    # Attempt to retrieve the API key from environment variables or Streamlit secrets
    pc_api_key = os.getenv("5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4")  # Use a valid environment variable name
    if not pc_api_key:
        try:
            pc_api_key = st.secrets["5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4"]  # Try retrieving from Streamlit secrets
        except KeyError:
            pass  # Handle missing secrets below

    # If still not found, prompt the user to input the API key
    if not pc_api_key:
        pc_api_key = st.text_input("Enter Pinecone API Key:", type="password", key="pinecone_api_key_input")
        if not pc_api_key:
            st.error("❌ Pinecone API Key is required to proceed.")
            st.stop()

    # Store the API key in session state for reuse
    st.session_state.pc_api_key = pc_api_key

    # Initialize Pinecone with the API key
    return Pinecone(api_key=st.session_state.pc_api_key)

# Main App
st.set_page_config(page_title="Hotel Management System", layout="wide")
st.title("🏨 Hotel Management System")

# Navigation
menu = ["Submit Booking", "Analyze Sentiment"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Submit Booking":
    st.header("🏨 Hotel Booking Form")

    # Customer ID handling
    has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))

    customer_id = st.text_input("Enter your Customer ID", "") if has_customer_id == "Yes" else str(random.randint(10001, 99999))
    if has_customer_id == "No":
        st.write(f"Your generated Customer ID: {customer_id}")

    # Form fields
    with st.form("booking_form"):
        name = st.text_input("Enter your name", "")
        checkin_date = st.date_input("Check-in Date", min_value=date.today())
        checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
        age = st.number_input("Enter your age", 18, 120, 30)
        stayers = st.number_input("Number of Guests", 1, 3, 1)
        preferred_cuisine = st.selectbox("Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
        preferred_booking = st.selectbox("Book through points?", ["Yes", "No"])
        special_requests = st.text_area("Special Requests", "")

        submitted = st.form_submit_button("Submit Booking")

        if submitted:
            if not name or not customer_id:
                st.warning("⚠️ Please provide name and Customer ID")
            else:
                try:
                    # Create booking record
                    booking_data = {
                        'customer_id': int(customer_id),
                        'name': name,
                        'check_in_date': pd.to_datetime(checkin_date),
                        'check_out_date': pd.to_datetime(checkout_date),
                        'age': age,
                        'number_of_stayers': stayers,
                        'preferred_cuisine': preferred_cuisine,
                        'booked_through_points': 1 if preferred_booking == "Yes" else 0,
                        'special_requests': special_requests
                    }

                    # Insert into MongoDB
                    new_bookings_collection.insert_one(booking_data)
                    st.success("✅ Booking Confirmed Successfully!")
                    
                    # Load ML model and make predictions (simplified version)
                    # [Include your ML prediction logic here if needed]

                except Exception as e:
                    st.error(f"❌ Error submitting booking: {str(e)}")

elif choice == "Analyze Sentiment":
    st.header("📊 Customer Sentiment Analysis")
    
    # Initialize Pinecone on first run
    if "pc_api_key" not in st.session_state:
        init_pinecone()

    # Sentiment analysis form
    with st.form("sentiment_form"):
        query = st.text_input("Enter your query", "Staff behavior feedback")
        start_date = st.date_input("Start Date", date(2024, 1, 1))
        end_date = st.date_input("End Date", date(2024, 12, 31))
        max_rating = st.slider("Maximum Rating", 1, 5, 3)
        analyze = st.form_submit_button("Analyze")

        if analyze:
            # Load reviews data
            df = load_excel_file("reviews_data.xlsx")
            if df.empty:
                st.error("❌ No reviews data found")
                st.stop()

            # Initialize components
            embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
            pc = init_pinecone()
            index = pc.Index("hotel-reviews")
            
            # Process query
            query_embedding = embeddings.embed_query(query)
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=5,
                filter={
                    "Rating": {"$lte": max_rating},
                    "review_date": {
                        "$gte": int(start_date.strftime("%Y%m%d")),
                        "$lte": int(end_date.strftime("%Y%m%d"))
                    }
                },
                include_metadata=True
            )

            # Process results
            if not results.matches:
                st.warning("⚠️ No matching reviews found")
            else:
                review_ids = [m.metadata["review_id"] for m in results.matches]
                filtered_reviews = df[df["review_id"].isin(review_ids)]
                
                # Generate summary with LLM
                client = Together(api_key=st.session_state.pc_api_key)  # Use the stored API key
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Vision-Free",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize sentiment about: '{query}' from these reviews: {filtered_reviews['Review'].tolist()}"
                    }]
                )
                
                st.subheader("Sentiment Summary")
                st.write(response.choices[0].message.content)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Hotel Management System v1.0")
