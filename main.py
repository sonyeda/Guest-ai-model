from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
import os

app = FastAPI('5492062267d446ef604e77b4495550013e28a71c18c2547d4cf72e00bc1fa6d4')

# MongoDB Connection
MONGO_URI = ("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

try:
    client = MongoClient('mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    db = client["hotel_booking"]
    bookings_collection = db["bookings"]
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

# Pydantic model for data validation
class Booking(BaseModel):
    customer_id: int
    name: str
    age: int
    check_in_date: str
    check_out_date: str
    preferred_cuisine: str
    booked_through_points: bool
    number_of_stayers: int
    special_requests: str | None = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Booking API!"}

# Endpoint to create a new booking
@app.post("/create_booking/")
def create_booking(booking: Booking):
    try:
        booking_dict = booking.dict()
        result = bookings_collection.insert_one(booking_dict)
        return {"message": "Booking created successfully!", "booking_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get all bookings
@app.get("/bookings/", response_model=List[Booking])
def get_bookings():
    try:
        bookings = list(bookings_collection.find({}, {"_id": 0}))
        return bookings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get a booking by customer ID
@app.get("/booking/{customer_id}", response_model=Booking)
def get_booking(customer_id: int):
    try:
        booking = bookings_collection.find_one({"customer_id": customer_id}, {"_id": 0})
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        return booking
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
