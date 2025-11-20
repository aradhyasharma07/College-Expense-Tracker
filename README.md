📘 College Expense Tracker — Smart Budgeting App for Students

A modern, interactive financial-tracking web application built using Python, Streamlit, and Machine Learning.
Designed specifically for students to track expenses, analyze spending habits, manage budgets, save goals, and forecast future spending.

🚀 Features
✅ Expense Recording

Add expenses with date, category, description, amount, and payment method

Instant data insertion

Input validation and non-negative checks

📊 Interactive Dashboard

Total spent

Daily average

Most frequent category

Recent transaction logs

30-day spending analysis

Category & payment method distribution

💰 Budget Management

Category-wise monthly budget limits

Budget vs. Actual comparison

Remaining balance indicators

“Within Budget” / “Over Budget” status tags

📈 Analytics Module

Monthly & weekly spending trends

Category-wise analysis

Comparison charts

Payment method statistics

🤖 Machine Learning Predictions

Built using RandomForestRegressor

Predicts the next 30 days of spending

Shows actual vs. predicted graphs

Accuracy metrics (MAE, R²)

🎯 Savings Goal Tracker

Create and manage multiple goals

Timeframe-based progress

Automatic progress bar calculation

📂 Data Import / Export

Upload CSV files

Column validation and preview




| Component       | Technology                           |
| --------------- | ------------------------------------ |
| Frontend        | Streamlit + Custom CSS               |
| Backend         | Python                               |
| Data Handling   | Pandas, NumPy                        |
| Visualizations  | Plotly, Matplotlib, Seaborn          |
| ML Model        | RandomForestRegressor (Scikit-Learn) |
| Date Operations | datetime                             |




College-Expense-Tracker/
│
├── app.py                 # Main Streamlit application
├── assets/                # Images, icons, CSS
├── data/                  # Default or sample CSV files
├── model/                 # ML model scripts
├── requirements.txt       # Dependencies
└── README.md              # Documentation





🔧 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/college-expense-tracker.git
cd college-expense-tracker

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py

🧠 Machine Learning Model

The Random Forest model uses:

Day of week

Day of month

Month

Daily aggregated spending

It predicts:

Next 30 days spending

Trend visualization

Error metrics like MAE and R²

📝 Future Improvements

User authentication

Cloud database integration (Firebase or MongoDB)

PDF report exports

Mobile-friendly UI

AI financial recommendations

🙌 Author

Aradhya Sharma
CSE-AIFT, Semester 3
Chitkara University, Rajpura

Faculty Mentor: Mr. Mudrik Kaushik

⭐ Support the Project

If you like this project, please ⭐ star the repo to show support!

Clean and edit data inside the app

Export updated dataset as CSV
